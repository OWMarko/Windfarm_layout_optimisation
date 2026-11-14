"""Score géométrique d'interaction par paires pour présélectionner des parcs."""

import numpy as np


def directions_avales(directions_deg):
    """Convertit un vent météorologique (origine) en vecteur aval est–nord."""
    radians = np.deg2rad(directions_deg)
    return np.column_stack((-np.sin(radians), -np.cos(radians)))


def score_par_paires(configuration, aval, poids, *, sigma0_m, kappa, longueur_m):
    """Somme des contributions des paires, pondérées par secteur de vent."""
    points = np.asarray(configuration, dtype=float).reshape(-1, 2)
    i, j = np.triu_indices(len(points), k=1)
    return float(contributions_par_paires(points[j] - points[i], aval, poids,
                 sigma0_m=sigma0_m, kappa=kappa, longueur_m=longueur_m).sum())


def contributions_par_paires(vecteurs, aval, poids, *, sigma0_m, kappa, longueur_m):
    """Renvoie la contribution de chaque paire dans chaque secteur de vent."""
    if sigma0_m <= 0 or kappa < 0 or longueur_m <= 0:
        raise ValueError("Le proxy demande sigma0 et L positifs, et kappa positif ou nul.")
    vecteurs = np.asarray(vecteurs, dtype=float).reshape(-1, 2)
    transverse = np.column_stack((-aval[:, 1], aval[:, 0]))
    # Dans une paire non perpendiculaire, seul le sens aval contribue.
    s = np.abs(vecteurs @ aval.T)
    t = vecteurs @ transverse.T
    largeur = sigma0_m + kappa * s
    contributions = np.exp(-0.5 * (t / largeur) ** 2 - s / longueur_m)
    # Écarte les zéros géométriques perturbés par l'arrondi trigonométrique.
    contributions[s <= 1e-9] = 0.0
    return contributions * poids


def descendre_sillage(configuration, candidats, diametre, aval, poids, parametres, graine,
                     *, limite_propositions=240, journaliser=None):
    """Échanges aléatoires sur un nuage filtré, acceptés si R diminue strictement."""
    from time import perf_counter

    debut = perf_counter()
    courant = np.asarray(configuration, dtype=float).reshape(-1, 2).copy()
    candidats = np.asarray(candidats, dtype=float).reshape(-1, 2)
    indices_par_point = {tuple(point): i for i, point in enumerate(candidats)}
    if len(indices_par_point) != len(candidats):
        raise ValueError("Le nuage doit contenir des points distincts.")
    if any(tuple(point) not in indices_par_point for point in courant):
        raise ValueError("Le parc de départ doit appartenir au nuage.")
    indices = np.array([indices_par_point[tuple(point)] for point in courant])
    vus = {tuple(sorted(indices))}
    generateur = np.random.default_rng(graine)
    score = score_par_paires(courant.ravel(), aval, poids, **parametres)
    initial = score
    acceptes, rejets, doublons = 0, 0, 0
    erreur_delta = None
    propositions = 0
    for proposition in range(1, limite_propositions + 1):
        libres = np.setdiff1d(np.arange(len(candidats)), indices, assume_unique=True)
        if len(libres) == 0:
            break
        propositions = proposition
        turbine = int(generateur.integers(len(courant)))
        destination = int(generateur.choice(libres))
        point = candidats[destination]
        conserves = np.delete(courant, turbine, axis=0)
        ligne = dict(proposition=proposition, turbine=turbine + 1, indice_nuage=destination,
                     score_avant=score, delta_local="", score_courant=score)
        if np.any(np.linalg.norm(conserves - point, axis=1) < 2 * diametre):
            rejets += 1
            ligne["statut"] = "rejet_espacement"
        else:
            nouveaux_indices = indices.copy()
            nouveaux_indices[turbine] = destination
            cle = tuple(sorted(nouveaux_indices))
            if cle in vus:
                doublons += 1
                ligne["statut"] = "doublon"
            else:
                vus.add(cle)
                ancien = contributions_par_paires(conserves - courant[turbine], aval, poids, **parametres).sum()
                nouveau = contributions_par_paires(conserves - point, aval, poids, **parametres).sum()
                delta = float(nouveau - ancien)
                ligne["delta_local"] = delta
                essai = courant.copy()
                essai[turbine] = point
                if erreur_delta is None or delta < 0:
                    complet = score_par_paires(essai.ravel(), aval, poids, **parametres)
                    if erreur_delta is None:
                        erreur_delta = abs((complet - score) - delta)
                        if not np.isclose(complet - score, delta, rtol=1e-10, atol=1e-12):
                            raise ValueError("Le delta local diffère du recalcul complet.")
                if delta < 0 and complet < score:
                    courant, indices, score = essai, nouveaux_indices, complet
                    acceptes += 1
                    ligne["statut"] = "accepte"
                else:
                    ligne["statut"] = "refus_score"
                ligne["score_courant"] = score
        if journaliser is not None:
            journaliser(ligne)
    return courant.ravel().tolist(), dict(
        score_initial=initial, score_final=score, variation_score=score - initial,
        propositions=propositions, echanges_acceptes=acceptes, rejets_espacement=rejets,
        doublons=doublons, erreur_controle_delta=erreur_delta,
        arret="plafond_propositions" if propositions == limite_propositions else "nuage_occupe",
        temps_raffinement_s=perf_counter() - debut,
    )


def score_sillage(configuration, angles_deg, poids, diametre, *, sigma0_d=0.5, kappa=0.05, longueur_d=10.0):
    """Calcule le score du pilote avec des longueurs exprimées en diamètres."""
    coordonnees = np.asarray(configuration, dtype=float)
    angles = np.asarray(angles_deg, dtype=float)
    poids = np.asarray(poids, dtype=float)
    if coordonnees.ndim != 1 or coordonnees.size < 2 or coordonnees.size % 2:
        raise ValueError("Il faut une liste de coordonnées x, y entrelacées.")
    if angles.ndim != 1 or not angles.size or angles.shape != poids.shape:
        raise ValueError("Il faut un poids pour chaque angle de vent.")
    if not all(np.isfinite(tableau).all() for tableau in (coordonnees, angles, poids)) or np.any(poids < 0):
        raise ValueError("Les entrées doivent être finies et les poids positifs ou nuls.")
    if (not np.isfinite([diametre, sigma0_d, kappa, longueur_d]).all()
            or min(diametre, sigma0_d, longueur_d) <= 0 or kappa < 0):
        raise ValueError("D, sigma0_d et longueur_d doivent être positifs, et kappa positif ou nul.")
    return score_par_paires(coordonnees, directions_avales(angles % 360), poids,
                           sigma0_m=sigma0_d * diametre, kappa=kappa, longueur_m=longueur_d * diametre)
