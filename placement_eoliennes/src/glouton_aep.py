"""Construction gloutonne par ajouts évalués avec AMON."""

import numpy as np


class BudgetAEPAtteint(Exception):
    """Arrêt demandé par l'évaluateur quand son budget est épuisé."""


def _donnees_geometriques(donnees):
    _, nombre, diametre, borne_inf, borne_sup, zone = donnees
    if isinstance(nombre, bool) or int(nombre) != nombre or nombre < 1:
        raise ValueError("L'effectif doit être un entier strictement positif.")
    bornes = np.asarray([borne_inf, borne_sup], dtype=float)
    if (bornes.shape != (2, 2) or not np.isfinite(bornes).all()
            or np.any(bornes[1] <= bornes[0])
            or not np.isfinite(diametre) or diametre <= 0):
        raise ValueError("Bornes ou diamètre invalides.")
    return int(nombre), float(diametre), bornes[0], bornes[1], zone


def candidats_glouton(donnees, pas_facteur=2.5):
    """Réseau ancré en borne inférieure, ordre x puis y, sans arrondi."""
    import constraints as cst

    _, diametre, borne_inf, borne_sup, zone = _donnees_geometriques(donnees)
    if not np.isfinite(pas_facteur) or pas_facteur <= 0:
        raise ValueError("Le facteur de pas doit être strictement positif.")
    pas = diametre * pas_facteur
    # La faisabilité exige une violation exactement nulle.
    candidats = [(float(x), float(y))
                 for x in np.arange(borne_inf[0], borne_sup[0], pas)
                 for y in np.arange(borne_inf[1], borne_sup[1], pas)
                 if cst.placing_constraint([x], [y], zone) == 0]
    return np.asarray(candidats, dtype=float).reshape(-1, 2)


def construire_glouton_aep(donnees, evaluer, pas_facteur=2.5):
    """Construit N turbines en évaluant tous les ajouts admissibles à chaque étape.

evaluer(x) reçoit des parcs partiels et renvoie une AEP complète finie.
Il doit compter les appels, vérifier le parc et lever BudgetAEPAtteint si
nécessaire. L'exception se propage : aucun parc partiel n'est annoncé terminé.
    """
    nombre, diametre, _, _, _ = _donnees_geometriques(donnees)
    candidats = [tuple(p) for p in candidats_glouton(donnees, pas_facteur)]
    nombre_candidats = len(candidats)
    if nombre_candidats < nombre:
        raise ValueError(f"Seulement {nombre_candidats} candidats pour {nombre} turbines.")
    places = []
    evaluations = 0
    historique = []
    for etape in range(nombre):
        valides = [p for p in candidats
                   if all(np.hypot(p[0] - q[0], p[1] - q[1]) >= 2 * diametre
                          for q in places)]
        if not valides:
            raise RuntimeError(f"Aucun ajout admissible pour la turbine {etape + 1}.")
        # La première turbine est fixée au premier candidat du réseau.
        examines = valides[:1] if etape == 0 else valides
        meilleur, aep_max = None, -np.inf
        for point in examines:
            x = [float(v) for p in places + [point] for v in p]
            valeur = evaluer(x)
            if valeur is None or not np.isfinite(valeur) or valeur < 0:
                raise ValueError("L'évaluateur a refusé un ajout géométriquement admissible.")
            aep = float(valeur)
            evaluations += 1
            # L'égalité conserve le premier point dans l'ordre du réseau.
            if aep > aep_max:
                meilleur, aep_max = point, aep
        places.append(meilleur)
        candidats.remove(meilleur)
        historique.append({"eoliennes": etape + 1, "aep_gwh": aep_max,
                           "candidats_evalues": len(examines),
                           "evaluations_demandees_cumulees": evaluations})
    return [float(v) for p in places for v in p], {
        "methode": "glouton_aep", "pas_facteur": float(pas_facteur),
        "pas_m": float(pas_facteur * diametre), "candidats": nombre_candidats,
        "evaluations_demandees": evaluations, "aep_gwh": aep_max,
        "historique": historique, "arret": "effectif_atteint",
    }


