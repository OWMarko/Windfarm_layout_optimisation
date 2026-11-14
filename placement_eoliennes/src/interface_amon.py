"""Accès aux instances, à la géométrie et au simulateur AMON."""

import math
from multiprocessing import cpu_count
from pathlib import Path
import sys


RACINE_PROJET = Path(__file__).resolve().parents[1]
RACINE_AMON = RACINE_PROJET.parent / "AMON-main"


def fichier_instance(numero_instance):
    return RACINE_AMON / "instances" / str(numero_instance) / "param.txt"


def lire_parametres(fichier_parametres):
    """Résout les fichiers de données par rapport au dossier AMON."""
    if not (RACINE_AMON / "blackbox.py").is_file():
        raise RuntimeError(f"AMON introuvable : {RACINE_AMON}.")
    if str(RACINE_AMON) not in sys.path:
        sys.path.insert(0, str(RACINE_AMON))
    from data import read_param_file

    nombre, diametre, hauteur, echelle, *fichiers = read_param_file(str(fichier_parametres))
    chemins = [str(RACINE_AMON / fichier) if fichier != "na" else fichier for fichier in fichiers]
    return nombre, diametre, hauteur, echelle, *chemins


def charger_instance(numero_instance, nombre_eoliennes=None):
    """Charge la géométrie du site et l'effectif demandé."""
    fichier = fichier_instance(numero_instance)
    nombre, diametre, _, echelle, _, limites, exclusions, _, _ = lire_parametres(fichier)
    import constraints as contraintes_amon
    from windfarm_setting import terrain_setting

    borne_inf, borne_sup, limites, exclusions = terrain_setting(limites, exclusions, scale_factor=echelle)
    zone = contraintes_amon.buildable_zone(limites, exclusions)[0]
    if nombre_eoliennes is not None:
        nombre = nombre_eoliennes
    return fichier, nombre, diametre, borne_inf, borne_sup, zone


def verifier_configuration(configuration, nombre_eoliennes, diametre, zone):
    """Vérifie le vecteur AMON, les doublons et les deux contraintes."""
    import constraints as contraintes_amon
    import numpy as np

    valeurs = np.asarray(configuration, dtype=float)
    if (not isinstance(configuration, list) or valeurs.shape != (2 * nombre_eoliennes,)
            or not np.isfinite(valeurs).all()):
        raise ValueError("X doit être une liste de 2 N coordonnées finies entrelacées.")
    x, y = configuration[0::2], configuration[1::2]
    if not contraintes_amon.checking_same_coords(x, y):
        raise ValueError("Deux turbines ont des coordonnées identiques ou quasi identiques.")
    espacement = float(contraintes_amon.spacing_constraint_min(x, y, diametre))
    placement = float(contraintes_amon.placing_constraint(x, y, zone))
    if (espacement, placement) != (0, 0):
        raise ValueError(f"Contraintes invalides : espacement={espacement}, placement={placement}.")
    return espacement, placement


def preparer_evaluation(fichier_parametres, dossier):
    """Prépare le modèle et les séries de vent une fois pour plusieurs placements."""
    _, diametre, hauteur, _, courbe, _, _, vitesse, direction = lire_parametres(fichier_parametres)
    import matplotlib
    from windfarm_setting import site_setting

    matplotlib.use("Agg")
    modele, vitesses, directions, _, _ = site_setting(courbe, diametre, hauteur, vitesse, direction, str(dossier))
    if not vitesses.index.equals(directions.index):
        raise ValueError("Les séries de vitesse et de direction du vent ne sont pas appariées.")
    return modele, vitesses.iloc[:, 0], directions.iloc[:, 0]


def evaluer_aep(configuration, site):
    """Évalue un placement déjà validé. L'AEP renvoyée est en GWh."""
    import numpy as np

    modele, vitesses, directions = site
    # Même découpage temporel que blackbox.aep_func, sans pool de processus.
    resultat = modele(configuration[0::2], configuration[1::2], ws=vitesses, wd=directions,
                      time=True, n_cpu=1, wd_chunks=cpu_count())
    contributions = resultat.aep()
    # La somme xarray peut ignorer les NaN et masquer une simulation invalide.
    if not np.isfinite(np.asarray(contributions)).all():
        raise RuntimeError("Contribution AEP non finie.")
    aep = float(contributions.sum())
    if not math.isfinite(aep) or aep < 0:
        raise RuntimeError(f"AEP invalide : {aep}.")
    return aep


def evaluer_aep_ponderee(configuration, modele, vitesses, directions, poids):
    """Annualise les puissances PyWake sur des états de vent pondérés."""
    import numpy as np

    vitesses = np.asarray(vitesses, dtype=float)
    directions = np.asarray(directions, dtype=float)
    poids = np.asarray(poids, dtype=float)
    if (vitesses.ndim != 1 or vitesses.size == 0 or vitesses.shape != directions.shape
            or vitesses.shape != poids.shape
            or not all(np.isfinite(valeurs).all() for valeurs in (vitesses, directions, poids))
            or np.any(poids < 0) or not np.isclose(poids.sum(), 1.0)):
        raise ValueError("Les états de vent doivent être appariés et leurs poids normalisés.")

    resultat = modele(configuration[0::2], configuration[1::2], ws=vitesses, wd=directions,
                      time=True, n_cpu=1, wd_chunks=1)
    if not np.array_equal(resultat.ws.to_numpy(), vitesses) or not np.array_equal(resultat.wd.to_numpy(), directions):
        raise RuntimeError("PyWake a modifié les états de vent associés aux poids.")
    puissance = resultat.Power.sum(dim="wt", skipna=False).to_numpy()
    if puissance.shape != poids.shape or not np.isfinite(puissance).all() or np.any(puissance < 0):
        raise RuntimeError("Puissance du parc invalide pour les états de vent pondérés.")
    # Puissance en W, 8 760 h par an et conversion de Wh en GWh.
    return float((24 * 365) * 1e-9 * np.dot(poids, puissance))
