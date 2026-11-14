"""Grille géométrique orientable, avec marge et coordonnées au centimètre."""

import numpy as np


MARGE_M = 1.0
DECIMALES = 2


def regler_pas(diametre, pas_x=None, pas_y=None):
    """Éloigne les pas du seuil strict 2D ; conserve les demandes dans les résultats."""
    if not np.isfinite(diametre) or diametre <= 0:
        raise ValueError("Le diamètre doit être fini et positif.")
    pas_x = 4 * diametre if pas_x is None else pas_x
    pas_y = 2 * diametre if pas_y is None else pas_y
    if not np.isfinite([pas_x, pas_y]).all() or min(pas_x, pas_y) < 2 * diametre:
        raise ValueError("Les pas demandés doivent être finis et au moins égaux à 2D.")
    return {
        "pas_x_demande_m": pas_x, "pas_y_demande_m": pas_y,
        "pas_x_m": max(pas_x, 2 * diametre + MARGE_M),
        "pas_y_m": max(pas_y, 2 * diametre + MARGE_M),
        "marge_securite_m": MARGE_M, "decimales_coordonnees_m": DECIMALES,
    }


def generer_noeuds(
    bornes_inferieures, bornes_superieures, pas_x, pas_y,
    angle_deg, decalage_x=0.0, decalage_y=0.0,
):
    """Couvre la boîte du terrain avec un réseau tourné autour de son centre."""
    valeurs = [pas_x, pas_y, angle_deg, decalage_x, decalage_y]
    if not np.isfinite(valeurs).all() or pas_x <= 0 or pas_y <= 0:
        raise ValueError("Les paramètres de grille doivent être finis et les pas positifs.")
    borne_inf = np.asarray(bornes_inferieures, dtype=float)
    borne_sup = np.asarray(bornes_superieures, dtype=float)
    if (borne_inf.shape != (2,) or borne_sup.shape != (2,)
            or not np.isfinite([borne_inf, borne_sup]).all()
            or np.any(borne_sup <= borne_inf)):
        raise ValueError("La boîte du terrain doit avoir deux bornes finies et ordonnées.")

    angle = np.deg2rad(angle_deg)
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    origine = (borne_inf + borne_sup) / 2 + [decalage_x, decalage_y]
    coins = np.array([
        [borne_inf[0], borne_inf[1]], [borne_inf[0], borne_sup[1]],
        [borne_sup[0], borne_inf[1]], [borne_sup[0], borne_sup[1]],
    ])
    # Vecteurs lignes : multiplication par R pour la transformation inverse.
    coins_locaux = (coins - origine) @ rotation
    m_min = int(np.floor(coins_locaux[:, 0].min() / pas_x)) - 1
    m_max = int(np.ceil(coins_locaux[:, 0].max() / pas_x)) + 1
    n_min = int(np.floor(coins_locaux[:, 1].min() / pas_y)) - 1
    n_max = int(np.ceil(coins_locaux[:, 1].max() / pas_y)) + 1

    noeuds = []
    for m in range(m_min, m_max + 1):
        for n in range(n_min, n_max + 1):
            point = origine + rotation @ np.array([m * pas_x, n * pas_y])
            noeuds.append(point)
    return np.round(np.asarray(noeuds), DECIMALES), origine


def generer_configuration(
    nombre_eoliennes, diametre, bornes_inferieures, bornes_superieures,
    zone_constructible, graine, pas_x=None, pas_y=None, angle_deg=0.0,
    decalage_x=0.0, decalage_y=0.0,
):
    """Filtre le réseau avec AMON puis sélectionne N_T positions par maximin."""
    import constraints as contraintes_amon
    from .placement_coffee_house import selectionner_configuration

    parametres = regler_pas(diametre, pas_x, pas_y)
    noeuds, origine = generer_noeuds(
        bornes_inferieures, bornes_superieures, parametres["pas_x_m"], parametres["pas_y_m"],
        angle_deg, decalage_x, decalage_y,
    )
    candidats = []
    # Le filtre porte sur les coordonnées déjà arrondies qui seront renvoyées.
    for x, y in noeuds:
        if contraintes_amon.placing_constraint([x], [y], zone_constructible) == 0:
            candidats.append([x, y])
    candidats = np.unique(np.asarray(candidats, dtype=float).reshape(-1, 2), axis=0)
    configuration = selectionner_configuration(
        candidats, nombre_eoliennes, diametre, np.random.default_rng(graine)
    )
    informations = {
        **parametres, "orientation": "geometrique_AMON", "angle_deg": angle_deg,
        "decalage_x_m": decalage_x, "decalage_y_m": decalage_y,
        "origine_grille_x_m": float(origine[0]), "origine_grille_y_m": float(origine[1]),
        "nombre_noeuds_grille": len(noeuds), "nombre_candidats_admissibles": len(candidats),
        "nombre_initiales": 1,
    }
    return configuration, informations
