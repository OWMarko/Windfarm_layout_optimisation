"""Réseau triangulaire orientable, centré sur la boîte du terrain."""

import numpy as np

from .placement_grille import DECIMALES


def generer_noeuds(bornes_inferieures, bornes_superieures, pas_m, angle_deg):
    """Retourne les nœuds au centimètre et le centre de la boîte."""
    if not np.isfinite([pas_m, angle_deg]).all() or pas_m <= 0:
        raise ValueError("L'angle doit être fini et le pas fini et positif.")
    borne_inf = np.asarray(bornes_inferieures, dtype=float)
    borne_sup = np.asarray(bornes_superieures, dtype=float)
    if (borne_inf.shape != (2,) or borne_sup.shape != (2,)
            or not np.isfinite([borne_inf, borne_sup]).all()
            or np.any(borne_sup <= borne_inf)):
        raise ValueError("La boîte du terrain doit avoir deux bornes finies et ordonnées.")

    angle = np.deg2rad(angle_deg)
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    base = pas_m * np.array([[1, 0.5], [0, np.sqrt(3) / 2]])
    reseau = rotation @ base
    origine = (borne_inf + borne_sup) / 2

    # Un nœud peut entrer dans la boîte après arrondi.
    demi_unite = 0.5 * 10 ** (-DECIMALES)
    bas = borne_inf - demi_unite
    haut = borne_sup + demi_unite
    coins = np.array([[bas[0], bas[1]], [bas[0], haut[1]],
                      [haut[0], bas[1]], [haut[0], haut[1]]])
    coins_indices = np.linalg.solve(reseau, (coins - origine).T).T
    i_min, j_min = np.floor(coins_indices.min(axis=0)).astype(int)
    i_max, j_max = np.ceil(coins_indices.max(axis=0)).astype(int)

    indices = np.array([(i, j) for i in range(i_min, i_max + 1)
                        for j in range(j_min, j_max + 1)])
    noeuds = origine + indices @ reseau.T
    return np.round(noeuds, DECIMALES), origine
