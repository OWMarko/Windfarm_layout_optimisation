"""Poids énergétiques sectoriels : extraction de lancer/analyser_vent."""

import numpy as np

from .interface_amon import lire_parametres


def poids_sillage(fichier_parametres):
    """36 secteurs, interpolation de puissance isolée, aucun appel de parc."""
    _, _, _, _, courbe, _, _, vitesse, direction = lire_parametres(fichier_parametres)
    from data import read_csv_wind_data

    ws, wd = read_csv_wind_data(vitesse, direction)
    if len(ws) == 0 or not ws.index.equals(wd.index) or not ws.index.is_unique:
        raise ValueError("Séries vides, non appariées ou dates dupliquées.")
    vitesses, directions = ws.to_numpy(dtype=float), wd.to_numpy(dtype=float)
    dates = np.asarray(ws.index, dtype="datetime64[s]")
    if (not np.isfinite(vitesses).all() or not np.isfinite(directions).all()
            or np.any(vitesses < 0) or np.any((directions < 0) | (directions >= 360))
            or np.isnat(dates).any() or np.any(np.diff(dates) <= np.timedelta64(0, "s"))):
        raise ValueError("Séries de vent invalides.")
    table = np.loadtxt(courbe, delimiter=";", skiprows=1)
    # Même prolongement que le protocole : zéro sous la table, aucun au-dessus.
    puissance = np.interp(vitesses, table[:, 0], table[:, 1] * 1000, left=0.0, right=np.nan)
    if not np.isfinite(puissance).all() or np.any(puissance < 0):
        raise ValueError("Puissance isolée invalide ou vitesse hors table.")
    secteurs = np.floor(((directions + 5) % 360) / 10).astype(int)
    poids = np.bincount(secteurs, weights=puissance, minlength=36)
    if poids.sum() <= 0:
        raise ValueError("L'énergie isolée doit être positive.")
    return np.arange(0, 360, 10), poids / poids.sum(), len(vitesses)
