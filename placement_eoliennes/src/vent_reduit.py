"""Réduction d'une série de vent en 64 états observés et pondérés."""

import numpy as np


NOMBRE_SECTEURS = 16
NOMBRE_STRATES = 4


def reduire_vent(vitesses, directions):
    """Renvoie les vitesses, directions et poids des 64 états représentatifs.

    Les secteurs sont centrés sur les multiples de 22,5° et le secteur nord
    est à cheval sur 0°. Dans chaque secteur, les observations triées par
    vitesse sont réparties dans quatre strates d'effectifs presque égaux.
    Chaque strate est représentée par le vecteur observé le plus proche de
    son vecteur moyen. Les ex æquo gardent l'ordre de la strate.
    """
    vitesses = np.asarray(vitesses, dtype=float)
    directions = np.asarray(directions, dtype=float)
    if vitesses.ndim != 1 or vitesses.shape != directions.shape or not vitesses.size:
        raise ValueError("Les séries de vitesse et de direction doivent être appariées et non vides.")
    if not np.isfinite(vitesses).all() or not np.isfinite(directions).all():
        raise ValueError("Les observations de vent doivent être finies.")
    if np.any(vitesses < 0) or np.any((directions < 0) | (directions >= 360)):
        raise ValueError("Vitesses positives ou nulles et directions dans [0, 360[ attendues.")

    largeur = 360 / NOMBRE_SECTEURS
    bornes = (np.arange(NOMBRE_SECTEURS) + 0.5) * largeur
    secteurs = np.searchsorted(bornes, directions, side="right") % NOMBRE_SECTEURS
    effectifs = np.bincount(secteurs, minlength=NOMBRE_SECTEURS)
    if np.any(effectifs < NOMBRE_STRATES):
        raise ValueError(f"Il faut au moins {NOMBRE_STRATES} observations par secteur. Effectifs : {effectifs.tolist()}.")

    ordre = np.lexsort((vitesses, secteurs))
    groupes = np.split(ordre, np.cumsum(effectifs)[:-1])
    radians = np.deg2rad(directions)
    vecteurs = -vitesses[:, None] * np.column_stack((np.sin(radians), np.cos(radians)))

    representants, tailles_strates = [], []
    for groupe in groupes:
        for strate in np.array_split(groupe, NOMBRE_STRATES):
            vents = vecteurs[strate]
            ecarts = vents - vents.mean(axis=0)
            representants.append(strate[np.argmin(np.sum(ecarts**2, axis=1))])
            tailles_strates.append(len(strate))

    poids = np.asarray(tailles_strates, dtype=float) / len(vitesses)
    return vitesses[representants], directions[representants], poids
