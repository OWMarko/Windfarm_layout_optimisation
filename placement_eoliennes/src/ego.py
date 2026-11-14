"""Prédiction gaussienne et amélioration espérée pour maximiser l'AEP."""

from math import erfc, pi, sqrt

import numpy as np


def _matern_52(x, y, longueur):
    distances = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=2)
    rayon = sqrt(5) * distances / longueur
    return (1 + rayon + rayon**2 / 3) * np.exp(-rayon)


def amelioration_esperee(x_observe, aep_observe, x_candidat, longueur=0.25, regularisation=1e-8):
    """Renvoie la moyenne, l'écart type latent et l'amélioration espérée en GWh.

    Les deux matrices de paramètres sont sur la même échelle normalisée.
    La régularisation s'applique à la diagonale de la covariance des observations.
    """
    x_observe = np.asarray(x_observe, dtype=float)
    x_candidat = np.asarray(x_candidat, dtype=float)
    aep_observe = np.asarray(aep_observe, dtype=float)
    if (x_observe.ndim != 2 or x_candidat.ndim != 2 or 0 in x_observe.shape
            or x_observe.shape[1] != x_candidat.shape[1]
            or aep_observe.shape != (x_observe.shape[0],)):
        raise ValueError("Attendu deux matrices de même dimension et une AEP par point observé.")
    if not all(np.isfinite(valeurs).all() for valeurs in (x_observe, x_candidat, aep_observe)):
        raise ValueError("Les paramètres et les AEP doivent être finis.")
    if not np.isfinite(longueur) or longueur <= 0:
        raise ValueError("La longueur de corrélation doit être finie et strictement positive.")
    if not np.isfinite(regularisation) or regularisation < 0:
        raise ValueError("La régularisation doit être finie et positive ou nulle.")

    # Centrer d'abord sur la première AEP préserve un signal exactement constant.
    ecarts = aep_observe - aep_observe[0]
    centre = ecarts.mean()
    echelle = ecarts.std()
    if echelle == 0:
        echelle = 1.0  # Échelle de 1 GWh pour conserver l'exploration.
    observations = (ecarts - centre) / echelle

    covariance = _matern_52(x_observe, x_observe, longueur)
    covariance += regularisation * np.eye(len(x_observe))
    cholesky = np.linalg.cholesky(covariance)
    covariance_croisee = _matern_52(x_observe, x_candidat, longueur)
    projection = np.linalg.solve(cholesky, covariance_croisee)
    moyenne = projection.T @ np.linalg.solve(cholesky, observations)
    variance = 1 - np.sum(projection**2, axis=0)
    ecart_type = np.sqrt(np.maximum(variance, 0))

    gain = moyenne - observations.max()
    ei = np.maximum(gain, 0)
    incertain = ecart_type > 0
    z = gain[incertain] / ecart_type[incertain]
    densite = np.exp(-0.5 * z**2) / sqrt(2 * pi)
    repartition = np.fromiter((erfc(-valeur / sqrt(2)) / 2 for valeur in z), dtype=float, count=z.size)
    ei[incertain] = gain[incertain] * repartition + ecart_type[incertain] * densite

    return aep_observe[0] + centre + echelle * moyenne, echelle * ecart_type, echelle * np.maximum(ei, 0)
