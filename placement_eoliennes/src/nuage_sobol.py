"""Construction Sobol du pilote comparer_nuages, conservée sans son lanceur."""

import numpy as np
from scipy.stats import qmc

from .placement_coffee_house import filtrer_candidats, selectionner_configuration


def construire_configuration(donnees, graine_nuage=66, graine_selecteur=166):
    """1 024 points Sobol brouillés, puis filtrage et maximin à un point initial."""
    _, nombre, diametre, borne_inf, borne_sup, zone = donnees
    unitaires = qmc.Sobol(d=2, scramble=True, rng=graine_nuage).random_base2(10)
    candidats = filtrer_candidats(qmc.scale(unitaires, borne_inf, borne_sup), zone)
    return selectionner_configuration(candidats, nombre, diametre,
                                       np.random.default_rng(graine_selecteur), 1)
