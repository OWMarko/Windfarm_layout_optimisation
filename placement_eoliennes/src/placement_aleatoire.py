"""Placement aléatoire séquentiel avec rejet."""

import numpy as np

import constraints as contraintes_amon


def generer_configuration(
    nombre_eoliennes,
    diametre,
    bornes_inferieures,
    bornes_superieures,
    zone_constructible,
    graine,
    nombre_essais_max,
    *, essais_totaux_max=None, mesures=None,
):
    """Place successivement les éoliennes en respectant les contraintes AMON."""
    generateur = np.random.default_rng(graine)
    coordonnees_x = []
    coordonnees_y = []
    if essais_totaux_max is not None and essais_totaux_max < 1:
        raise ValueError("Le plafond total de tirages doit être positif.")
    tirages = 0

    for indice_eolienne in range(nombre_eoliennes):
        for _ in range(nombre_essais_max):
            if essais_totaux_max is not None and tirages >= essais_totaux_max:
                raise RuntimeError(
                    f"Plafond total de {essais_totaux_max} tirages atteint après "
                    f"{len(coordonnees_x)} turbines."
                )
            tirages += 1
            if mesures is not None:
                mesures["tirages"] = tirages
            x = generateur.uniform(bornes_inferieures[0], bornes_superieures[0])
            y = generateur.uniform(bornes_inferieures[1], bornes_superieures[1])

            placement = contraintes_amon.placing_constraint([x], [y], zone_constructible)
            if placement != 0:
                continue

            x_candidats = coordonnees_x + [x]
            y_candidats = coordonnees_y + [y]
            coordonnees_distinctes = contraintes_amon.checking_same_coords(
                x_candidats, y_candidats
            )
            espacement = contraintes_amon.spacing_constraint_min(
                x_candidats, y_candidats, diametre
            )

            if coordonnees_distinctes and espacement == 0:
                coordonnees_x.append(x)
                coordonnees_y.append(y)
                break
        else:
            raise RuntimeError(
                f"Impossible de placer l'éolienne {indice_eolienne + 1} "
                f"après {nombre_essais_max} essais."
            )

    configuration = []
    for x, y in zip(coordonnees_x, coordonnees_y):
        configuration.extend([float(x), float(y)])

    return configuration
