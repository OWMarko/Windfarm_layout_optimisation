"""Placement maximin glouton sur un nuage fini de candidats admissibles."""

import numpy as np


def generer_candidats(
    nombre_candidats,
    bornes_inferieures,
    bornes_superieures,
    zone_constructible,
    generateur,
):
    """Tire un nuage dans la boîte du terrain, puis le filtre avec AMON."""
    nuage = generateur.uniform(
        bornes_inferieures, bornes_superieures, size=(nombre_candidats, 2)
    )
    return filtrer_candidats(nuage, zone_constructible)


def filtrer_candidats(nuage, zone_constructible):
    """Filtre les points avec AMON et renvoie les candidats distincts dans un ordre fixe."""
    import constraints as contraintes_amon

    candidats_admissibles = []
    for x, y in nuage:
        placement = contraintes_amon.placing_constraint([x], [y], zone_constructible)
        if placement == 0:
            candidats_admissibles.append([x, y])

    # unique retire les doublons et fixe l'ordre lexicographique des candidats.
    candidats = np.asarray(candidats_admissibles, dtype=float).reshape(-1, 2)
    return np.unique(candidats, axis=0)


def selectionner_configuration(
    candidats, nombre_eoliennes, diametre, generateur, nombre_initiales=1
):
    """Sélectionne exactement N_T points d'un nuage admissible par maximin."""
    if nombre_eoliennes < 1:
        raise ValueError("Le nombre d'éoliennes doit être positif.")
    if not 1 <= nombre_initiales <= nombre_eoliennes:
        raise ValueError("Le nombre de turbines initiales doit être compris entre 1 et N_T.")
    if not np.isfinite(diametre) or diametre <= 0:
        raise ValueError("Le diamètre doit être fini et strictement positif.")

    if len(candidats) < nombre_eoliennes:
        raise RuntimeError(
            f"Seulement {len(candidats)} candidats distincts admissibles pour "
            f"{nombre_eoliennes} turbines."
        )

    indices_selectionnes = []
    distances_minimales = np.full(len(candidats), np.inf)
    for _ in range(nombre_eoliennes):
        admissibles = np.flatnonzero(distances_minimales >= 2 * diametre)
        if len(admissibles) == 0:
            raise RuntimeError(
                f"Sélection maximin bloquée après {len(indices_selectionnes)} turbines : "
                "aucun candidat restant ne respecte la séparation 2D."
            )

        if len(indices_selectionnes) < nombre_initiales:
            indice = int(generateur.choice(admissibles))
        else:
            indice = int(np.argmax(distances_minimales))
        indices_selectionnes.append(indice)

        # Invariant : distances_minimales[p] = min des distances à S_k.
        distances = np.linalg.norm(candidats - candidats[indice], axis=1)
        distances_minimales = np.minimum(distances_minimales, distances)

    configuration = []
    for indice in indices_selectionnes:
        x, y = candidats[indice]
        configuration.extend([float(x), float(y)])
    return configuration


def generer_configuration(
    nombre_eoliennes,
    diametre,
    bornes_inferieures,
    bornes_superieures,
    zone_constructible,
    graine,
    nombre_candidats=5000,
    nombre_initiales=1,
):
    """Construit le même nuage aléatoire puis appelle la sélection partagée."""
    if nombre_candidats < 1:
        raise ValueError("Le nombre de candidats doit être positif.")
    generateur = np.random.default_rng(graine)
    candidats = generer_candidats(
        nombre_candidats, bornes_inferieures, bornes_superieures,
        zone_constructible, generateur,
    )
    return selectionner_configuration(
        candidats, nombre_eoliennes, diametre, generateur, nombre_initiales
    )
