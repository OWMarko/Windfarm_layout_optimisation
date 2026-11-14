"""Placement par ensemble indépendant glouton dans un graphe de conflits."""

import numpy as np

from .placement_coffee_house import generer_candidats


def construire_conflits(candidats, diametre):
    """Une arête relie deux positions distantes de strictement moins de 2D."""
    if not np.isfinite(diametre) or diametre <= 0:
        raise ValueError("Le diamètre doit être fini et positif.")
    candidats = np.asarray(candidats, dtype=float)
    if candidats.ndim != 2 or candidats.shape[1] != 2 or not np.isfinite(candidats).all():
        raise ValueError("Les candidats doivent former un tableau fini de taille M × 2.")
    nombre = len(candidats)
    conflits = np.zeros((nombre, nombre), dtype=bool)
    for indice in range(nombre - 1):
        distances = np.linalg.norm(candidats[indice + 1:] - candidats[indice], axis=1)
        voisins = distances < 2 * diametre
        conflits[indice, indice + 1:] = voisins
        conflits[indice + 1:, indice] = voisins
    return conflits


def selectionner_configuration(candidats, conflits, nombre_eoliennes):
    """Recalcule les degrés sur le graphe courant et choisit leur minimum."""
    if nombre_eoliennes < 1:
        raise ValueError("Le nombre d'éoliennes doit être positif.")
    disponibles = np.arange(len(candidats))
    selection = []
    for _ in range(nombre_eoliennes):
        if len(disponibles) == 0:
            raise RuntimeError(
                f"Graphe épuisé après {len(selection)} turbines pour {nombre_eoliennes} demandées."
            )
        graphe_courant = conflits[np.ix_(disponibles, disponibles)]
        degres = graphe_courant.sum(axis=1)
        # Premier indice parmi les ex æquo : ordre lexicographique du nuage partagé.
        indice = int(disponibles[np.argmin(degres)])
        selection.append(indice)
        disponibles = disponibles[(disponibles != indice) & ~conflits[indice, disponibles]]
    return np.asarray(candidats)[selection].reshape(-1).tolist()


def generer_configuration(
    nombre_eoliennes, diametre, bornes_inferieures, bornes_superieures,
    zone_constructible, graine, nombre_candidats=1000,
):
    """Réutilise le nuage admissible, construit les conflits et place N_T turbines."""
    if nombre_candidats < 1:
        raise ValueError("Le nombre de candidats doit être positif.")
    candidats = generer_candidats(
        nombre_candidats, bornes_inferieures, bornes_superieures,
        zone_constructible, np.random.default_rng(graine),
    )
    if len(candidats) < nombre_eoliennes:
        raise RuntimeError(f"Seulement {len(candidats)} candidats admissibles pour {nombre_eoliennes} turbines.")
    conflits = construire_conflits(candidats, diametre)
    configuration = selectionner_configuration(candidats, conflits, nombre_eoliennes)
    informations = {"nombre_candidats_admissibles": len(candidats),
                    "nombre_aretes": int(np.count_nonzero(conflits) // 2),
                    "departage": "premier_indice_lexicographique"}
    return configuration, informations
