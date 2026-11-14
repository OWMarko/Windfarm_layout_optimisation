"""Stratégie élitiste (mu + lambda), à mutations d'une seule turbine.

Le cours optimisation5, pages 3–5, présente notamment (mu, lambda).
La sélection dans parents ET enfants est ici une adaptation, sans garantie
d'optimalité globale pour le problème contraint et le budget fini.
"""

from math import sqrt
from time import perf_counter

import numpy as np

from .optimisation import cle_parc


def evoluer(parents, aep_parents, essai, diametre, *, mu=4, lambda_enfants=4,
            budget_total=64, evaluations_initiales=4, bascule_evaluations=28,
            graine_parent=42, graine_mutation=43, limite_propositions=6400,
            propositions_par_enfant=100, journaliser=None):
    """Reçoit une population déjà évaluée ET imputée à l'essai.

    ``essai.evaluer`` valide, déduplique et impute ; None signifie rejet
    géométrique. Une panne de simulation est propagée, jamais pénalisée.
    Les parents restent figés jusqu'à la sélection de fin de génération.
    """
    if (mu < 1 or lambda_enfants < 1 or len(parents) != mu or len(aep_parents) != mu
            or evaluations_initiales < mu or budget_total < evaluations_initiales
            or essai.imputes != evaluations_initiales or diametre <= 0
            or not np.isfinite(diametre) or limite_propositions < 1
            or propositions_par_enfant < 1):
        raise ValueError("Population, budget ou paramètres de mutation invalides.")
    taille = len(parents[0])
    if taille < 2 or taille % 2 or any(len(x) != taille for x in parents):
        raise ValueError("Les parents doivent avoir les mêmes 2N coordonnées.")
    if (not np.isfinite(np.asarray(parents)).all()
            or not np.isfinite(aep_parents).all() or np.any(np.asarray(aep_parents) < 0)
            or len({cle_parc(x) for x in parents}) != mu):
        raise ValueError("Les parents doivent être distincts, finis et évalués.")
    population = [(list(x), float(aep)) for x, aep in zip(parents, aep_parents)]
    choisir = np.random.default_rng(graine_parent)
    muter = np.random.default_rng(graine_mutation)
    generations, completes, partielles, temps_selection = 0, 0, 0, 0.0
    propositions_initiales = essai.propositions
    historique_generations = []
    arret = "budget_evaluations"
    while essai.imputes < budget_total and not essai.budget_atteint:
        if essai.propositions - propositions_initiales >= limite_propositions:
            arret = "limite_propositions"
            break
        generations += 1
        fixes = [(x.copy(), aep) for x, aep in population]
        enfants = []
        propositions_generation = 0
        while (len(enfants) < lambda_enfants and essai.imputes < budget_total
               and not essai.budget_atteint
               and propositions_generation < propositions_par_enfant * lambda_enfants
               and essai.propositions - propositions_initiales < limite_propositions):
            indice_parent = int(choisir.integers(mu))
            turbine = int(muter.integers(taille // 2))
            supplementaires = essai.imputes - evaluations_initiales
            sigma = diametre / (sqrt(2) * (1 if supplementaires < bascule_evaluations else 2))
            deplacement = sigma * muter.normal(size=2)
            candidat = fixes[indice_parent][0].copy()
            candidat[2*turbine] += float(deplacement[0])
            candidat[2*turbine+1] += float(deplacement[1])
            avant = essai.imputes
            propositions_generation += 1
            valeur = essai.evaluer(candidat, phase="raffinement", generation=generations,
                parent=indice_parent + 1, turbine=turbine + 1, sigma_m=sigma,
                norme_deplacement_m=float(np.linalg.norm(deplacement)))
            if valeur is not None and essai.imputes > avant:
                enfants.append((candidat, float(valeur)))
        debut_selection = perf_counter()
        # Une clé géométrique lexicographique départage les égalités sans
        # information sur les AEP futures ni préférence cachée pour une graine.
        distincts = {}
        for x, valeur in fixes + enfants:
            distincts.setdefault(cle_parc(x), (x, valeur))
        population = sorted(distincts.values(), key=lambda p: (-p[1], cle_parc(p[0])))[:mu]
        temps_selection += perf_counter() - debut_selection
        complete = len(enfants) == lambda_enfants
        completes += int(complete)
        partielles += int(not complete)
        # La sélection parents + enfants vient réellement de se terminer.
        essai.generations_completes = completes
        trace = dict(generation=generations, enfants_evalues=len(enfants),
                     propositions=propositions_generation, complete=complete,
                     budget_impute=essai.imputes, meilleure_aep_gwh=population[0][1],
                     parents=[x for x, _ in fixes], population=[x for x, _ in population])
        historique_generations.append(trace)
        if journaliser is not None:
            journaliser(trace)
        if not enfants:
            arret = "generation_sans_enfant_nouveau"
            break
    return dict(arret=arret, generations=generations, generations_completes=completes,
                generations_partielles=partielles, population=[x for x, _ in population],
                aep_population=[aep for _, aep in population], temps_selection_s=temps_selection,
                historique_generations=historique_generations)
