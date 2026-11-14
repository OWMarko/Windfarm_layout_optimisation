"""Raffinement CMA-ES avec pénalités géométriques."""

import numpy as np
from .glouton_aep import BudgetAEPAtteint, _donnees_geometriques

def _penalite_geometrique(x, diametre, zone):
    """Pénalités graduées, appliquées dès une violation stricte AMON."""
    import constraints as cst

    xs, ys = x[::2], x[1::2]
    if not cst.checking_same_coords(xs, ys):
        return 1e12
    placement = float(cst.placing_constraint(xs, ys, zone))
    if placement > 0:
        return 1e9 + placement * 1e5
    espacement = float(cst.spacing_constraint_min(xs, ys, diametre))
    if espacement > 0:
        return 1e8 + espacement * 1e5
    raise ValueError("L'évaluateur a renvoyé None pour un parc admissible.")


def raffiner_cmaes(x0, aep0, donnees, evaluer, *, graine=42, sigma0=5.0,
                   limite_propositions=6400, notifier_generation=None, population=None):
    """Raffine un départ connu avec pycma ; aucune réévaluation implicite de x0.

evaluer(x) est appelé pour chaque proposition. Il retourne une AEP, None
pour un rejet géométrique, ou lève BudgetAEPAtteint. Les générations incomplètes
ne passent pas à tell(). Le meilleur faisable est conservé indépendamment
de la moyenne interne du solveur. Les compteurs physiques restent chez l'appelant.
population=None conserve le défaut historique de pycma ; un entier impose popsize.
    """
    import cma

    nombre, diametre, borne_inf, borne_sup, zone = _donnees_geometriques(donnees)
    x = np.asarray(x0, dtype=float)
    if (x.shape != (2 * nombre,) or not np.isfinite(x).all()
            or not np.isfinite(aep0) or aep0 < 0
            or not np.isfinite(sigma0) or sigma0 <= 0
            or isinstance(limite_propositions, bool)
            or int(limite_propositions) != limite_propositions or limite_propositions < 1):
        raise ValueError("Départ, AEP, sigma ou limite de propositions invalides.")
    if population is not None and (isinstance(population, bool)
            or int(population) != population or population < 2):
        raise ValueError("La population CMA-ES doit être un entier supérieur ou égal à deux.")
    import constraints as cst
    if (not cst.checking_same_coords(x[::2], x[1::2])
            or cst.placing_constraint(x[::2], x[1::2], zone) != 0
            or cst.spacing_constraint_min(x[::2], x[1::2], diametre) != 0):
        raise ValueError("Le départ CMA-ES doit être admissible.")
    if np.any(x < np.tile(borne_inf, nombre)) or np.any(x > np.tile(borne_sup, nombre)):
        raise ValueError("Le départ est hors des bornes de la boîte.")
    options = {
        "bounds": [np.tile(borne_inf, nombre).tolist(), np.tile(borne_sup, nombre).tolist()],
        "maxfevals": int(limite_propositions), "seed": int(graine),
        "verbose": -1, "verb_disp": 0, "verb_log": 0,
    }
    if population is not None:
        options["popsize"] = int(population)
    solveur = cma.CMAEvolutionStrategy(x.tolist(), float(sigma0), options)
    meilleur_x, meilleure_aep = x.tolist(), float(aep0)
    propositions, rejets, generations = 0, 0, 0
    arret = "solveur"
    couts = []
    try:
        while not solveur.stop():
            solutions = solveur.ask()
            couts = []
            for solution in solutions:
                if propositions >= limite_propositions:
                    arret = "limite_propositions"
                    break
                candidat = [float(v) for v in solution]
                valeur = evaluer(candidat)
                propositions += 1
                if valeur is None:
                    rejets += 1
                    cout = _penalite_geometrique(candidat, diametre, zone)
                else:
                    if not np.isfinite(valeur) or valeur < 0:
                        raise ValueError("AEP invalide renvoyée au solveur CMA-ES.")
                    aep = float(valeur)
                    cout = -aep * 1e6
                    if aep > meilleure_aep:
                        meilleur_x, meilleure_aep = candidat, aep
                couts.append(cout)
            if len(couts) != len(solutions):
                break
            solveur.tell(solutions, couts)
            generations += 1
            # Une génération n'est achevée qu'après la mise à jour effective.
            couts = []
            if notifier_generation is not None:
                notifier_generation(generations, propositions)
            if propositions >= limite_propositions:
                arret = "limite_propositions"
                break
    except BudgetAEPAtteint:
        arret = "budget_aep"
    return {
        "methode": "cmaes", "version_cma": cma.__version__, "graine": int(graine),
        "sigma0_m": float(sigma0), "population": int(solveur.popsize),
        "propositions_traitees": propositions, "rejets_geometriques": rejets,
        "generations_completes": generations, "generations_partielles": int(bool(couts)),
        "propositions_generation_partielle": len(couts), "arret": arret,
        "arrets_solveur": {str(k): str(v) for k, v in solveur.stop().items()},
        "meilleur_x": meilleur_x, "meilleure_aep_gwh": meilleure_aep,
    }
