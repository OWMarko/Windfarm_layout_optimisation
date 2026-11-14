"""Déplacements axiaux aléatoires avec acceptation gloutonne ou de Metropolis."""

from math import exp, log
from time import perf_counter

import numpy as np

from .interface_amon import verifier_configuration
from .optimisation import cle_parc


def trajectoire(depart, aep_depart, donnees, evaluer, journaliser, *, regle,
                budget=6, limite_propositions=240, echeance=float("inf"),
                graine_proposition=42, graine_acceptation=43):
    """Conserve séparément le parc courant et le meilleur parc rencontré."""
    if regle not in ("glouton", "recuit"):
        raise ValueError("Règle d'acceptation inconnue.")
    _, nombre, diametre, _, _, zone = donnees
    verifier_configuration(depart, nombre, diametre, zone)
    courant, meilleur = depart.copy(), depart.copy()
    aep_courante = meilleure_aep = float(aep_depart)
    propositions = np.random.default_rng(graine_proposition)
    acceptations = np.random.default_rng(graine_acceptation)
    vus = {cle_parc(depart)}
    imputees = 0
    arret = "plafond_propositions"
    for rang in range(1, limite_propositions + 1):
        if imputees >= budget:
            arret = "budget_voisins"
            break
        if perf_counter() >= echeance:
            arret = "limite_temps"
            break
        pas = diametre if imputees < 3 else diametre / 2
        turbine = int(propositions.integers(nombre))
        axe = int(propositions.integers(2))
        signe = int(propositions.choice((-1, 1)))
        candidat = courant.copy()
        candidat[2 * turbine + axe] += signe * pas
        temperature = (0.10 / log(2)) * 0.9 ** imputees
        ligne = dict(proposition=rang, turbine=turbine + 1, axe="xy"[axe], signe=signe,
                     pas_m=pas, j_temperature=imputees, temperature_gwh=temperature,
                     aep_avant_gwh=aep_courante, evaluation_imputee=0, doublon_interne=0,
                     accepte=0, baisse_acceptee=0, nouveau_meilleur=0)
        try:
            verifier_configuration(candidat, nombre, diametre, zone)
        except ValueError as erreur:
            ligne.update(statut="rejet_geometrique", message=str(erreur))
        else:
            cle = cle_parc(candidat)
            ligne["doublon_interne"] = int(cle in vus)
            ligne["evaluation_imputee"] = int(cle not in vus)
            vus.add(cle)
            imputees += ligne["evaluation_imputee"]
            resultat = evaluer(candidat)
            ligne.update(resultat)
            if resultat["statut"] == "evalue":
                delta = resultat["aep_gwh"] - aep_courante
                probabilite = float(delta > 0) if regle == "glouton" else min(1.0, exp(min(0.0, delta / temperature)))
                u = None
                accepte = delta > 0
                if regle == "recuit":
                    accepte = delta >= 0
                    if delta < 0:
                        u = float(acceptations.random())
                        accepte = u < probabilite
                ligne.update(delta_gwh=delta, probabilite_acceptation=probabilite, tirage_acceptation=u,
                             accepte=int(accepte), baisse_acceptee=int(accepte and delta < 0))
                if accepte:
                    courant, aep_courante = candidat, resultat["aep_gwh"]
                if resultat["aep_gwh"] > meilleure_aep:
                    meilleur, meilleure_aep = candidat.copy(), resultat["aep_gwh"]
                    ligne["nouveau_meilleur"] = 1
        ligne.update(budget_impute=imputees, aep_courante_gwh=aep_courante, meilleure_aep_gwh=meilleure_aep)
        journaliser(ligne)
    if imputees >= budget:
        arret = "budget_voisins"
    return dict(courant=courant, aep_courante_gwh=aep_courante, meilleur=meilleur,
                meilleure_aep_gwh=meilleure_aep, evaluations_imputees=imputees, arret=arret)


def recuit_simule(depart, aep_depart, essai, diametre, *, budget_total=64,
                     evaluations_initiales=4, bascule_evaluations=28,
                     graine_proposition=42, graine_acceptation=43,
                     temperature_initiale=0.10 / log(2),
                     refroidissement=(1/20)**(1/59), limite_propositions=6400,
                     journaliser=None):
    """Recuit sous budget ; les départs sont déjà imputés.

    Le calendrier dépend des évaluations supplémentaires autorisées. Un doublon ou
    un rejet n'avance pas ce calendrier. Le meilleur et le courant sont distincts.
    """
    if (len(depart) < 2 or len(depart) % 2 or not np.isfinite(depart).all()
            or not np.isfinite(aep_depart) or aep_depart < 0
            or essai.imputes != evaluations_initiales or budget_total < evaluations_initiales
            or diametre <= 0 or not np.isfinite(diametre)
            or temperature_initiale <= 0 or not np.isfinite(temperature_initiale)
            or not 0 < refroidissement <= 1 or limite_propositions < 1):
        raise ValueError("Départ ou paramètres du recuit invalides.")
    courant, meilleur = list(depart), list(depart)
    aep_courante = meilleure_aep = float(aep_depart)
    proposer = np.random.default_rng(graine_proposition)
    accepter = np.random.default_rng(graine_acceptation)
    acceptes = baisses = 0
    temps_selection = 0.0
    propositions_initiales = essai.propositions
    arret = "budget_evaluations"
    while essai.imputes < budget_total and not essai.budget_atteint:
        if essai.propositions - propositions_initiales >= limite_propositions:
            arret = "limite_propositions"
            break
        j = essai.imputes - evaluations_initiales
        pas = diametre if j < bascule_evaluations else diametre / 2
        temperature = temperature_initiale * refroidissement**j
        turbine = int(proposer.integers(len(courant) // 2))
        axe, signe = int(proposer.integers(2)), int(proposer.choice((-1, 1)))
        candidat = courant.copy()
        candidat[2*turbine + axe] += signe * pas
        avant = essai.imputes
        valeur = essai.evaluer(candidat, phase="raffinement", turbine=turbine+1,
            axe="xy"[axe], signe=signe, pas_m=pas, j_temperature=j,
            temperature_gwh=temperature, aep_avant_gwh=aep_courante)
        if valeur is None:
            continue
        debut_selection = perf_counter()
        delta = float(valeur) - aep_courante
        probabilite = 1.0 if delta >= 0 else exp(delta / temperature)
        u = None if delta >= 0 else float(accepter.random())
        accepte = delta >= 0 or u < probabilite
        if accepte:
            courant, aep_courante = candidat, float(valeur)
            acceptes += 1
            baisses += int(delta < 0)
        if valeur > meilleure_aep:
            meilleur, meilleure_aep = candidat.copy(), float(valeur)
        temps_selection += perf_counter() - debut_selection
        if journaliser is not None:
            journaliser(dict(budget_impute=essai.imputes, j_temperature=j,
                evaluation_imputee=int(essai.imputes > avant),
                temperature_gwh=temperature, pas_m=pas, delta_gwh=delta,
                probabilite_acceptation=probabilite, tirage_acceptation=u,
                accepte=accepte, aep_courante_gwh=aep_courante,
                meilleure_aep_gwh=meilleure_aep))
    return dict(arret=arret, courant=courant, aep_courante_gwh=aep_courante,
                meilleur=meilleur, meilleure_aep_gwh=meilleure_aep,
                acceptes=acceptes, baisses_acceptees=baisses,
                temps_selection_s=temps_selection)
