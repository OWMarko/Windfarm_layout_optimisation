"""Comparer deux constructions et cinq raffinements avec des départs neufs."""

from math import log
from time import perf_counter, process_time
import numpy as np

from .commun import dossier_sortie, ecrire_csv, sauvegarder
from .evaluation import Evaluateur, Essai, PontUnOpt
from ..src import interface_amon as amon
from ..src.optimisation import cle_parc, raffiner_1opt
from ..src.placement_coffee_house import generer_configuration as coffee_house
from ..src.placement_grille import generer_configuration as grille
from ..src.recuit import recuit_simule
from ..src.evolution import evoluer
from ..src.glouton_aep import candidats_glouton, BudgetAEPAtteint
from ..src.cma_es import raffiner_cmaes
from ..src.score_sillage import descendre_sillage, directions_avales
from ..src.vent import poids_sillage

METHODES = ('1opt', 'recuit', 'evolution', 'sillage', 'cmaes')


def ajouter_arguments(parser):
    parser.add_argument('--instances', nargs='+', type=int, choices=range(1, 7), default=[1])
    parser.add_argument('--effectifs', nargs='+', type=int, default=[30])
    parser.add_argument('--familles', nargs='+', choices=('coffee_house', 'grille'), default=['coffee_house', 'grille'])
    parser.add_argument('--methodes', nargs='+', choices=METHODES, default=list(METHODES))
    parser.add_argument('--budget', type=int, choices=(4, 8, 12), default=12, help='Évaluations par chaîne, dont quatre départs.')
    parser.add_argument('--graine', type=int, default=20261113, help='Entropie des graines Coffee-house et recherches locales.')
    parser.add_argument('--temoin', action='store_true', help='Poursuivre les constructions Coffee-house jusqu’au même budget.')
    parser.add_argument('--sortie', help='Dossier neuf pour CSV, coordonnées et PNG.')


def graines_contexte(graine, instance, n):
    """Flux distincts pour construction, proposition, parent et acceptation."""
    # 576 est un identifiant fixe de flux aléatoire, pas un budget de ce run.
    sequences = np.random.SeedSequence([graine, 576, instance, n]).spawn(44)
    construction = [int(s.generate_state(1)[0]) for s in sequences[:40]]
    flux = {}
    for methode, seq in zip(('coffee_house', '1opt', 'recuit', 'evolution'), sequences[40:]):
        flux[methode] = {nom: int(s.generate_state(1)[0]) for nom, s in
                        zip(('proposition', 'parent', 'acceptation'), seq.spawn(3))}
    return construction, flux


def construire(donnees, famille, graines, nombre, journal, instance):
    parcs, couts, vus = [], [], set()
    propositions = list(zip(graines, [0.0]*len(graines))) if famille == 'coffee_house' else [
        (42, angle) for angle in (30., 75., 120., 165.)]
    for graine, angle in propositions:
        t, cpu = perf_counter(), process_time()
        ligne = dict(instance=instance, n=donnees[1], famille=famille, graine=graine,
                     angle_deg=angle, statut='echec_construction')
        try:
            args = (*donnees[1:5], donnees[5], graine)
            x = coffee_house(*args, 5000, 1) if famille == 'coffee_house' else grille(
                *args, pas_x=320., pas_y=161., angle_deg=angle)[0]
            amon.verifier_configuration(x, donnees[1], donnees[2], donnees[-1])
            if cle_parc(x) in vus:
                ligne['statut'] = 'doublon'
            else:
                vus.add(cle_parc(x))
                parcs.append(x)
                ligne.update(statut='admissible', rang=len(parcs))
        except (ValueError, RuntimeError) as erreur:
            ligne['message'] = str(erreur)
        ligne.update(temps_construction_validation_s=perf_counter()-t,
                     cpu_construction_validation_s=process_time()-cpu)
        journal.append(ligne)
        if ligne['statut'] == 'admissible':
            # Les tentatives précédentes restent dans le coût de construction.
            couts.append((sum(l['temps_construction_validation_s'] for l in journal
                              if l['instance']==instance and l['n']==donnees[1] and l['famille']==famille),
                          sum(l['cpu_construction_validation_s'] for l in journal
                              if l['instance']==instance and l['n']==donnees[1] and l['famille']==famille)))
        if len(parcs) == nombre:
            return parcs, couts
    raise RuntimeError(f'{instance}/{donnees[1]}/{famille} : seulement {len(parcs)} départs distincts sur {nombre}.')


def raffiner(essai, parcs, valeurs, flux, budget):
    x0, aep0 = essai.meilleur_x.copy(), essai.meilleure_aep
    methode, d = essai.methode, essai.diametre
    traces = []
    graines = flux.get(methode, {})
    if methode == '1opt':
        raffiner_1opt(PontUnOpt(essai), graines['proposition'], limite_propositions=6400)
        return dict(arret='budget' if essai.budget_atteint else 'voisinages_epuises')
    if methode == 'recuit':
        resultat = recuit_simule(x0, aep0, essai, d, budget_total=budget,
            evaluations_initiales=4, bascule_evaluations=4,
            graine_proposition=graines['proposition'], graine_acceptation=graines['acceptation'],
            temperature_initiale=.10/log(2), refroidissement=(1/20)**(1/7),
            limite_propositions=6400, journaliser=traces.append)
    elif methode == 'evolution':
        resultat = evoluer(parcs[:4], valeurs, essai, d, mu=4, lambda_enfants=4,
            budget_total=budget, evaluations_initiales=4, bascule_evaluations=4,
            graine_parent=graines['parent'], graine_mutation=graines['proposition'],
            limite_propositions=6400, propositions_par_enfant=100, journaliser=traces.append)
    elif methode == 'cmaes':
        resultat = raffiner_cmaes(x0, aep0, essai.donnees, essai.evaluer,
            graine=42, sigma0=5., population=4, limite_propositions=6400)
        essai.generations_completes = resultat['generations_completes']
    else:
        reseau = candidats_glouton(essai.donnees, pas_facteur=2.5)
        candidats = np.asarray(list(dict.fromkeys(tuple(p) for p in
            list(reseau)+[p for parc in parcs[:4] for p in np.asarray(parc).reshape(-1,2)])))
        angles, poids, _ = poids_sillage(essai.donnees[0])
        aval = directions_avales(angles)
        parametres = dict(sigma0_m=d/2, kappa=.05, longueur_m=10*d)
        for k in range(64):
            if essai.budget_atteint:
                break
            x, trace = descendre_sillage(x0, candidats, d, aval, poids, parametres,
                                        1042+k, limite_propositions=240)
            essai.evaluer(x, phase='descente_sillage', descente=k+1, graine=1042+k,
                          score_final=trace['score_final'])
            traces.append(trace)
        resultat = dict(arret='budget' if essai.budget_atteint else 'limite_descentes',
                        candidats=len(candidats), parametres=parametres)
    resultat['traces'] = traces
    return resultat


def executer(args):
    if min(args.effectifs) < 1 or args.graine < 0:
        raise ValueError('Effectifs positifs et graine non négative requis.')
    for noms in (args.instances, args.effectifs, args.familles, args.methodes):
        if len(set(noms)) != len(noms):
            raise ValueError('Une liste de paramètres contient des doublons.')
    if args.temoin and 'coffee_house' not in args.familles:
        raise ValueError('Le témoin demande la famille coffee_house.')
    sortie = dossier_sortie(args.sortie, 'comparaison')
    evaluateur = Evaluateur(sortie)
    debut, cpu = perf_counter(), process_time()
    constructions, essais, comparaison, trajectoires = [], [], [], []
    plan = dict(instances=args.instances, effectifs=args.effectifs, familles=args.familles,
                methodes=args.methodes, budget=args.budget, graines={}, temoin=args.temoin,
                points_bruts=5000, angles=[30,75,120,165], graine_grille=42,
                pas_x_m=320, pas_y_m=161, initialisation=4,
                recuit=dict(temperature_initiale_gwh=.10/log(2), refroidissement=(1/20)**(1/7), bascule=4),
                evolution=dict(mu=4, lambda_enfants=4, bascule=4, propositions_par_enfant=100),
                sillage=dict(graine=1042, max_descentes=64, propositions=240, pas_facteur=2.5,
                             sigma0_d=.5, kappa=.05, longueur_d=10),
                cmaes=dict(graine=42, population=4, sigma0_m=5, limite_propositions=6400))
    for instance in args.instances:
        for n in args.effectifs:
            donnees = amon.charger_instance(instance, n)
            graines, flux = graines_contexte(args.graine, instance, n)
            plan['graines'][f'i{instance}_N{n}'] = dict(construction=graines, optimisation=flux)
            sauvegarder(sortie/'plan.json', plan)
            for famille in args.familles:
                nombre = args.budget if args.temoin and famille=='coffee_house' else 4
                parcs, couts = construire(donnees, famille, graines, nombre, constructions, instance)
                ecrire_csv(sortie/'constructions.csv', constructions)
                evaluateur.preparer(instance, donnees)
                methodes = (['exploration'] if args.temoin and famille=='coffee_house' else [])+args.methodes
                for methode in methodes:
                    cout = couts[3]
                    essai = Essai(evaluateur, donnees, instance, famille, methode, args.budget, *cout)
                    essais.append(essai)
                    info = {}
                    try:
                        valeurs = [essai.evaluer(x, phase='initialisation') for x in parcs[:4]]
                        aep_depart = essai.meilleure_aep
                        if methode=='exploration':
                            for rang, x in enumerate(parcs[4:], start=4):
                                essai.temps_construction_s, essai.cpu_construction_s = couts[rang]
                                essai.evaluer(x, phase='exploration')
                        elif not essai.budget_atteint:
                            info = raffiner(essai, parcs, valeurs, flux, args.budget)
                    except BudgetAEPAtteint:
                        info['arret'] = 'budget'
                    finally:
                        ecrire_csv(sortie/'appels.csv', evaluateur.appels)
                        ecrire_csv(sortie/'propositions.csv', [l for e in essais for l in e.propositions_lignes])
                    if essai.historique:
                        essai.historique[-1].update(essai.mesures())
                    for palier in sorted({4,8,args.budget}):
                        if palier > essai.imputes:
                            continue
                        ligne = essai.historique[palier-1]
                        comparaison.append(dict(instance=instance, n=n, famille=famille, methode=methode,
                            palier=palier, aep_gwh=ligne['meilleure_aep_gwh'], aep_depart_gwh=aep_depart,
                            gain_gwh=ligne['meilleure_aep_gwh']-aep_depart,
                            temps_total_impute_s=ligne['temps_total_impute_s'],
                            cpu_total_impute_s=ligne['cpu_total_impute_s'],
                            temps_simulation_impute_s=ligne['temps_simulation_impute_s'],
                            propositions=ligne['propositions'], rejets=ligne['rejets'], doublons=ligne['doublons'],
                            appels_physiques=ligne['appels_physiques'], fichier_coordonnees=ligne['fichier_coordonnees']))
                    trajectoires.extend(essai.historique)
                    ecrire_csv(sortie/'comparaison.csv', comparaison)
                    ecrire_csv(sortie/'trajectoires.csv', trajectoires)
                    sauvegarder(sortie/f'i{instance}_N{n}_{famille}_{methode}.json', info)
                    print(f'I{instance} N{n} {famille}/{methode}: {essai.imputes}/{args.budget}, {essai.meilleure_aep:.6f} GWh/an', flush=True)
    sauvegarder(sortie/'mesures.json', dict(temps_mural_s=perf_counter()-debut, cpu_s=process_time()-cpu,
        appels_physiques=len(evaluateur.appels), evaluations_imputees=sum(e.imputes for e in essais),
        temps_preparation_s=evaluateur.temps_preparation_s,
        temps_simulation_s=sum(a['temps_simulation_s'] for a in evaluateur.appels),
        perimetre='Exécution séquentielle hors imports et tracé. Temps des chaînes hors préparation commune.'))
    from .tracer import tracer_dossier
    tracer_dossier(sortie)
    print(f'{len(evaluateur.appels)} appels physiques ; sorties : {sortie}')
    return sortie
