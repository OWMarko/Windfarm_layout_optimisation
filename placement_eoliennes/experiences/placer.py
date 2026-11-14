"""Construire et exporter des parcs ; l'option --sans-aep évite le simulateur."""

import math
import hashlib
import json
from time import perf_counter

from .commun import dossier_sortie, ecrire_csv, ecrire_coordonnees, sauvegarder
from .evaluation import Evaluateur
from ..src import interface_amon as amon


def ajouter_arguments(parser):
    parser.add_argument('--instance', type=int, choices=range(1,7), default=1)
    parser.add_argument('--eoliennes', type=int, default=5)
    parser.add_argument('--methode', choices=('aleatoire','coffee_house','grille','quinconce','graphe','sobol'), default='grille')
    parser.add_argument('--graine', type=int, default=42)
    parser.add_argument('--nombre', type=int, default=1, help='Constructions par angle.')
    parser.add_argument('--angles', nargs='+', type=float, default=[30.])
    parser.add_argument('--pas-x', type=float)
    parser.add_argument('--pas-y', type=float)
    parser.add_argument('--decalage-x', type=float, default=0.)
    parser.add_argument('--decalage-y', type=float, default=0.)
    parser.add_argument('--candidats', type=int, default=5000)
    parser.add_argument('--sans-aep', action='store_true')
    parser.add_argument('--sortie')


def construire(args, donnees, graine, angle):
    communs = (*donnees[1:5], donnees[5], graine)
    if args.methode == 'grille':
        from ..src.placement_grille import generer_configuration
        return generer_configuration(*communs, pas_x=args.pas_x, pas_y=args.pas_y,
            angle_deg=angle, decalage_x=args.decalage_x, decalage_y=args.decalage_y)
    if args.methode == 'coffee_house':
        from ..src.placement_coffee_house import generer_configuration
        return generer_configuration(*communs, args.candidats, 1), {}
    if args.methode == 'aleatoire':
        from ..src.placement_aleatoire import generer_configuration
        return generer_configuration(*communs, 2000), {}
    if args.methode == 'graphe':
        from ..src.placement_graphe import generer_configuration
        return generer_configuration(*communs, args.candidats)
    if args.methode == 'sobol':
        from ..src.nuage_sobol import construire_configuration
        return construire_configuration(donnees, graine, graine+100), {}
    import numpy as np
    from ..src.placement_quinconce import generer_noeuds
    from ..src.placement_grille import regler_pas
    from ..src.placement_coffee_house import filtrer_candidats, selectionner_configuration
    pas = regler_pas(donnees[2], args.pas_x or 2*donnees[2], args.pas_x or 2*donnees[2])['pas_x_m']
    noeuds, _ = generer_noeuds(donnees[3], donnees[4], pas, angle)
    candidats = filtrer_candidats(noeuds, donnees[-1])
    return selectionner_configuration(candidats, donnees[1], donnees[2], np.random.default_rng(graine)), dict(pas_m=pas)


def executer(args):
    if min(args.eoliennes, args.nombre, args.candidats) < 1 or args.graine < 0:
        raise ValueError('Effectif, nombre et candidats positifs ; graine non négative.')
    parametres = args.angles + [args.decalage_x,args.decalage_y] + [v for v in (args.pas_x,args.pas_y) if v is not None]
    if not all(math.isfinite(v) for v in parametres):
        raise ValueError('Les paramètres géométriques doivent être finis.')
    if any(v is not None and v <= 0 for v in (args.pas_x,args.pas_y)):
        raise ValueError('Les pas doivent être strictement positifs.')
    if args.methode != 'grille' and (args.decalage_x or args.decalage_y):
        raise ValueError('Les translations concernent seulement la grille.')
    if args.methode not in ('grille','quinconce') and (args.pas_x is not None or args.pas_y is not None):
        raise ValueError('Les pas concernent la grille ou le quinconce.')
    if args.methode == 'quinconce' and args.pas_y is not None:
        raise ValueError('Le quinconce utilise un seul pas --pas-x.')
    sortie = dossier_sortie(args.sortie, 'placement')
    donnees = amon.charger_instance(args.instance,args.eoliennes)
    evaluateur = Evaluateur(sortie)
    angles = args.angles if args.methode in ('grille','quinconce') else [0.]
    lignes, grilles = [], []
    from .tracer import tracer_parc
    for angle in angles:
        for k in range(args.nombre):
            t = perf_counter()
            x, info = construire(args, donnees, args.graine+k, angle)
            amon.verifier_configuration(x, donnees[1], donnees[2], donnees[-1])
            temps = perf_counter()-t
            nom = f'placement_{len(lignes)+1}'
            ecrire_coordonnees(sortie/f'{nom}.csv', x)
            aep = None
            if not args.sans_aep:
                aep = evaluateur.evaluer(args.instance,donnees,x)[0]['aep_gwh']
            ligne = dict(instance=args.instance,n=args.eoliennes,methode=args.methode,
                         graine=args.graine+k,angle_deg=angle,aep_gwh=aep,
                         temps_construction_validation_s=temps,fichier_coordonnees=f'{nom}.csv')
            lignes.append(ligne)
            ecrire_csv(sortie/'placements.csv',lignes)
            if args.methode == 'grille' and aep is not None:
                from ..src.vent import poids_sillage
                from ..src.score_sillage import score_sillage
                from ..src.optimisation import cle_parc
                directions, poids, _ = poids_sillage(donnees[0])
                identifiant = hashlib.sha256(json.dumps(cle_parc(x)).encode()).hexdigest()
                grilles.append(dict(instance=args.instance,n=args.eoliennes,parc_id=identifiant,
                    angle_deg=angle % 180, tau=args.decalage_x/info['pas_x_m'],aep_gwh=aep,
                    score_R=score_sillage(x,directions,poids,donnees[2])))
                ecrire_csv(sortie/'grilles.csv',grilles)
            sauvegarder(sortie/f'{nom}_parametres.json',info)
            tracer_parc(donnees,x,sortie/f'{nom}.png',aep)
    ecrire_csv(sortie/'appels.csv',evaluateur.appels)
    print(f'{len(lignes)} parcs admissibles ; {len(evaluateur.appels)} appels PyWake ; {sortie}')
    return sortie
