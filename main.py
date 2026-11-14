"""Commandes du projet : construire, comparer et analyser les placements."""

import argparse
import os
from pathlib import Path


def main(options=None):
    for nom in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[nom] = '1'
    os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'.cache/matplotlib'))
    from placement_eoliennes.experiences import placer, comparer, tracer, krigeage, decision
    parser = argparse.ArgumentParser(description='AMON wind-farm placement and AEP optimisation.')
    commandes = parser.add_subparsers(dest='commande')
    for nom,module,aide in (
        ('placer',placer,'Construire un parc ; --sans-aep pour la géométrie seule.'),
        ('comparer',comparer,'Comparer des constructions et raffinements depuis zéro.'),
        ('tracer',tracer,'Tracer les CSV d’une comparaison, sans simulation.'),
        ('krigeage',krigeage,'Valider des prédictions hors angle sur un CSV explicite.'),
        ('decision',decision,'Front AEP-effectif et choix sous une cible d’énergie.'),
    ):
        commande = commandes.add_parser(nom,help=aide,description=aide)
        module.ajouter_arguments(commande)
        commande.set_defaults(executer=module.executer)
    args = parser.parse_args(options)
    if args.commande is None:
        parser.print_help()
        return
    try:
        return args.executer(args)
    except (ValueError,LookupError,RuntimeError) as erreur:
        parser.exit(1,f'Erreur : {erreur}\n')


if __name__ == '__main__':
    main()
