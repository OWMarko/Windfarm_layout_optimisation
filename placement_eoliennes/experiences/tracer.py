"""Figures Matplotlib depuis les sorties CSV ; aucune simulation."""

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .commun import lire_csv


def tracer_parc(donnees, x, cible, aep=None, titre=None):
    fig, ax = plt.subplots(figsize=(7,5))
    zone = donnees[-1]
    for polygone in getattr(zone,'geoms',[zone]):
        xx, yy = polygone.exterior.xy
        ax.fill(xx,yy,color='0.94',edgecolor='0.3')
        for trou in polygone.interiors:
            xx,yy = trou.xy
            ax.fill(xx,yy,color='white',edgecolor='0.5',hatch='//')
    ax.scatter(x[::2],x[1::2],s=25,color='#0072B2')
    ax.set(xlabel='x (m)',ylabel='y (m)',title=titre or (
        f'N = {donnees[1]}' + (f' ; {aep:.3f} GWh/year' if aep is not None else '')))
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(cible,dpi=160)
    plt.close(fig)


def tracer_dossier(dossier):
    dossier = Path(dossier)
    sorties = dossier/'figures'
    sorties.mkdir(exist_ok=True)
    comparaison = lire_csv(dossier/'comparaison.csv')
    trajectoires = lire_csv(dossier/'trajectoires.csv')
    contextes = sorted({(int(l['instance']),int(l['n'])) for l in comparaison})
    for instance,n in contextes:
        rows = [l for l in trajectoires if int(l['instance'])==instance and int(l['n'])==n]
        fig,ax = plt.subplots(figsize=(9,5))
        for famille,methode in sorted({(l['famille'],l['methode']) for l in rows}):
            chaine = [l for l in rows if l['famille']==famille and l['methode']==methode]
            ax.step([int(l['budget_impute']) for l in chaine],[float(l['meilleure_aep_gwh']) for l in chaine],
                    where='post',label=f'{famille} / {methode}')
        ax.set(xlabel='Charged AEP evaluations',ylabel='Best feasible AEP (GWh/year)',title=f'Instance {instance}, N = {n}')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8,bbox_to_anchor=(1.02,1),loc='upper left')
        fig.tight_layout(); fig.savefig(sorties/f'convergence_i{instance}_N{n}.png',dpi=160); plt.close(fig)
    for instance in sorted({int(l['instance']) for l in comparaison}):
        for budget in sorted({int(l['palier']) for l in comparaison}):
            rows = [l for l in comparaison if int(l['instance'])==instance and int(l['palier'])==budget]
            fig,ax = plt.subplots(figsize=(9,5))
            for famille,methode in sorted({(l['famille'],l['methode']) for l in rows}):
                chaine = sorted([l for l in rows if l['famille']==famille and l['methode']==methode],key=lambda l:int(l['n']))
                ax.plot([int(l['n']) for l in chaine],[float(l['aep_gwh']) for l in chaine],marker='o',label=f'{famille} / {methode}')
            ax.set(xlabel='Number of turbines',ylabel='AEP (GWh/year)',title=f'Instance {instance}, budget {budget}')
            ax.grid(alpha=.2); ax.legend(fontsize=8,bbox_to_anchor=(1.02,1),loc='upper left')
            fig.tight_layout(); fig.savefig(sorties/f'aep_N_i{instance}_B{budget}.png',dpi=160); plt.close(fig)
    for instance,n in contextes:
        rows = [l for l in comparaison if int(l['instance'])==instance and int(l['n'])==n]
        budget = max(int(l['palier']) for l in rows)
        rows = [l for l in rows if int(l['palier'])==budget]
        fig,ax = plt.subplots(figsize=(9,5))
        for l in rows:
            ax.scatter(float(l['temps_total_impute_s']),float(l['aep_gwh']),label=f"{l['famille']} / {l['methode']}")
        ax.set(xlabel='Charged standalone time (s)',ylabel='AEP (GWh/year)',title=f'Instance {instance}, N = {n}, budget {budget}')
        ax.grid(alpha=.2); ax.legend(fontsize=8,bbox_to_anchor=(1.02,1),loc='upper left')
        fig.tight_layout(); fig.savefig(sorties/f'aep_temps_i{instance}_N{n}.png',dpi=160); plt.close(fig)
        meilleur = max(rows,key=lambda l:float(l['aep_gwh']))
        fichier = (dossier/meilleur['fichier_coordonnees']).resolve()
        if not fichier.is_relative_to(dossier.resolve()):
            raise ValueError('Les coordonnées doivent appartenir au dossier du run.')
        x = [float(l[champ]) for l in lire_csv(fichier) for champ in ('x_m','y_m')]
        from ..src import interface_amon as amon
        donnees = amon.charger_instance(instance,n)
        amon.verifier_configuration(x,n,donnees[2],donnees[-1])
        titre = (f'Instance {instance}, N = {n}, budget {budget}\n'
                 f"{meilleur['famille']} / {meilleur['methode']} : {float(meilleur['aep_gwh']):.3f} GWh/year")
        tracer_parc(donnees,x,sorties/f'carte_i{instance}_N{n}.png',titre=titre)
    return sorties


def ajouter_arguments(parser):
    parser.add_argument('--campagne',type=Path,required=True)


def executer(args):
    print(tracer_dossier(args.campagne))
