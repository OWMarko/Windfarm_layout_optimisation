"""Validation hors angle de quatre modèles à partir d'un CSV fourni."""

from decimal import Decimal
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from ..src.ego import _matern_52, amelioration_esperee
from .commun import dossier_sortie, lire_csv, ecrire_csv

LONGUEUR, REGULARISATION = 0.25, 1e-8
MODELES = ("gp_actuel", "affine_R", "krigeage_R", "krigeage_intercept")
NOMS = ("GP", "Affine score", "Kriging with score drift", "Kriging with constant drift")

def kriger_gls(f, f_test, y, facteur, croisee):
    """Tendance GLS et correction du résidu, sans calcul d'inverse."""
    k_f = cho_solve(facteur, f)
    gram = f.T @ k_f
    beta = cho_solve(cho_factor(gram, lower=True), f.T @ cho_solve(facteur, y))
    residu = y-f @ beta
    return f_test @ beta + croisee.T @ cho_solve(facteur, residu)

def predire_hors_angle(parcs):
    angles = np.array([p["angle_deg"] for p in parcs])
    x = np.array([[p["angle_deg"]/180, (p["tau"]+0.25)/0.5] for p in parcs])
    y = np.array([float(p["aep"]) for p in parcs])
    r = np.array([p["score_R"] for p in parcs])
    predictions = np.full((len(parcs), len(MODELES)), np.nan)
    comptes = np.zeros_like(predictions, dtype=int)
    plis = []
    geometries = np.array([p["parc_id"] for p in parcs])
    for angle in sorted(set(angles)):
        test = angles == angle
        # Un même parc sous une autre paramétrisation n'est pas un apprentissage indépendant.
        train = (angles != angle) & ~np.isin(geometries, geometries[test])
        if train.sum() < 2:
            raise ValueError(f"Apprentissage insuffisant après exclusion de l'angle {angle} et de ses géométries.")
        # L'échelle de R est ajustée sur le seul apprentissage ; l'intercept
        # rend ce changement d'unité neutre pour les prédictions OLS et GLS.
        centre, echelle = r[train].mean(), r[train].std()
        if echelle == 0:
            raise ValueError("Score R constant dans un apprentissage.")
        f = np.column_stack((np.ones(train.sum()), (r[train]-centre)/echelle))
        f_test = np.column_stack((np.ones(test.sum()), (r[test]-centre)/echelle))
        predictions[test, 0] = amelioration_esperee(x[train], y[train], x[test],
            longueur=LONGUEUR, regularisation=REGULARISATION)[0]
        beta, _, rang, _ = np.linalg.lstsq(f, y[train], rcond=None)
        if rang != 2:
            raise ValueError("Tendance affine de rang incomplet.")
        predictions[test, 1] = f_test @ beta
        k = _matern_52(x[train], x[train], LONGUEUR) + REGULARISATION*np.eye(train.sum())
        facteur = cho_factor(k, lower=True)
        croisee = _matern_52(x[train], x[test], LONGUEUR)
        predictions[test, 2] = kriger_gls(f, f_test, y[train], facteur, croisee)
        predictions[test, 3] = kriger_gls(f[:, :1], f_test[:, :1], y[train], facteur, croisee)
        comptes[test, :] += 1
        plis.append((float(angle), int(train.sum()), int(test.sum())))
    if not np.all(comptes == 1) or not np.isfinite(predictions).all():
        raise ValueError("Il faut exactement une prédiction finie hors pli par parc et modèle.")
    return predictions, plis

def classer(parcs, predictions):
    y = np.array([float(p["aep"]) for p in parcs])
    bilan, ordres = [], []
    for m in range(5):
        valeurs = -predictions[:, m] if m < 4 else np.array([p["score_R"] for p in parcs])
        ordre = sorted(range(len(parcs)), key=lambda i: (valeurs[i], parcs[i]["angle_deg"], parcs[i]["tau"], parcs[i]["parc_id"]))
        ordres.append(ordre)
        uniques, vus = [], set()
        for i in ordre:
            if parcs[i]["parc_id"] not in vus:
                uniques.append(i)
                vus.add(parcs[i]["parc_id"])
        meilleur = max(parcs[i]["aep"] for i in uniques[:4])
        rmse = float(np.sqrt(np.mean((predictions[:, m]-y)**2))) if m < 4 else None
        bilan.append(dict(modele=NOMS[m] if m < 4 else "Score de sillage croissant", rmse=rmse,
            meilleure_aep_top4=str(meilleur), top4=", ".join(parcs[i]["parc_id"] for i in uniques[:4])))
    return bilan, ordres


def ajouter_arguments(parser):
    parser.add_argument('--csv',type=Path,required=True,
        help='Colonnes parc_id, angle_deg, tau, score_R, aep_gwh ; un seul site et N.')
    parser.add_argument('--sortie')


def executer(args):
    rows = lire_csv(args.csv)
    if not rows:
        raise ValueError('Tableau vide.')
    for champ in ('instance','n'):
        if champ in rows[0] and len({l[champ] for l in rows}) != 1:
            raise ValueError('La validation doit conserver un seul site et un seul effectif.')
    parcs, vus = [], {}
    for l in rows:
        if not all(k in l for k in ('parc_id','angle_deg','tau','score_R','aep_gwh')):
            raise ValueError('Colonnes requises : parc_id, angle_deg, tau, score_R, aep_gwh.')
        p = dict(parc_id=l['parc_id'],angle_deg=float(l['angle_deg']),tau=float(l['tau']),
                 score_R=float(l['score_R']),aep=Decimal(l['aep_gwh']))
        if not p['aep'].is_finite() or p['aep'] < 0 or not np.isfinite([p['angle_deg'],p['tau'],p['score_R']]).all():
            raise ValueError('Données non finies ou AEP négative.')
        if not p['parc_id']:
            raise ValueError('Chaque géométrie doit avoir un identifiant.')
        if p['parc_id'] in vus:
            if abs(p['aep']-vus[p['parc_id']]['aep']) > Decimal('1e-10'):
                raise ValueError('Deux AEP différentes pour une même géométrie.')
            continue
        vus[p['parc_id']] = p
        parcs.append(p)
    predictions, plis = predire_hors_angle(parcs)
    bilan, _ = classer(parcs,predictions)
    sortie = dossier_sortie(args.sortie,'krigeage')
    lignes = []
    for i,p in enumerate(parcs):
        lignes.append(dict(parc_id=p['parc_id'],angle_deg=p['angle_deg'],aep_gwh=str(p['aep']),
                          **{nom:float(predictions[i,j]) for j,nom in enumerate(MODELES)}))
    ecrire_csv(sortie/'predictions.csv',lignes)
    ecrire_csv(sortie/'mesures.csv',bilan)
    ecrire_csv(sortie/'plis.csv',[dict(angle_deg=a,apprentissage=b,test=c) for a,b,c in plis])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(2,2,figsize=(9,8))
    observe = np.array([float(p['aep']) for p in parcs])
    lo,hi = min(observe.min(),predictions.min()),max(observe.max(),predictions.max())
    for j,ax in enumerate(axes.flat):
        ax.scatter(observe,predictions[:,j],s=20)
        ax.plot([lo,hi],[lo,hi],color='0.5',linestyle='--')
        ax.set(xlabel='Observed AEP (GWh/year)',ylabel='Predicted AEP (GWh/year)',
               title=f'{NOMS[j]}: RMSE {bilan[j]["rmse"]:.3f}')
    fig.tight_layout(); fig.savefig(sortie/'observe_predit.png',dpi=160); plt.close(fig)
    for ligne in bilan[:4]:
        print(f"{ligne['modele']}: RMSE = {ligne['rmse']:.6f} GWh/year")
    print(sortie)
    return sortie
