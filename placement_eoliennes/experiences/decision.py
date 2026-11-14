"""Comparer les effectifs observés et appliquer une cible d'énergie."""

from collections import defaultdict
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from .commun import dossier_sortie, ecrire_csv, lire_csv


def fraction(x):
    return Fraction(str(x))

def decimal(x, chiffres=12):
    # Les fractions exportées portent la valeur exacte ; ces colonnes sont flottantes.
    return repr(float(x))

def resoudre(lignes):
    """Intersection exacte des demi-droites Ei-lambda Ni >= Ej-lambda Nj."""
    meilleurs = {}
    for l in lignes:
        n = l["eoliennes"]
        if n not in meilleurs or fraction(l["aep_gwh"]) > fraction(meilleurs[n]["aep_gwh"]):
            meilleurs[n] = l
    front, seuils = [], []
    for n, l in sorted(meilleurs.items()):
        e = fraction(l["aep_gwh"])
        domine = any(m < n and fraction(q["aep_gwh"]) >= e for m, q in meilleurs.items())
        bas, haut = Fraction(0), None
        for q in lignes:  # Toutes les observations vérifiées, y compris dominées.
            m, f = q["eoliennes"], fraction(q["aep_gwh"])
            if m == n:
                continue
            seuil = (e-f) / (n-m)
            if n > m:
                haut = seuil if haut is None else min(haut, seuil)
            else:
                bas = max(bas, seuil)
        supporte = haut is None or bas <= haut
        front.append(dict(eoliennes=n, aep_gwh=l["aep_gwh"], parc_id=l["parc_id"], methode=l["methode"],
            non_domine=not domine, optimal_pour_un_lambda_positif=supporte and (haut is None or haut > 0),
            coordonnees=l["coordonnees"], source=l["source"]))
        if supporte:
            seuils.append(dict(eoliennes=n, parc_id=l["parc_id"], aep_gwh=l["aep_gwh"],
                lambda_min_exact=str(bas), lambda_max_exact=str(haut) if haut is not None else "+inf",
                lambda_min=decimal(bas), lambda_max=decimal(haut) if haut is not None else "+inf",
                ca_sur_p_min=decimal(1000*bas, 9), ca_sur_p_max=decimal(1000*haut, 9) if haut is not None else "+inf"))
            # Vérification des deux bornes et de l'intérieur contre tous les candidats.
            points = [bas, bas+1] if haut is None else [bas, (bas+haut)/2, haut]
            for z in points:
                if any(e-z*n < fraction(q["aep_gwh"])-z*q["eoliennes"] for q in lignes):
                    raise ValueError("Intervalle de décision incorrect.")
    egalites = []
    for z in sorted({fraction(l["lambda_min_exact"]) for l in seuils if fraction(l["lambda_min_exact"]) > 0}):
        scores = [(fraction(l["aep_gwh"])-z*l["eoliennes"], l) for l in lignes]
        maximum = max(s for s, _ in scores)
        ids = sorted({l["parc_id"] for s, l in scores if s == maximum})
        ns = sorted({l["eoliennes"] for s, l in scores if s == maximum})
        if len(ns) < 2:
            raise ValueError("Seuil sans égalité entre effectifs.")
        egalites.append({"lambda_exact": str(z), "lambda": decimal(z),
                         "effectifs": ";".join(map(str, ns)), "parcs": ";".join(ids)})
    return front, seuils, egalites


def ajouter_arguments(parser):
    parser.add_argument('--csv',type=Path,required=True,help='comparaison.csv produit par comparer.')
    parser.add_argument('--cible',type=float,default=60.,help='Cible illustrative en GWh/an.')
    parser.add_argument('--sortie')


def executer(args):
    import math
    if not math.isfinite(args.cible) or args.cible < 0:
        raise ValueError('Cible finie et positive ou nulle requise.')
    rows = lire_csv(args.csv)
    if not rows:
        raise ValueError('Tableau vide.')
    groupes, chaines = defaultdict(list), defaultdict(list)
    for l in rows:
        for champ in ('instance','n','palier','aep_gwh','famille','methode'):
            if champ not in l:
                raise ValueError(f'Colonne absente : {champ}')
        aep = Decimal(l['aep_gwh'])
        if not aep.is_finite() or aep < 0 or int(l['n']) < 1:
            raise ValueError('AEP ou effectif invalide.')
        groupes[(l['instance'],l['palier'])].append(l)
        chaines[(l['instance'],l['palier'],l['famille'],l['methode'])].append(l)
    sortie = dossier_sortie(args.sortie,'decision')
    front, decisions, gains, seuils, egalites = [], [], [], [], []
    for (instance,palier), parcs in groupes.items():
        candidats = [dict(eoliennes=int(l['n']),aep_gwh=l['aep_gwh'],
                          parc_id=l.get('fichier_coordonnees',f"{l['famille']}/{l['methode']}/N{l['n']}"),
                          methode=f"{l['famille']}/{l['methode']}",
                          coordonnees=l.get('fichier_coordonnees',''),source=args.csv.name) for l in parcs]
        _, intervalles, transitions = resoudre(candidats)
        seuils.extend(dict(instance=instance,palier=palier,**l) for l in intervalles)
        egalites.extend(dict(instance=instance,palier=palier,**l) for l in transitions)
        for l in parcs:
            n,e = int(l['n']),Decimal(l['aep_gwh'])
            domine = any(int(q['n']) <= n and Decimal(q['aep_gwh']) >= e and
                         (int(q['n']) < n or Decimal(q['aep_gwh']) > e) for q in parcs)
            if not domine:
                front.append(l)
        suffisants = [l for l in parcs if Decimal(l['aep_gwh']) >= Decimal(str(args.cible))]
        if suffisants:
            nmin = min(int(l['n']) for l in suffisants)
            for l in suffisants:
                if int(l['n']) == nmin:
                    decisions.append(dict(l,cible_gwh=args.cible,statut='plus_petit_effectif_teste'))
        else:
            decisions.append(dict(instance=instance,palier=palier,cible_gwh=args.cible,statut='cible_non_atteinte'))
    for (instance,palier,famille,methode), parcs in chaines.items():
        parcs = sorted(parcs,key=lambda l:int(l['n']))
        if len({l['n'] for l in parcs}) != len(parcs):
            raise ValueError('Plusieurs lignes pour une même méthode, instance, N et palier.')
        for a,b in zip(parcs,parcs[1:]):
            gain=(Decimal(b['aep_gwh'])-Decimal(a['aep_gwh']))/Decimal(int(b['n'])-int(a['n']))
            gains.append(dict(instance=instance,palier=palier,famille=famille,methode=methode,
                              n_depart=a['n'],n_final=b['n'],gain_gwh_par_turbine=str(gain)))
    ecrire_csv(sortie/'front.csv',front)
    ecrire_csv(sortie/'decision.csv',decisions)
    ecrire_csv(sortie/'gains_N.csv',gains)
    ecrire_csv(sortie/'seuils.csv',seuils)
    ecrire_csv(sortie/'egalites.csv',egalites)
    print(f'{len(rows)} résultats ; front N-AEP et cible {args.cible:g} GWh/an : {sortie}')
    return sortie
