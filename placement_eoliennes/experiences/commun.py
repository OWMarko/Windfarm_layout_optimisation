"""Fichiers CSV, coordonnées et sorties d'une exécution indépendante."""

import csv
from datetime import datetime
import json
from pathlib import Path

RACINE = Path(__file__).resolve().parents[2]


def dossier_sortie(chemin, prefixe):
    dossier = Path(chemin) if chemin else RACINE / 'resultats' / f'{prefixe}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    if dossier.exists() and any(dossier.iterdir()):
        raise ValueError(f'Le dossier de sortie doit être vide : {dossier}')
    dossier.mkdir(parents=True, exist_ok=True)
    return dossier.resolve()


def ecrire_csv(chemin, lignes):
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    champs = list(dict.fromkeys(k for ligne in lignes for k in ligne))
    with chemin.open('w', encoding='utf-8-sig', newline='') as flux:
        sortie = csv.DictWriter(flux, champs)
        sortie.writeheader()
        sortie.writerows(lignes)


def lire_csv(chemin):
    with Path(chemin).open(encoding='utf-8-sig', newline='') as flux:
        return list(csv.DictReader(flux))


def sauvegarder(chemin, objet):
    def encoder(x):
        if hasattr(x, 'tolist'):
            return x.tolist()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x).__name__)
    Path(chemin).write_text(json.dumps(objet, indent=2, ensure_ascii=False, default=encoder), encoding='utf-8')


def ecrire_coordonnees(chemin, x):
    ecrire_csv(chemin, [dict(turbine=i+1, x_m=a, y_m=b)
                       for i, (a, b) in enumerate(zip(x[::2], x[1::2]))])


def journaliser(chemin, ligne):
    with Path(chemin).open('a', encoding='utf-8') as flux:
        flux.write(json.dumps(ligne, ensure_ascii=False) + '\n')
