"""Évaluations partagées pendant un run et budgets propres à chaque méthode."""

import math
from time import perf_counter, process_time

from .commun import ecrire_coordonnees, journaliser
from ..src import interface_amon as amon
from ..src.glouton_aep import BudgetAEPAtteint
from ..src.optimisation import cle_parc


class Evaluateur:
    def __init__(self, dossier):
        self.dossier = dossier
        self.sites, self.cache = {}, {}
        self.appels = []
        self.temps_preparation_s = self.cpu_preparation_s = 0.0

    def preparer(self, instance, donnees):
        if instance not in self.sites:
            t, cpu = perf_counter(), process_time()
            accessoires = self.dossier / f'site_{instance}'
            accessoires.mkdir(exist_ok=True)
            self.sites[instance] = amon.preparer_evaluation(donnees[0], accessoires)
            self.temps_preparation_s += perf_counter()-t
            self.cpu_preparation_s += process_time()-cpu

    def evaluer(self, instance, donnees, x):
        amon.verifier_configuration(x, donnees[1], donnees[2], donnees[-1])
        cle = (instance, donnees[1], cle_parc(x))
        if cle in self.cache:
            return self.cache[cle], False
        self.preparer(instance, donnees)
        identifiant = len(self.appels)+1
        fichier = f'coordonnees/appel_{identifiant}.csv'
        ecrire_coordonnees(self.dossier / fichier, x)
        appel = dict(id_appel=identifiant, instance=instance, n=donnees[1],
                     fichier_coordonnees=fichier, statut='en_cours', aep_gwh=None)
        self.appels.append(appel)
        journaliser(self.dossier / 'appels.jsonl', appel)
        t, cpu = perf_counter(), process_time()
        try:
            aep = float(amon.evaluer_aep(x, self.sites[instance]))
            if not math.isfinite(aep) or aep < 0:
                raise RuntimeError('AEP non finie ou négative.')
            appel.update(aep_gwh=aep, statut='termine')
            self.cache[cle] = appel
        except BaseException as erreur:
            appel.update(statut='interrompu' if isinstance(erreur, KeyboardInterrupt) else 'echec',
                         message=str(erreur))
            raise
        finally:
            appel.update(temps_simulation_s=perf_counter()-t, cpu_simulation_s=process_time()-cpu)
            journaliser(self.dossier / 'appels.jsonl', appel)
        return appel, True


class Essai:
    """Chaque parc distinct consomme une imputation, même s'il est partagé."""

    def __init__(self, evaluateur, donnees, instance, famille, methode, budget,
                 temps_construction_s=0.0, cpu_construction_s=0.0):
        self.evaluateur, self.donnees = evaluateur, donnees
        self.instance, self.n, self.diametre = instance, donnees[1], donnees[2]
        self.famille, self.methode, self.budget = famille, methode, budget
        self.imputes = self.propositions = self.rejets = self.doublons = self.appels_physiques = 0
        self.generations_completes = 0
        self.meilleur_x = self.meilleure_aep = self.appel_meilleur = None
        self.valeurs, self.historique, self.propositions_lignes = {}, [], []
        self.temps_simulation_s = self.cpu_simulation_s = 0.0
        self.temps_simulation_impute_s = self.cpu_simulation_impute_s = 0.0
        self.temps_construction_s, self.cpu_construction_s = temps_construction_s, cpu_construction_s
        self.debut, self.debut_cpu = perf_counter(), process_time()
        self.preparation = evaluateur.temps_preparation_s
        self.preparation_cpu = evaluateur.cpu_preparation_s

    @property
    def budget_atteint(self):
        return self.imputes >= self.budget

    def mesures(self):
        # Retirer le calcul physique effectivement subi, ajouter son coût imputé.
        temps = (perf_counter()-self.debut - (self.evaluateur.temps_preparation_s-self.preparation)
                 - self.temps_simulation_s + self.temps_simulation_impute_s + self.temps_construction_s)
        cpu = (process_time()-self.debut_cpu - (self.evaluateur.cpu_preparation_s-self.preparation_cpu)
               - self.cpu_simulation_s + self.cpu_simulation_impute_s + self.cpu_construction_s)
        return dict(temps_total_impute_s=temps, cpu_total_impute_s=cpu,
                    temps_simulation_impute_s=self.temps_simulation_impute_s,
                    temps_construction_s=self.temps_construction_s)

    def evaluer(self, x, **meta):
        if self.budget_atteint:
            raise BudgetAEPAtteint('Budget imputé atteint.')
        self.propositions += 1
        x = [float(v) for v in x]
        ligne = dict(instance=self.instance, n=self.n, famille=self.famille, methode=self.methode,
                     proposition=self.propositions, statut='rejet_geometrique', appel_physique=0,
                     evaluation_imputee=0, aep_gwh=None, **meta)
        try:
            amon.verifier_configuration(x, self.n, self.diametre, self.donnees[-1])
        except ValueError as erreur:
            self.rejets += 1
            ligne['message'] = str(erreur)
            valeur = None
        else:
            cle = cle_parc(x)
            if cle in self.valeurs:
                self.doublons += 1
                valeur = self.valeurs[cle]['aep_gwh']
                ligne.update(statut='doublon', aep_gwh=valeur)
            else:
                appel, nouveau = self.evaluateur.evaluer(self.instance, self.donnees, x)
                valeur = appel['aep_gwh']
                self.valeurs[cle] = appel
                self.imputes += 1
                self.appels_physiques += int(nouveau)
                self.temps_simulation_impute_s += appel['temps_simulation_s']
                self.cpu_simulation_impute_s += appel['cpu_simulation_s']
                if nouveau:
                    self.temps_simulation_s += appel['temps_simulation_s']
                    self.cpu_simulation_s += appel['cpu_simulation_s']
                if self.meilleure_aep is None or valeur > self.meilleure_aep:
                    self.meilleur_x, self.meilleure_aep = x.copy(), valeur
                    self.appel_meilleur = appel
                ligne.update(statut='evalue', aep_gwh=valeur, id_appel=appel['id_appel'],
                             appel_physique=int(nouveau), evaluation_imputee=1)
        ligne.update(budget_impute=self.imputes, meilleure_aep_gwh=self.meilleure_aep,
                     rejets=self.rejets, doublons=self.doublons, propositions=self.propositions,
                     appels_physiques=self.appels_physiques, **self.mesures())
        if self.appel_meilleur is not None:
            ligne['fichier_coordonnees'] = self.appel_meilleur['fichier_coordonnees']
        self.propositions_lignes.append(ligne)
        journaliser(self.evaluateur.dossier / 'propositions.jsonl', ligne)
        if ligne['evaluation_imputee']:
            self.historique.append(ligne.copy())
        return valeur


class PontUnOpt:
    """Adapter le compteur au 1-opt sans changer ses voisinages."""
    def __init__(self, essai):
        self.essai, self.diametre = essai, essai.diametre
        self.debut = essai.propositions

    @property
    def meilleure_configuration(self):
        return self.essai.meilleur_x

    @property
    def budget_atteint(self):
        return self.essai.budget_atteint

    @property
    def propositions(self):
        return self.essai.propositions-self.debut

    def examiner(self, x, **meta):
        avant = self.essai.meilleure_aep
        self.essai.evaluer(x, phase='raffinement', **meta)
        return self.essai.meilleure_aep > avant
