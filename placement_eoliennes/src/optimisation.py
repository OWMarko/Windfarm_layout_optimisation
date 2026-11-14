"""Criblage de grilles et recherche locale sous budget d'appels à PyWake."""

from time import perf_counter

import numpy as np

from .interface_amon import evaluer_aep, verifier_configuration
from .placement_grille import generer_configuration


def cle_parc(configuration):
    """Identifie un parc indépendamment de la numérotation des turbines."""
    return tuple(sorted(zip(configuration[0::2], configuration[1::2])))


class EssaiAEP:
    """Compte les propositions et ne simule que les placements valides et inédits."""

    def __init__(self, donnees, site, budget):
        _, self.nombre, self.diametre, _, _, self.zone = donnees
        self.site = site
        self.budget = budget
        self.propositions = 0
        self.rejets = 0
        self.doublons = 0
        self.echecs_construction = 0
        self.appels_pywake = 0
        self.temps_pywake_s = 0.0
        self.historique = []
        self.meilleure_configuration = None
        self.meilleure_aep = None
        self.deja_evalues = set()

    @property
    def budget_atteint(self):
        return self.appels_pywake >= self.budget

    def initialiser_depart(self, configuration, aep_depart=None):
        """Charge un parc fourni et, si disponible, son AEP pour ce même site."""
        if self.meilleure_configuration is not None or self.propositions:
            raise ValueError("Le départ doit être fixé avant toute proposition.")
        verifier_configuration(configuration, self.nombre, self.diametre, self.zone)
        if aep_depart is None:
            self.examiner(configuration)
        else:
            if not np.isfinite(aep_depart) or aep_depart < 0:
                raise ValueError("L'AEP du départ doit être finie et positive ou nulle.")
            self.meilleure_configuration = configuration.copy()
            self.meilleure_aep = aep_depart
            self.deja_evalues.add(cle_parc(configuration))

    def examiner(self, configuration, *, angle=None, pas=None):
        """Évalue au plus une fois une implantation et indique si elle améliore l'AEP."""
        if self.budget_atteint:
            return False
        self.propositions += 1
        try:
            verifier_configuration(configuration, self.nombre, self.diametre, self.zone)
        except ValueError:
            self.rejets += 1
            return False
        cle = cle_parc(configuration)
        if cle in self.deja_evalues:
            self.doublons += 1
            return False
        self.deja_evalues.add(cle)
        self.appels_pywake += 1
        debut = perf_counter()
        try:
            aep = evaluer_aep(configuration, self.site)
        finally:
            self.temps_pywake_s += perf_counter() - debut
        ameliore = self.meilleure_aep is None or aep > self.meilleure_aep
        if ameliore:
            self.meilleure_aep = aep
            self.meilleure_configuration = configuration
        self.historique.append({"appel": self.appels_pywake, "aep_gwh": aep,
                                "meilleure_aep_gwh": self.meilleure_aep,
                                "angle_deg": angle, "pas_m": pas})
        return ameliore


def examiner_grille(essai, donnees, graine, angle):
    _, nombre, diametre, borne_inf, borne_sup, zone = donnees
    try:
        configuration, _ = generer_configuration(nombre, diametre, borne_inf, borne_sup, zone, graine,
                                                 angle_deg=angle)
    except RuntimeError:
        essai.echecs_construction += 1
        return
    essai.examiner(configuration, angle=angle)


def cribler_grilles(essai, donnees, graine, angle_initial):
    """Échantillonne les angles de la grille à pas réguliers sur 180 degrés."""
    for indice in range(essai.budget):
        angle = (angle_initial + indice * 180 / essai.budget) % 180
        examiner_grille(essai, donnees, graine, angle)


def voisins(configuration, pas, generateur):
    """Déplace une seule turbine selon un des quatre axes signés."""
    indices = generateur.permutation(len(configuration) // 2)
    directions = generateur.permutation(((pas, 0), (-pas, 0), (0, pas), (0, -pas)))
    for dx, dy in directions:
        for indice in indices:
            candidat = configuration.copy()
            coordonnee = 2 * indice if dx else 2 * indice + 1
            deplacement = float(dx if dx else dy)
            candidat[coordonnee] = round(candidat[coordonnee] + deplacement, 2)
            yield candidat


def voisins_discrets(configuration, candidats, generateur):
    """Énumère sans remise des échanges entre une turbine et un nœud libre."""
    occupees = set(cle_parc(configuration))
    libres = [tuple(point) for point in candidats if tuple(point) not in occupees]
    for indice in generateur.permutation(len(occupees) * len(libres)):
        turbine, position = divmod(indice, len(libres))
        candidat = configuration.copy()
        candidat[2 * turbine:2 * turbine + 2] = [float(v) for v in libres[position]]
        yield candidat


def descendre(essai, produire_voisins, *, pas=None, limite_propositions=None):
    """Repart du nouveau parc après chaque première amélioration."""
    while essai.meilleure_configuration is not None and not essai.budget_atteint:
        depart = essai.meilleure_configuration
        for candidat in produire_voisins(depart):
            if limite_propositions is not None and essai.propositions >= limite_propositions:
                return
            if essai.examiner(candidat, pas=pas) or essai.budget_atteint:
                break
        if essai.meilleure_configuration is depart:
            return


def raffiner_1opt(essai, graine, *, limite_propositions=None):
    """Raffine le départ courant par déplacements axiaux aux pas D et D/2."""
    generateur = np.random.default_rng(graine)
    for pas in (essai.diametre, essai.diametre / 2):
        descendre(essai, lambda depart: voisins(depart, pas, generateur),
                  pas=pas, limite_propositions=limite_propositions)


def raffiner_echanges(essai, candidats, graine, *, limite_propositions):
    """Raffine le départ courant par échanges sur un nuage de positions."""
    generateur = np.random.default_rng(graine)
    descendre(essai, lambda depart: voisins_discrets(depart, candidats, generateur),
              limite_propositions=limite_propositions)


def raffiner_placement(essai, donnees, graine, angle_initial):
    """Conserve la grille initiale du parcours historique."""
    examiner_grille(essai, donnees, graine, angle_initial % 180)
    raffiner_1opt(essai, graine)
