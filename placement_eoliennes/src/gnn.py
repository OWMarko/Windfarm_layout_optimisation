"""Petit réseau de graphes dirigés, entraîné sur l'AEP avec NumPy et SciPy.

Les arêtes utilisent le même noyau géométrique que le score de sillage.
Seuls les poids des deux couches et de la lecture finale sont appris.
"""

import numpy as np
from scipy.optimize import minimize


def graphe_sillage(configuration, angles_deg, poids, diametre):
    """Renvoie les attributs des nœuds et deux matrices de messages normalisées.

    W[i,j] mesure l'interaction de i vers j. A_in agrège les prédécesseurs,
    A_out les successeurs. Les lignes nulles restent nulles.
    Attributs : degrés entrant/sortant pondérés, distance moyenne/minimale en D.
    """
    x = np.asarray(configuration, dtype=float)
    angles = np.asarray(angles_deg, dtype=float)
    masses = np.asarray(poids, dtype=float)
    if x.ndim != 1 or x.size < 4 or x.size % 2 or not np.isfinite(x).all():
        raise ValueError("Il faut les 2N coordonnées finies d'au moins deux turbines.")
    if (angles.ndim != 1 or not angles.size or masses.shape != angles.shape
            or not np.isfinite(angles).all() or not np.isfinite(masses).all()
            or np.any(masses < 0) or not np.isclose(masses.sum(), 1.0)
            or not np.isfinite(diametre) or diametre <= 0):
        raise ValueError("Vent normalisé et diamètre strictement positif requis.")
    points = x.reshape(-1, 2)
    r = points[None, :, :] - points[:, None, :]
    distances = np.linalg.norm(r, axis=2) / diametre
    if np.any(distances[np.triu_indices(len(points), 1)] == 0):
        raise ValueError("Le graphe ne doit pas contenir de turbines confondues.")
    radians = np.deg2rad(angles)
    aval = np.column_stack((-np.sin(radians), -np.cos(radians)))
    transverse = np.column_stack((-aval[:, 1], aval[:, 0]))
    s = r @ aval.T
    t = r @ transverse.T
    # La largeur n'est calculée qu'avec une distance aval positive.
    distance_aval = np.maximum(s, 0)
    noyau = np.exp(-0.5 * (t / (0.5 * diametre + 0.05 * distance_aval)) ** 2
                   - distance_aval / (10 * diametre))
    noyau[s <= 1e-9] = 0
    interactions = noyau @ masses
    entrant, sortant = interactions.sum(axis=0), interactions.sum(axis=1)
    minimum = distances.copy()
    np.fill_diagonal(minimum, np.inf)
    h = np.column_stack((entrant, sortant, distances.sum(axis=1) / (len(points) - 1),
                         minimum.min(axis=1)))

    def normaliser(a):
        somme = a.sum(axis=1, keepdims=True)
        return np.divide(a, somme, out=np.zeros_like(a), where=somme > 0)

    return h, normaliser(interactions.T), normaliser(interactions)


class PetitGNN:
    """Deux couches tanh avec messages entrants/sortants et lecture moyenne.

    Le critère est 0.5*MSE sur l'AEP standardisée + 0.5*l2*||theta||².
    La pénalité porte sur les 109 paramètres, biais compris. Pas d'arrêt choisi
    avec les étiquettes du test. Toutes les normalisations viennent du fit.
    """

    def __init__(self, largeur=4, l2=1e-3, maxiter=200, graine=42):
        if (not isinstance(largeur, int) or largeur < 1 or not isinstance(maxiter, int)
                or maxiter < 1 or not np.isfinite(l2) or l2 < 0):
            raise ValueError("Largeur, maxiter et régularisation invalides.")
        self.largeur, self.l2, self.maxiter, self.graine = largeur, l2, maxiter, graine
        self.formes = [(4, largeur)] * 3 + [(largeur,)]
        self.formes += [(largeur, largeur)] * 3 + [(largeur,), (largeur,), (1,)]
        self.tailles = [int(np.prod(forme)) for forme in self.formes]
        self.nombre_parametres = sum(self.tailles)

    def _deplier(self, theta):
        return [bloc.reshape(forme) for bloc, forme in
                zip(np.split(theta, np.cumsum(self.tailles)[:-1]), self.formes)]

    @staticmethod
    def _verifier_entrees(h, a_in, a_out):
        h, a_in, a_out = (np.asarray(a, dtype=float) for a in (h, a_in, a_out))
        if (h.ndim != 3 or h.shape[-1] != 4 or min(h.shape[:2]) < 1
                or a_in.shape != (h.shape[0], h.shape[1], h.shape[1])
                or a_out.shape != a_in.shape
                or not all(np.isfinite(a).all() for a in (h, a_in, a_out))):
            raise ValueError("Attendu H de forme (B,N,4), A_in et A_out de forme (B,N,N).")
        return h, a_in, a_out

    def _avant(self, theta, h, a_in, a_out):
        poids = self._deplier(theta)
        traces = []
        for couche in range(2):
            ws, wi, wo, biais = poids[4 * couche:4 * couche + 4]
            mi, mo = a_in @ h, a_out @ h
            suivant = np.tanh(h @ ws + mi @ wi + mo @ wo + biais)
            traces.append((h, mi, mo, suivant))
            h = suivant
        pool = h.mean(axis=1)
        return pool @ poids[-2] + poids[-1][0], (poids, traces, pool)

    def _perte_gradient(self, theta, h, a_in, a_out, y):
        prediction, (poids, traces, pool) = self._avant(theta, h, a_in, a_out)
        residu = prediction - y
        perte = 0.5 * np.mean(residu ** 2) + 0.5 * self.l2 * np.dot(theta, theta)
        derivee = residu / len(y)
        gradients = [None] * len(poids)
        gradients[-2] = pool.T @ derivee
        gradients[-1] = np.array([derivee.sum()])
        dh = np.broadcast_to(derivee[:, None, None] * poids[-2][None, None, :] / h.shape[1],
                             traces[-1][-1].shape).copy()
        for couche in (1, 0):
            entree, mi, mo, sortie = traces[couche]
            ws, wi, wo, _ = poids[4 * couche:4 * couche + 4]
            dz = dh * (1 - sortie ** 2)
            for offset, entree_locale in enumerate((entree, mi, mo)):
                gradients[4 * couche + offset] = np.einsum("bni,bnj->ij", entree_locale, dz)
            gradients[4 * couche + 3] = dz.sum(axis=(0, 1))
            dh = (dz @ ws.T + a_in.transpose(0, 2, 1) @ (dz @ wi.T)
                  + a_out.transpose(0, 2, 1) @ (dz @ wo.T))
        gradient = np.concatenate([g.ravel() for g in gradients]) + self.l2 * theta
        return float(perte), gradient

    def fit(self, h, a_in, a_out, y):
        h, a_in, a_out = self._verifier_entrees(h, a_in, a_out)
        y = np.asarray(y, dtype=float)
        if y.shape != (len(h),) or not np.isfinite(y).all():
            raise ValueError("Il faut une AEP finie par graphe d'apprentissage.")
        self.moyenne_x_ = h.mean(axis=(0, 1))
        self.echelle_x_ = h.std(axis=(0, 1))
        self.echelle_x_[self.echelle_x_ < 1e-12] = 1.0
        self.moyenne_y_, self.echelle_y_ = float(y.mean()), float(y.std())
        if self.echelle_y_ < 1e-12:
            self.echelle_y_ = 1.0
        x = (h - self.moyenne_x_) / self.echelle_x_
        cible = (y - self.moyenne_y_) / self.echelle_y_
        rng = np.random.default_rng(self.graine)
        # Échelle commune fixe pour limiter la saturation initiale des trois chemins.
        theta = rng.normal(0, 0.2, self.nombre_parametres)
        objectif = lambda p: self._perte_gradient(p, x, a_in, a_out, cible)
        initiale = objectif(theta)[0]
        resultat = minimize(objectif, theta, jac=True, method="L-BFGS-B",
                            options={"maxiter": self.maxiter, "ftol": 1e-10, "gtol": 1e-6})
        self.theta_ = resultat.x.copy()
        if not np.isfinite(self.theta_).all() or not np.isfinite(resultat.fun):
            raise RuntimeError("L'entraînement du GNN a produit une valeur non finie.")
        self.bilan_ = dict(succes=bool(resultat.success), statut=int(resultat.status),
                           message=str(resultat.message), iterations=int(resultat.nit),
                           evaluations_objectif=int(resultat.nfev), perte_initiale=initiale,
                           perte_finale=float(resultat.fun),
                           norme_gradient_inf=float(np.max(np.abs(resultat.jac))),
                           nombre_parametres=self.nombre_parametres)
        return self

    def predict(self, h, a_in, a_out):
        h, a_in, a_out = self._verifier_entrees(h, a_in, a_out)
        if not hasattr(self, "theta_"):
            raise RuntimeError("Entraîner le modèle avant la prédiction.")
        standard = (h - self.moyenne_x_) / self.echelle_x_
        prediction, _ = self._avant(self.theta_, standard, a_in, a_out)
        return self.moyenne_y_ + self.echelle_y_ * prediction
