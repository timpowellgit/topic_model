from builtins import super

from sklearn.decomposition import factor_analysis
import numpy as np
import time

class FARotate(factor_analysis.FactorAnalysis):

    def __init__(self, n_components=None, tol=1e-2, copy=True, max_iter=1000,
                 noise_variance_init=None, svd_method='randomized',
                 iterated_power=3, random_state=0, rotation=None,):
        super().__init__(n_components=n_components, tol=tol, copy=copy, max_iter=max_iter,
                 noise_variance_init=noise_variance_init, svd_method=svd_method,
                 iterated_power=iterated_power, random_state=random_state)
        self.rotation= rotation

    def fit(self, X, y=None):
        start = time.time()
        super().fit(X)
        print 'fit took this long: ', time.time()-start
        print 'fit done going to rotate'
        if self.rotation is not None:
            self.components_ = self._rotate(self.components_,method=self.rotation)



    def _rotate(self, components, method="varimax", n_components=None, tol=1e-6):

        "Rotate the factor analysis solution."
        implemented = ("varimax", "quartimax")
        if method in implemented:
            return _ortho_rotation(components.T, method=method,
                                   tol=tol)[:self.n_components]
        else:
            raise ValueError("'method' must be in %s, not %s"
                             % (implemented, method))


def _ortho_rotation(components, method='varimax', tol=1e-6, max_iter=100):
    """Return rotated components."""
    start =time.time()
    nrow, ncol = components.shape
    rotation_matrix = np.eye(ncol)
    var = 0

    for _ in range(max_iter):
        comp_rot = np.dot(components, rotation_matrix)
        if method == "varimax":
            tmp = np.diag((comp_rot ** 2).sum(axis=0)) / nrow
            tmp = np.dot(comp_rot, tmp)
        elif method == "quartimax":
            tmp = 0
        u, s, v = np.linalg.svd(
            np.dot(components.T, comp_rot ** 3 - tmp))
        rotation_matrix = np.dot(u, v)
        var_new = np.sum(s)
        if var != 0 and var_new < var * (1 + tol):
            break
        var = var_new
    print 'rotate took this long: ', time.time()-start
    return np.dot(components, rotation_matrix).T