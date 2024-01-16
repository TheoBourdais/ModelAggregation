import numpy as np
import scipy.linalg


class Agregator:
    DEFINING_ATTRIBUTES = ["K", "models", "kernel_name", "intercept_choice"]

    def __init__(self, K, models, kernel_name=None, intercept_choice="fit"):
        self.K = K
        self.models = models
        self.kernel_name = kernel_name
        assert isinstance(intercept_choice, np.ndarray) or intercept_choice in [
            "fit",
            None,
        ]
        self.intercept_choice = intercept_choice
        if isinstance(intercept_choice, np.ndarray):
            assert intercept_choice.shape[0] == len(models)
            self.intercept = intercept_choice
            self.fit_intercept = False
        elif intercept_choice is None:
            self.intercept = np.zeros(len(models))
            self.fit_intercept = False
        else:
            self.fit_intercept = True

    @property
    def attributes(self):
        return {k: getattr(self, k) for k in self.DEFINING_ATTRIBUTES}

    @property
    def model_number(self):
        return len(self.models)

    @property
    def coef_(self):
        return self.V

    @property
    def intercept(self):
        try:
            return self.intercept_
        except AttributeError:
            raise AttributeError("You must fit the model before calling its intercept")

    @intercept.setter
    def intercept(self, value):
        if value is not None:
            self.intercept_ = value

    @property
    def kernel_matrix(self):
        return self._kernel_matrix

    def fit(self, X, y, alpha=1.0):
        new = self.__class__(**self.attributes)
        new.X = X
        n = X.shape[0]
        new.y = y
        new._kernel_matrix = new.K(X, X)
        new.Mmat = self.call_models(X)
        to_invert = np.einsum("ijkl,ki->ijl", new.kernel_matrix, new.Mmat)
        to_invert = np.einsum("ijl,lj->ij", to_invert, new.Mmat)
        new.cho_factor = scipy.linalg.cho_factor(
            to_invert + y.shape[0] * alpha * np.eye(to_invert.shape[0])
        )
        new.V, new.intercept = Agregator.solve_lstsq(
            cho_factor=new.cho_factor,
            Y=y,
            M=new.Mmat,
            intercept=None if new.fit_intercept else new.intercept,
            alpha=alpha,
        )
        return new

    def call_models(self, X):
        return np.stack(list(map(lambda x: x.predict(X), self.models)), axis=0)

    def regularized_lstsq(A, y, reg=1e-10):
        AtA = A.T @ A
        AtA.flat[:: AtA.shape[0] + 1] += reg
        return scipy.linalg.solve(AtA, A.T @ y, assume_a="pos")

    def solve_lstsq(cho_factor, Y, M, intercept, alpha=1e-10):
        fit_intercept = intercept is None
        if fit_intercept:
            intercept = Agregator.regularized_lstsq(M.T, Y, reg=Y.shape[0] * alpha)
        intercept_aggregate = M.T @ intercept
        V = scipy.linalg.cho_solve(cho_factor, Y - intercept_aggregate)
        V = V[None, :] * M
        return V.T, intercept if fit_intercept else None

    @property
    def model_covariance(self):
        try:
            return self.model_covariance_
        except AttributeError:
            self.model_covariance_ = np.cov(self.Mmat - self.y[None, :])
            return self.model_covariance_

    @model_covariance.setter
    def model_covariance(self, value):
        self.model_covariance_ = value

    def covariance(self, x, M, K_eval, alpha, return_alpha=False):
        vec = np.einsum("ak,kija->kji", self.Mmat, K_eval)
        inv_vec = scipy.linalg.cho_solve(self.cho_factor, vec.reshape(vec.shape[0], -1))
        inv_vec = inv_vec.reshape(vec.shape)
        cov_alpha = np.einsum("iijk->ijk", self.K(x, x)) - np.einsum(
            "jki,jli->ikl", inv_vec, vec
        )

        cov = np.einsum("ijk,kj->i", cov_alpha, self.model_covariance)
        cov += np.einsum("ijk,ji,ki->i", cov_alpha, M, M)
        cov += np.einsum("jk,ji,ki->i", self.model_covariance, alpha, alpha)
        if return_alpha:
            return cov, cov_alpha
        return cov

    def predict(self, X, return_alpha=False, return_cov=False):
        return self(X, return_alpha=return_alpha, return_cov=return_cov)

    def __call__(self, x, return_alpha=False, return_cov=False):
        try:
            K_eval = self.K(self.X, x)
            M = self.call_models(x)
            alpha = np.einsum("ij,ikjm->mk", self.V, K_eval) + self.intercept[:, None]
            pred_aggregate = np.einsum("ij,ij->j", alpha, M)
            if not (return_alpha or return_cov):
                return pred_aggregate
            res = {"pred_aggregate": pred_aggregate}
            if return_alpha:
                res["alpha"] = alpha
            if return_cov:
                cov = self.covariance(x, M, K_eval, alpha, return_alpha=return_alpha)
                if return_alpha:
                    res["covariance"] = cov[0]
                    res["covariance_alpha"] = cov[1]
                else:
                    res["covariance"] = cov
            return res
        except AttributeError:
            raise AttributeError("You must fit the model before calling it")

    def __repr__(self) -> str:
        return f"[{self.kernel_name} Agregator](models={[m.__repr__() for m in self.models]})"

    def __str__(self) -> str:
        return self.__repr__()
