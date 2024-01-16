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
        self.y = y
        new._kernel_matrix = new.K(X, X)
        new.Mmat = self.call_models(X)
        to_invert = np.einsum("ijkl,ki->ijl", new.kernel_matrix, new.Mmat)
        to_invert = to_invert.reshape(n, n * new.model_number)
        V, intercept = Agregator.solve_lstsq(
            A=to_invert,
            Y=y,
            M=new.Mmat,
            intercept=None if new.fit_intercept else new.intercept,
            alpha=alpha,
        )
        new.V = V.reshape(n, new.model_number)
        new.intercept = intercept
        return new

    def call_models(self, X):
        return np.stack(list(map(lambda x: x.predict(X), self.models)), axis=0)

    def regularized_lstsq(A, y, reg=1e-10):
        AtA = A.T @ A
        AtA.flat[:: AtA.shape[0] + 1] += reg
        return scipy.linalg.solve(AtA, A.T @ y, assume_a="pos")

    def solve_lstsq(A, Y, M, intercept, alpha=1e-10):
        if intercept is not None:
            intercept_aggregate = M.T @ intercept
            V = Agregator.regularized_lstsq(A, Y - intercept_aggregate, reg=alpha)
            return V, None
        residual = Y - M.T @ np.linalg.lstsq(M.T, Y, rcond=None)[0]
        residual_A = A - M.T @ np.linalg.lstsq(M.T, A, rcond=None)[0]
        V = Agregator.regularized_lstsq(residual_A, residual, reg=alpha * Y.shape[0])
        intercept = Agregator.regularized_lstsq(M.T, Y - A @ V, reg=alpha * Y.shape[0])
        return V, intercept

    @property
    def model_covariance(self):
        try:
            return self.model_covariance_
        except AttributeError:
            self.model_covariance_ = np.cov(self.Mmat)
            return self.model_covariance_

    def predict(self, X, return_alpha=False):
        return self(X, return_alpha=return_alpha)

    def __call__(self, x, return_alpha=False):
        try:
            K_eval = self.K(self.X, x)
            M = self.call_models(x)
            alpha = np.einsum("ij,ikjm->mk", self.V, K_eval) + self.intercept[:, None]
            pred_aggregate = np.einsum("ij,ij->j", alpha, M)
            if return_alpha:
                return pred_aggregate, alpha
            return pred_aggregate
        except AttributeError:
            raise AttributeError("You must fit the model before calling it")

    def __repr__(self) -> str:
        return f"[{self.kernel_name} Agregator](models={[m.__repr__() for m in self.models]})"

    def __str__(self) -> str:
        return self.__repr__()
