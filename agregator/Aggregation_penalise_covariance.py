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

    def fit(self, X, y, alpha=1.0, cov_regularizer=1.0):
        new = self.__class__(**self.attributes)
        new.X = X
        n = X.shape[0]
        self.y = y
        new._kernel_matrix = new.K(X, X)
        new.Mmat = self.call_models(X)
        to_invert = np.einsum("ijkl,ki->ijl", new.kernel_matrix, new.Mmat)
        to_invert = to_invert.reshape(n, n * new.model_number)
        errors = y[None, :] - new.Mmat
        to_invert_2 = np.einsum("ijkl,ki->ikjl", new.kernel_matrix, errors)
        to_invert_2 = to_invert_2.reshape(n * new.model_number, n * new.model_number)
        V, intercept = Agregator.solve_lstsq(
            A=np.concatenate(
                [to_invert, np.sqrt(cov_regularizer) * to_invert_2], axis=0
            ),
            Y=np.concatenate([y, np.zeros(n * new.model_number)], axis=0),
            M=new.Mmat,
            intercept=None if new.fit_intercept else new.intercept,
            alpha=alpha,
            cov_regularizer=cov_regularizer,
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

    def solve_lstsq(A, Y, M, intercept, alpha=1e-10, cov_regularizer=1e-10):
        return_intercept = intercept is None
        errors = Y[: M.shape[1]][None, :] - M
        mat = np.concatenate(
            [M.T]
            + [
                np.sqrt(cov_regularizer) * np.diag(errors[:, i])
                for i in range(errors.shape[1])
            ],
            axis=0,
        )
        if intercept is None:
            intercept = Agregator.regularized_lstsq(mat, Y, reg=alpha)
        intercept_aggregate = mat @ intercept

        V = Agregator.regularized_lstsq(
            A, Y - intercept_aggregate, reg=alpha * Y.shape[0]
        )
        return V, intercept if return_intercept else None

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
