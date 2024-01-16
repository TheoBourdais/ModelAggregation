import numpy as np
import scipy.linalg


class GaussianAggregate:
    def __init__(self, K) -> None:
        self.K = K

    def fit(self, X, y, alpha=1.0):
        self.X = X
        self.y = y
        self.model_number = len(X)
        self.N = sum([x.shape[0] for x in X])
        self.models_K = [
            self.K(
                x=[X[i]],
                y=[X[i]],
                indexes_x=np.array([i + 1]),
                indexes_y=np.array([i + 1]),
            )
            for i in range(self.model_number)
        ]
        self.cho_factors = [
            scipy.linalg.cho_factor(
                model_K + alpha * model_K.shape[0] * np.eye(model_K.shape[0])
            )
            for model_K in self.models_K
        ]
        self.coefs = [
            scipy.linalg.cho_solve(
                cho_factor,
                y[i],
            )
            for i, cho_factor in enumerate(self.cho_factors)
        ]
        self.kernel_matrix = self.K(
            x=X,
            y=X,
            indexes_x=np.arange(1, self.model_number + 1),
            indexes_y=np.arange(1, self.model_number + 1),
        )

    def call_models(self, K_call=None, x=None):
        if K_call is None:
            K_call = [
                self.K(
                    x=[x],
                    y=[self.X[i]],
                    indexes_x=np.array([i + 1]),
                    indexes_y=np.array([i + 1]),
                )
                for i in range(self.model_number)
            ]
        return np.stack([K_call[i] @ self.coefs[i] for i in range(self.model_number)])

    def covariances(self, K_call=None, K_Y=None, x=None):
        if K_call is None or K_Y is None:
            K_call = [
                self.K(
                    x=[x],
                    y=[self.X[i]],
                    indexes_x=np.array([i + 1]),
                    indexes_y=np.array([i + 1]),
                )
                for i in range(self.model_number)
            ]
            K_Y = [
                self.K(
                    x=[x],
                    y=[self.X[i]],
                    indexes_x=np.array([0]),
                    indexes_y=np.array([i + 1]),
                )
                for i in range(self.model_number)
            ]
        inv = [
            scipy.linalg.cho_solve(self.cho_factors[i], K_call[i].T)
            for i in range(self.model_number)
        ]
        cov_mats = np.zeros((K_call[0].shape[0], self.model_number, self.model_number))
        index_i = 0
        for i, xi in enumerate(self.X):
            index_j = 0
            for j, xj in enumerate(self.X):
                cov_mats[:, i, j] = np.einsum(
                    "kn,ln,kl->n",
                    inv[i],
                    inv[j],
                    self.kernel_matrix[
                        index_i : index_i + xi.shape[0], index_j : index_j + xj.shape[0]
                    ],
                )
                index_j += xj.shape[0]
            index_i += xi.shape[0]
        cov_Y = np.stack(
            [np.einsum("kn,nk->n", inv[i], K_Y[i]) for i in range(self.model_number)],
            axis=1,
        )
        return cov_mats, cov_Y

    def predict(self, x, return_alpha=False):
        return self(x, return_alpha)

    def __call__(self, x, return_alpha=False):
        try:
            print(x.shape)
            K_call = [
                self.K(
                    x=[x],
                    y=[self.X[i]],
                    indexes_x=np.array([i + 1]),
                    indexes_y=np.array([i + 1]),
                )
                for i in range(self.model_number)
            ]
            K_Y = [
                self.K(
                    x=[x],
                    y=[self.X[i]],
                    indexes_x=np.array([0]),
                    indexes_y=np.array([i + 1]),
                )
                for i in range(self.model_number)
            ]
            cov_mats, cov_Y = self.covariances(K_call=K_call, K_Y=K_Y)
            Mx = self.call_models(K_call=K_call)
            alpha_coeff = np.linalg.solve(
                cov_mats + 1e-6 * np.eye(self.model_number)[None, :, :], cov_Y
            )
            if return_alpha:
                return np.einsum("ni,in->n", alpha_coeff, Mx), alpha_coeff
            return np.einsum("ni,in->n", alpha_coeff, Mx)

        except AttributeError:
            raise AttributeError("You must fit the model before calling it")

    def __repr__(self) -> str:
        return f"GaussianAggregate {hash(self._kernel_matrix.tobytes())}"

    def __str__(self) -> str:
        return self.__repr__()

    """def fit2(self, X, y, alpha=1.0):
        self.X = X
        self.y = y
        self.model_number = len(X)
        self.N = sum([x.shape[0] for x in X])
        self._kernel_matrix = self.K(
            x=X,
            y=X,
            indexes_x=np.arange(1, self.model_number + 1),
            indexes_y=np.arange(1, self.model_number + 1),
        )
        self._cho = scipy.linalg.cho_factor(
            self._kernel_matrix + alpha * self.N * np.eye(self.N)
        )
        self._coef = scipy.linalg.cho_solve(self._cho, np.concatenate(y))
        return self"""
