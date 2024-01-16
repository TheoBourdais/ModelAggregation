import numpy as np
import scipy.linalg


class GaussianAggregateSimple:
    def __init__(self, K) -> None:
        self.K = K

    def fit(self, X, y, alpha=1.0):
        self.X = X
        self.y = y
        self.model_number = len(X)
        self.N = sum([x.shape[0] for x in X])
        self.kernel_matrix = self.K(
            x=X,
            y=X,
            indexes_x=np.arange(1, self.model_number + 1),
            indexes_y=np.arange(1, self.model_number + 1),
        )
        self.cho_factor = scipy.linalg.cho_factor(
            self.kernel_matrix + alpha * self.N * np.eye(self.N)
        )
        self.coef = scipy.linalg.cho_solve(self.cho_factor, np.concatenate(y))

    def predict(self, x):
        return self(x)

    def __call__(self, x):
        try:
            K_call = self.K(
                x=[x],
                y=self.X,
                indexes_x=np.array([0]),
                indexes_y=np.arange(1, self.model_number + 1),
            )
            return K_call @ self.coef

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
