import numpy as np
import scipy.linalg
from sklearn.decomposition import PCA
from tqdm.notebook import tqdm
from scipy.linalg import lstsq
import time


class AgregatorPoint:
    DEFINING_ATTRIBUTES = ["K", "solvers", "kernel_name"]

    def __init__(self, K, solvers, PCA_covariance_ratio, kernel_name=None):
        self.K = K
        self.solvers = solvers
        self.kernel_name = kernel_name

    @property
    def attributes(self):
        return {k: getattr(self, k) for k in self.DEFINING_ATTRIBUTES}

    @property
    def model_number(self):
        return len(self.solvers)

    @property
    def coef_(self):
        return self.V

    def to_components(self, f):
        assert f.shape[-1] == f.shape[-2] and f.shape[-1] == self.grid_number + 1
        return f.reshape(*(f.shape[:-2]), -1)

    def from_components(self, c):
        return c.reshape((*(c.shape[:-1]), self.grid_number + 1, self.grid_number + 1))

    def call_models(self, F):
        return np.stack(
            [
                np.stack(
                    [solver(f, N_target=self.grid_number) for solver in self.solvers]
                )
                for f in tqdm(F)
            ]
        )

    def F_equality(F1, F2):
        x = np.random.rand(2, 100)
        return np.allclose(
            np.stack(list(map(lambda f: f(x), F1))),
            np.stack(list(map(lambda f: f(x), F2))),
        )

    def fit(self, F, y, alpha=1.0, cov_regularizer=1.0, eps=1e-15):
        """if alpha * y.shape[0] + np.log(eps) < 0:
        print(
            f"alpha {alpha} must be larger than np.log(eps)/y.shape[0] {-np.log(eps)/y.shape[0]}"
        )
        alpha = -np.log(eps) / y.shape[0] + 1e-10
        print(f"alpha set to {alpha}")"""
        n = y.shape[0]
        if not hasattr(self, "y") or not np.allclose(y, self.y):
            self.y = y
            self.grid_number = y.shape[1] - 1
            self.y_components = self.to_components(y)
            self.components_number = self.y_components.shape[1]

        if not hasattr(self, "F") or not AgregatorPoint.F_equality(F, self.F):
            self.F = F
            self.kernel_matrix = self.K(F, F)
            self.Mmat = self.call_models(F)
            self.Mmat_components = self.to_components(self.Mmat)

        to_invert = np.einsum(
            "ijkl,ikm->ijlm", self.kernel_matrix, self.Mmat_components
        )
        to_invert = to_invert.reshape(n, n * self.model_number, self.components_number)
        errors = np.sqrt(
            np.log(eps + (self.y_components[:, None, :] - self.Mmat_components) ** 2)
            - np.log(eps)
        )
        to_invert_2 = np.einsum("ijkl,ikm->ikjlm", self.kernel_matrix, errors)
        to_invert_2 = to_invert_2.reshape(
            n * self.model_number, n * self.model_number, self.components_number
        )
        Vs = []
        for i in tqdm(range(self.components_number)):
            A = np.concatenate(
                [
                    to_invert[:, :, i],
                    np.sqrt(cov_regularizer * to_invert.shape[0])
                    * to_invert_2[:, :, i],
                ],
                axis=0,
            )
            Y_to_solve = np.concatenate(
                [self.y_components[:, i], np.zeros(to_invert_2.shape[1])], axis=0
            )
            V = AgregatorPoint.regularized_lstsq(
                A, Y_to_solve, reg=alpha * to_invert.shape[0]
            )
            Vs.append(V.reshape(n, self.model_number))
        self.V = np.stack(Vs, axis=-1)
        return self

    def regularized_lstsq(A, y, reg=1e-10):
        A2 = np.concatenate([A, np.sqrt(reg) * np.eye(A.shape[1])], axis=0)
        y2 = np.concatenate([y, np.zeros(A.shape[1])], axis=0)
        return lstsq(A2, y2)[0]

    def __call__(self, x, return_alpha=False):
        try:
            K_eval = self.K(self.F, x)
            Mmat_components_eval = self.to_components(self.call_models(x))
            alpha = np.einsum("ijp,ikjm->mkp", self.V, K_eval)
            pred_aggregate = np.einsum("ijp,jip->jp", alpha, Mmat_components_eval)
            print(pred_aggregate.shape, alpha.shape, Mmat_components_eval.shape)
            pred_aggregate = self.from_components(pred_aggregate)
            if return_alpha:
                return pred_aggregate, alpha
            return pred_aggregate
        except AttributeError:
            raise AttributeError("You must fit the model before calling it")

    def __repr__(self) -> str:
        return f"[{self.kernel_name} Agregator](solvers={[m.__repr__() for m in self.solvers]})"

    def __str__(self) -> str:
        return self.__repr__()
