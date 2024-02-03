import numpy as np
import scipy.linalg
from sklearn.decomposition import PCA
from tqdm.notebook import tqdm
from scipy.linalg import lstsq


class Agregator:
    DEFINING_ATTRIBUTES = ["K", "solvers", "kernel_name"]

    def __init__(self, K, solvers, PCA_covariance_ratio, kernel_name=None):
        self.K = K
        self.solvers = solvers
        self.kernel_name = kernel_name
        self.pca = PCA(n_components=PCA_covariance_ratio)

    @property
    def attributes(self):
        return {k: getattr(self, k) for k in self.DEFINING_ATTRIBUTES}

    @property
    def model_number(self):
        return len(self.solvers)

    @property
    def coef_(self):
        return self.V

    @property
    def kernel_matrix(self):
        try:
            return self._kernel_matrix
        except AttributeError:
            raise AttributeError("You must fit the model before calling its intercept")

    def to_components(self, f):
        assert f.shape[-1] == f.shape[-2] and f.shape[-1] == self.grid_number + 1
        return self.pca.fit_transform(f.reshape(np.prod(f.shape[:-2]), -1)).reshape(
            (*f.shape[:-2], self.components_number)
        )

    def from_components(self, c):
        return self.pca.inverse_transform(
            c.reshape(-1, self.components_number)
        ).reshape((*c.shape[:-1], self.grid_number, self.grid_number))

    def call_models(self, F):
        return np.stack(
            [
                np.stack(
                    [solver(f, N_target=self.grid_number) for solver in self.solvers]
                )
                for f in tqdm(F)
            ]
        )

    def fit(self, F, y, alpha=1.0, cov_regularizer=1.0):
        self.F = F
        n = y.shape[0]
        self.y = y
        self.grid_number = y.shape[1] - 1
        self.y_components = self.pca.fit_transform(y.reshape(n, -1))
        self.components_number = self.y_components.shape[1]
        self._kernel_matrix = self.K(F, F)
        self.Mmat = self.call_models(F)
        self.Mmat_components = self.to_components(self.Mmat)
        to_invert = np.einsum(
            "ijkl,ikm->ijlm", self.kernel_matrix, self.Mmat_components
        )
        to_invert = to_invert.reshape(n, n * self.model_number, self.components_number)
        errors = y[:, None, :] - self.Mmat_components
        to_invert_2 = np.einsum("ijkl,ikm->ikjlm", self.kernel_matrix, errors)
        to_invert_2 = to_invert_2.reshape(
            n * self.model_number, n * self.model_number, self.components_number
        )
        Vs = []
        for i in range(self.components_number):
            A = np.concatenate(
                [to_invert[:, :, i], np.sqrt(cov_regularizer) * to_invert_2[:, :, i]],
                axis=0,
            )
            Y_to_solve = np.concatenate(
                [self.Y_components[:, i], np.zeros(to_invert_2.shape[1])], axis=0
            )
            V = Agregator.regularized_lstsq(
                A, Y_to_solve, reg=alpha * Y_to_solve.shape[0]
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
            Mmat_components = self.to_components(self.call_models(x))
            alpha = np.einsum("ijp,ikjm->mkp", self.V, K_eval)
            pred_aggregate = np.einsum("ijp,jip->jp", alpha, Mmat_components)
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
