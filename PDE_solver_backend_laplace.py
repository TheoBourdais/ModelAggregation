import numpy as np
from sklearn.metrics import pairwise_distances
from functools import partial
from tqdm import tqdm
import itertools
import cvxpy as cp
import matplotlib.pyplot as plt


class PDESolverLaplace:
    def __init__(self, X_int, X_boundary, sigma, name=None) -> None:
        self.X_int = X_int
        self.X_boundary = X_boundary
        self.X_shared = {}
        self.shared_value = {}
        self.neighbors = []
        self.X_all = np.concatenate([X_int, X_boundary])
        self.Nd = X_int.shape[0]
        self.sigma = sigma
        self.memory = {}
        if name is not None:
            self.name = name

        self.solved_shared = {}

    def add_neighbors(self, models, x_shareds):
        for model, x_shared in zip(models, x_shareds):
            try:
                assert self.X_shared[model] == x_shared
            except KeyError:
                self.X_shared[model] = x_shared
        self.neighbors = list(self.X_shared.keys())
        self.X_all = np.concatenate(
            [self.X_int] + list(self.X_shared.values()) + [self.X_boundary]
        )
        self.solved_shared = {key: False for key in self.neighbors}

    def setup_fit(self, f, g, nugget=1e-5):
        self.nugget = nugget
        self.K_mat = PDESolverLaplace.get_kernel_matrix(
            self.X_all, self.Nd, self.sigma, nugget
        )
        self.L = np.linalg.inv(np.linalg.cholesky(self.K_mat))
        self.K_inv = self.L.T @ self.L
        self.g_vec = np.array([g(x) for x in self.X_boundary])
        self.f_vec = np.array([f(x) for x in self.X_int])
        self.target_values = np.concatenate([self.g_vec, self.f_vec])

    def finish_fit(self):
        self.a = np.concatenate(
            [self.shared_value[n] for n in self.neighbors] + [self.target_values]
        )
        self.coeff = self.K_inv @ self.a

    def covariate_with_other(self, other_GP, x):
        M = self.find_covariance_matrix(other_GP)
        v1 = self.K_inv @ PDESolverLaplace.get_kernel_vector(
            self.X_all, self.Nd, self.sigma, x
        )
        v2 = other_GP.K_inv @ PDESolverLaplace.get_kernel_vector(
            other_GP.X_all, other_GP.Nd, other_GP.sigma, x
        )
        return np.einsum("ij,ij->j", v1, M @ v2)

    def covariate_with_sol(self, x):
        v = PDESolverLaplace.get_kernel_vector(self.X_all, self.Nd, self.sigma, x)
        return np.linalg.norm(self.L @ v, axis=0) ** 2

    def find_covariance_matrix(self, other_GP):
        try:
            return self.memory[other_GP]
        except:
            M = PDESolverLaplace.get_covariance_matrix(
                self.X_all,
                other_GP.X_all,
                self.Nd,
                other_GP.Nd,
                self.sigma,
                self.nugget,
            )
            self.memory[other_GP] = M
            return M

    def get_shared_indices(self, model):
        begin = 0
        for n, x_shared in self.X_shared.items():
            if n != model:
                begin += x_shared.shape[0]
            else:
                break
        end = begin + self.X_shared[model].shape[0]
        return begin, end

    def __call__(self, x):
        return np.dot(
            self.coeff,
            PDESolverLaplace.get_kernel_vector(self.X_all, self.Nd, self.sigma, x),
        )

    def laplacian(self, x):
        return np.dot(
            self.coeff,
            PDESolverLaplace.get_laplacian_kernel_vector(
                self.X_all, self.Nd, self.sigma, x
            ),
        )

    def __repr__(self) -> str:
        try:
            return self.name
        except:
            return "PDE Solver with K_mat : " + self.K_mat.__repr__()

    def get_target_values(x_int, x_ext, f, g):
        g_vec = np.array([g(x) for x in x_ext])
        f_vec = np.array([f(x) for x in x_int])
        return np.concatenate([g_vec, f_vec])

    def target_for_fit(self, pairs_to_ignore, index1, index2):
        res = np.zeros(index2 - index1)
        index = 0
        for n, X in self.X_shared.items():
            if (self, n) not in pairs_to_ignore and (n, self) not in pairs_to_ignore:
                try:
                    res += (
                        self.K_inv[index1:index2, index : index + X.shape[0]]
                        @ self.shared_value[n]
                    )
                except KeyError:
                    pass
            index += X.shape[0]
        res += (
            self.K_inv[index1:index2, -self.target_values.shape[0] :]
            @ self.target_values
        )
        return res

    def fit(self):
        pairs = {}
        begin = 0
        size = 0
        models = []
        for model in self.neighbors:
            if not self.solved_shared[model]:
                pairs[(self, model)] = (begin, begin + self.X_shared[model].shape[0])
                models.append(model)
                begin = begin + self.X_shared[model].shape[0]
                model.shared_value.pop(self, None)
                self.shared_value.pop(model, None)
        indices = []
        size = begin
        matrices = [np.zeros((size, size))]
        targets = [np.zeros(size)]
        for model in models:
            begin, end = self.get_shared_indices(model)
            indices.append((begin, end))
        for i, m1 in enumerate(models):
            matrix = np.zeros((size, size))
            index = pairs[(self, m1)]
            index_in_model = m1.get_shared_indices(self)
            matrix[index[0] : index[1], index[0] : index[1]] = m1.K_inv[
                index_in_model[0] : index_in_model[1],
                index_in_model[0] : index_in_model[1],
            ]
            target = np.zeros(size)
            target[index[0] : index[1]] = m1.target_for_fit(
                pairs, index_in_model[0], index_in_model[1]
            )
            matrices.append(matrix)
            targets.append(target)

            for j, m2 in enumerate(models):
                index2 = pairs[(self, m2)]
                matrices[0][index[0] : index[1], index2[0] : index2[1]] = self.K_inv[
                    indices[i][0] : indices[i][1], indices[j][0] : indices[j][1]
                ]
            targets[0][index[0] : index[1]] = self.target_for_fit(
                pairs, indices[i][0], indices[i][1]
            )

        mat_left = np.sum(
            matrices,
            axis=0,
        )
        target = np.sum(
            targets,
            axis=0,
        )
        shared_value = -np.linalg.solve(mat_left, target)
        for model in models:
            index = pairs[(self, model)]
            self.shared_value[model] = shared_value[index[0] : index[1]]
            model.shared_value[self] = shared_value[index[0] : index[1]]
            self.solved_shared[model] = True
            model.solved_shared[self] = True
        self.finish_fit()

    def joint_fit(models, f, g, nugget=1e-5):
        for model in models:
            model.setup_fit(f, g, nugget)

        pairs = {}
        begin = 0
        for model in models:
            for other_model in model.neighbors:
                if not ((other_model, model) in pairs or (model, other_model) in pairs):
                    pairs[(model, other_model)] = (
                        begin,
                        begin + model.X_shared[other_model].shape[0],
                    )
                    begin = begin + model.X_shared[other_model].shape[0]

        matrices = []
        targets = []
        total_size = begin
        for model in models:
            indices = []
            matrix = np.zeros((total_size, total_size))
            target = np.zeros(total_size)
            for other_model in model.neighbors:
                begin, end = model.get_shared_indices(other_model)
                indices.append((begin, end))
            for i, m1 in enumerate(model.neighbors):
                for j, m2 in enumerate(model.neighbors):
                    try:
                        indices1 = pairs[(model, m1)]
                    except:
                        indices1 = pairs[(m1, model)]
                    try:
                        indices2 = pairs[(model, m2)]
                    except:
                        indices2 = pairs[(m2, model)]
                    matrix[
                        indices1[0] : indices1[1], indices2[0] : indices2[1]
                    ] = model.K_inv[
                        indices[i][0] : indices[i][1], indices[j][0] : indices[j][1]
                    ]
                target[indices1[0] : indices1[1]] = (
                    model.K_inv[
                        indices[i][0] : indices[i][1], -model.target_values.shape[0] :
                    ]
                    @ model.target_values
                )
            matrices.append(matrix)
            targets.append(target)

        mat_left = np.sum(
            matrices,
            axis=0,
        )
        target = np.sum(
            targets,
            axis=0,
        )
        shared_value = -np.linalg.solve(mat_left, target)

        for model in models:
            model.shared_value = {}
            for n in model.neighbors:
                try:
                    indices = pairs[(model, n)]
                except:
                    indices = pairs[(n, model)]
                model.shared_value[n] = shared_value[indices[0] : indices[1]]
            model.finish_fit()

    def get_kernel_matrix(X, Nd, sigma, nugget):
        return PDESolverLaplace.get_covariance_matrix(X, X, Nd, Nd, sigma, nugget)

    def get_covariance_matrix(X, Y, Nd_x, Nd_y, sigma, nugget):
        distances = pairwise_distances(X, Y=Y) ** 2
        nugget_mat = nugget * (distances == 0).astype(int)
        K11 = np.exp(-distances / 2 / sigma**2)
        K12 = (
            -K11[Nd_x:, :Nd_y] * (distances[Nd_x:, :Nd_y] - 2 * sigma**2) / sigma**4
        )
        K21 = (
            -K11[:Nd_x, Nd_y:] * (distances[:Nd_x, Nd_y:] - 2 * sigma**2) / sigma**4
        )
        K22 = (
            K11[:Nd_x, :Nd_y]
            * ((distances[:Nd_x, :Nd_y] - 4 * sigma**2) ** 2 - 8 * sigma**4)
            / sigma**8
        )
        K11 += nugget_mat
        K22 += nugget_mat[:Nd_x, :Nd_y]
        K11 = K11[Nd_x:, Nd_y:]

        return np.block([[K11, K12], [K21, K22]])

    def get_kernel_vector(X, Nd, sigma, x):
        distances = pairwise_distances(X, Y=x) ** 2
        K1 = np.exp(-distances / 2 / sigma**2)
        K2 = -K1[:Nd] * (distances[:Nd] - 2 * sigma**2) / sigma**4
        return np.concatenate([K1[Nd:], K2])

    def get_laplacian_kernel_vector(x_all, Nd, sigma, x):
        distances = pairwise_distances(x_all, Y=x) ** 2
        K0 = np.exp(-distances / 2 / sigma**2)
        K1 = K0 * (distances - 2 * sigma**2) / sigma**4
        K2 = (
            -K0[:Nd]
            * ((distances[:Nd] - 4 * sigma**2) ** 2 - 8 * sigma**4)
            / sigma**8
        )
        return np.concatenate([K1[Nd:], K2])


class AggregateLaplace:
    def __init__(self, models, nugget=1e-5) -> None:
        self.models = np.array(models)
        self.nugget = nugget

    def covariate_inner_models_with_sol(self, x):
        return np.array(
            list(map(lambda model: model.covariate_with_sol(x), self.models))
        ).T

    def inner_models_cov_matrix(self, x):
        triangular_indices = np.tril_indices(self.models.shape[0])
        pairs = np.stack(
            [self.models[triangular_indices[0]], self.models[triangular_indices[1]]],
            axis=-1,
        )
        covs = np.array(
            list(
                map(
                    lambda model_pair: AggregateLaplace.covariate_models(
                        model_pair[0], model_pair[1], x
                    ),
                    pairs,
                )
            )
        )
        cov_mat = np.zeros((self.models.shape[0], self.models.shape[0], x.shape[0]))
        cov_mat[triangular_indices] = covs
        cov_mat = cov_mat + np.transpose(cov_mat, (1, 0, 2))
        indexes = np.arange(self.models.shape[0])
        cov_mat[indexes, indexes, :] -= (
            np.diagonal(cov_mat, axis1=0, axis2=1).T / 2 - self.nugget
        )
        return cov_mat.T

    def covariate_with_sol(self, x):
        return np.dot(self.alpha(x), self.covariate_inner_models_with_sol(x))

    def alpha(self, x):
        # return np.linalg.solve(self.inner_models_cov_matrix(x),self.covariate_inner_models_with_sol(x,self.sigma))
        COV_mat = self.inner_models_cov_matrix(x)
        COV_Y = self.covariate_inner_models_with_sol(x)
        print("COV mat", COV_mat)
        print("COV Y", COV_Y)
        alphas = np.stack(
            list(
                map(lambda A, B: np.linalg.lstsq(A, B, rcond=None)[0], COV_mat, COV_Y)
            ),
            axis=0,
        )
        return alphas

    def __call__(self, x):
        M = np.array(list(map(lambda model: model(x), self.models))).T
        print("M", M)
        alpha = self.alpha(x)
        print("alpha", alpha)
        return np.einsum("ij,ij->i", alpha, M)

    def covariate_models(model1, model2, x):
        if hasattr(model1, "models"):
            covs = np.array(
                list(
                    map(
                        lambda model: AggregateLaplace.covariate_models(
                            model, model2, x
                        ),
                        model1.models,
                    )
                )
            )
            return np.dot(model1.alpha(x), covs)
        if hasattr(model2, "models"):
            covs = np.array(
                list(
                    map(
                        lambda model: AggregateLaplace.covariate_models(
                            model, model1, x
                        ),
                        model2.models,
                    )
                )
            )
            return np.dot(model2.alpha(x), covs)
        return model1.covariate_with_other(model2, x)
