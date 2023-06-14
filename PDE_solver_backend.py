import numpy as np
from sklearn.metrics import pairwise_distances
from functools import partial
from tqdm import tqdm
from random import shuffle


class PDESolver:
    def __init__(self, X_int, X_boundary, sigma, name=None) -> None:
        self.X_int = X_int
        self.X_boundary = X_boundary
        self.X_all = np.concatenate([X_int, X_boundary])
        self.Nd = X_int.shape[0]
        self.sigma = sigma
        self.memory = {}
        if name is not None:
            self.name = name

        self.X_shared = {}
        self.shared_value = {}
        self.neighbors = []
        self.solved_shared = {}

    def setup_fit(self, f, g, nugget):
        self.nugget = nugget
        self.K_mat = PDESolver.get_kernel_matrix(
            self.X_all, self.Nd, self.sigma, nugget
        )
        self.L = np.linalg.inv(np.linalg.cholesky(self.K_mat))
        self.K_inv = self.L.T @ self.L

        self.K_mat_laplace = PDESolver.get_kernel_matrix_laplace(
            self.X_all, self.Nd, self.sigma, nugget
        )
        self.K_inv_laplace = np.linalg.inv(self.K_mat_laplace)

        self.g_vec = np.array([g(x) for x in self.X_boundary])
        self.f_vec = np.array([f(x) for x in self.X_int])

    def fit_interior(self, f, g, tau, dtau, use_shared, nugget=1e-5):
        if not hasattr(self, "g_vec"):
            self.setup_fit(f, g, nugget)
        z_shared, L = self.get_shared_values(use_shared)
        self.gauss_newton_solution = PDESolver.gauss_newton(
            x_int=self.X_int,
            x_ext=self.X_boundary,
            z_shared=z_shared,
            L=L,
            f_vec=self.f_vec,
            g_vec=self.g_vec,
            tau=tau,
            dtau=dtau,
        )
        # self.a = self.gauss_newton_solution["alpha"]

    def finish_fit(self):
        z_shared = np.concatenate([self.shared_value[n] for n in self.neighbors])
        self.a = np.concatenate(
            [
                self.gauss_newton_solution["z"],
                z_shared,
                self.g_vec,
                self.gauss_newton_solution["z_lap"],
            ]
        )
        self.coeff = self.K_inv @ self.a
        # to be deleted
        # size = self.gauss_newton_solution["z"].shape[0]
        # self.coeff[:size] = 0
        # self.coeff[size:] = self.K_inv_laplace @ np.concatenate(
        #    [z_shared, self.g_vec, self.gauss_newton_solution["z_lap"]]
        # )

    def get_shared_values(self, use_shared):
        if use_shared:
            z_shared = []
            for n in self.neighbors:
                z_shared.append(self.shared_value[n])
            return np.concatenate(z_shared), self.L
        K = PDESolver.get_kernel_matrix(
            np.concatenate([self.X_int, self.X_boundary]),
            self.Nd,
            self.sigma,
            self.nugget,
        )
        return np.empty((0)), np.linalg.inv(np.linalg.cholesky(K))

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

    def get_shared_indices(self, model):
        begin = 0  # self.X_int.shape[0]
        for n, x_shared in self.X_shared.items():
            if n != model:
                begin += x_shared.shape[0]
            else:
                break
        end = begin + self.X_shared[model].shape[0]
        return begin, end

    def setup_joint_fit(models):
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
        total_size = begin
        for model in models:
            indices = []
            matrix = np.zeros((total_size, total_size))
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
                    ] = model.K_inv_laplace[
                        indices[i][0] : indices[i][1], indices[j][0] : indices[j][1]
                    ]
            matrices.append(matrix)
        mat_left = np.sum(
            matrices,
            axis=0,
        )
        for model in models:
            model.joint_fit_utils = {"pairs": pairs, "mat_left": mat_left}
        return pairs, mat_left

    def joint_fit_boundaries(models):
        # we might need to remove the point obeservations to really have convergence
        if not hasattr(models[0], "joint_fit_utils"):
            pairs, mat_left = PDESolver.setup_joint_fit(models)
        else:
            pairs = models[0].joint_fit_utils["pairs"]
            mat_left = models[0].joint_fit_utils["mat_left"]

        targets = []
        for model in models:
            target = np.zeros(mat_left.shape[0])
            model_target = np.concatenate(
                [model.g_vec, model.gauss_newton_solution["z_lap"]]
            )
            for other_model in model.neighbors:
                begin, end = model.get_shared_indices(other_model)
                try:
                    indices1 = pairs[(model, other_model)]
                except:
                    indices1 = pairs[(other_model, model)]
                Nbegin = model.X_int.shape[0]
                Nend = model.X_int.shape[0] + model.X_boundary.shape[0]
                target[indices1[0] : indices1[1]] = (
                    # model.K_inv[begin:end, :Nbegin] @ model.a[:Nbegin]+
                    model.K_inv_laplace[begin:end, -Nend:]
                    @ model_target
                )
            targets.append(target)

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

    def joint_fit(models, f, g, tau, dtau, nugget=1e-5, tol=1e-6):
        dz = {}
        for model in models:
            if not hasattr(model, "g_vec"):
                model.setup_fit(f, g, nugget)
            z_shared, L = model.get_shared_values(False)
            model.gauss_newton_solution, dz[model] = PDESolver.gauss_newton_step(
                x_int=model.X_int,
                x_ext=model.X_boundary,
                z=np.zeros(model.X_int.shape[0]),
                z_shared=z_shared,
                L=L,
                f_vec=model.f_vec,
                g_vec=model.g_vec,
                tau=tau,
                dtau=dtau,
            )
        PDESolver.joint_fit_boundaries(models)
        progress = tqdm()
        dz_norm = 10
        while dz_norm > tol:
            indexes = [k for k in range(len(models))]
            shuffle(indexes)
            for model in [models[k] for k in indexes]:
                z_shared, L = model.get_shared_values(True)
                model.gauss_newton_solution, dz[model] = PDESolver.gauss_newton_step(
                    x_int=model.X_int,
                    x_ext=model.X_boundary,
                    z=model.gauss_newton_solution["z"],
                    z_shared=z_shared,
                    L=L,
                    f_vec=model.f_vec,
                    g_vec=model.g_vec,
                    tau=tau,
                    dtau=dtau,
                )

            dz_norm = np.max([np.linalg.norm(dz[model], np.inf) for m in models])
            progress.set_description(f"Current residual {dz_norm:.3e}")
            progress.update()
            PDESolver.joint_fit_boundaries(models)
        progress.close()

    def covariate_with_other(self, other_GP, x, sigma):
        M = self.find_covariance_matrix(other_GP, sigma)
        v1 = self.K_inv @ PDESolver.get_kernel_vector(
            self.X_all, self.Nd, self.sigma, x
        )
        v2 = other_GP.K_inv @ PDESolver.get_kernel_vector(
            other_GP.X_all, other_GP.Nd, other_GP.sigma, x
        )
        return np.einsum("ij,ij->j", v1, M @ v2)

    def covariate_with_sol(self, x, sigma=None):
        if sigma is None:
            v = PDESolver.get_kernel_vector(self.X_all, self.Nd, self.sigma, x)
            return np.linalg.norm(self.L @ v, axis=0) ** 2
        return self.covariate_with_other(self, x, sigma)

    def find_covariance_matrix(self, other_GP, sigma):
        try:
            return self.memory[other_GP][sigma]
        except:
            M = PDESolver.get_covariance_matrix(
                self.X_all, other_GP.X_all, self.Nd, other_GP.Nd, sigma, self.nugget
            )
            try:
                self.memory[other_GP][sigma] = M
            except:
                self.memory[other_GP] = {sigma: M}
            return M

    def __call__(self, x):
        return np.dot(
            self.coeff, PDESolver.get_kernel_vector(self.X_all, self.Nd, self.sigma, x)
        )

    def laplacian(self, x):
        return np.dot(
            self.coeff,
            PDESolver.get_laplacian_kernel_vector(self.X_all, self.Nd, self.sigma, x),
        )

    def __repr__(self) -> str:
        try:
            return self.name
        except:
            return "PDE Solver with K_mat : " + self.K_mat.__repr__()

    def differential_matrix(z, N_int, N_ext, dtau):
        return np.block(
            [[-np.eye(N_int)], [np.zeros((N_ext, N_int))], [np.diag(dtau(z))]]
        )

    def gauss_newton(x_int, x_ext, z_shared, L, f_vec, g_vec, tau, dtau):
        z = np.zeros(x_int.shape[0])
        alpha = lambda z: np.concatenate([z, z_shared, g_vec, f_vec - tau(z)])
        dz = 3 * np.ones_like(z)
        res = {}
        progress = tqdm()
        while np.linalg.norm(dz, np.inf) > 1e-6:
            res, dz = PDESolver.gauss_newton_step(
                x_int, x_ext, z, z_shared, L, f_vec, g_vec, tau, dtau
            )
            progress.set_description(
                f"Current residual {np.linalg.norm(dz,np.inf):.3e}"
            )
            progress.update()
        progress.close()
        return res

    def gauss_newton_step(x_int, x_ext, z, z_shared, L, f_vec, g_vec, tau, dtau):
        H = PDESolver.differential_matrix(
            z, x_int.shape[0], x_ext.shape[0] + z_shared.shape[0], dtau
        )
        target = L @ np.concatenate([z, z_shared, g_vec, f_vec - tau(z)])
        mat = L @ H
        dz = np.linalg.lstsq(mat, target, rcond=None)[0]
        z += dz
        return {
            "z": z,
            "z_lap": f_vec - tau(z),
            "alpha": np.concatenate([z, z_shared, g_vec, f_vec - tau(z)]),
        }, dz

    def get_kernel_matrix(X, Nd, sigma, nugget):
        return PDESolver.get_covariance_matrix(X, X, Nd, Nd, sigma, nugget)
        distances = pairwise_distances(X) ** 2
        K11 = np.exp(-distances / 2 / sigma**2)
        K12 = -K11[:, :Nd] * (distances[:, :Nd] - 2 * sigma**2) / sigma**4
        K22 = (
            K11[:Nd, :Nd]
            * ((distances[:Nd, :Nd] - 4 * sigma**2) ** 2 - 8 * sigma**4)
            / sigma**8
        )
        return np.block([[K11, K12], [K12.T, K22]])

    def get_covariance_matrix(X, Y, Nd_x, Nd_y, sigma, nugget):
        distances = pairwise_distances(X, Y=Y) ** 2
        nugget_mat = nugget * (distances == 0).astype(int)
        K11 = np.exp(-distances / 2 / sigma**2)
        K12 = -K11[:, :Nd_y] * (distances[:, :Nd_y] - 2 * sigma**2) / sigma**4
        K21 = -K11[:Nd_x, :] * (distances[:Nd_x, :] - 2 * sigma**2) / sigma**4
        K22 = (
            K11[:Nd_x, :Nd_y]
            * ((distances[:Nd_x, :Nd_y] - 4 * sigma**2) ** 2 - 8 * sigma**4)
            / sigma**8
        )
        K11 += nugget_mat
        K22 += nugget_mat[:Nd_x, :Nd_y]
        return np.block([[K11, K12], [K21, K22]])

    def get_kernel_matrix_laplace(X, Nd, sigma, nugget):
        return PDESolver.get_covariance_matrix_laplace(X, X, Nd, Nd, sigma, nugget)

    def get_covariance_matrix_laplace(X, Y, Nd_x, Nd_y, sigma, nugget):
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
        return np.concatenate([K1, K2])

    def get_laplacian_kernel_vector(x_all, Nd, sigma, x):
        distances = pairwise_distances(x_all, Y=x) ** 2
        K0 = np.exp(-distances / 2 / sigma**2)
        K1 = K0 * (distances - 2 * sigma**2) / sigma**4
        K2 = (
            -K0[:Nd]
            * ((distances[:Nd] - 4 * sigma**2) ** 2 - 8 * sigma**4)
            / sigma**8
        )
        return np.concatenate([K1, K2])


class Aggregate:
    def __init__(self, models, sigma, nugget=1e-5) -> None:
        self.models = np.array(models)
        self.nugget = nugget
        self.sigma = sigma

    def covariate_inner_models_with_sol(self, x, sigma):
        return np.array(
            list(map(lambda model: model.covariate_with_sol(x, sigma), self.models))
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
                    lambda model_pair: Aggregate.covariate_models(
                        model_pair[0], model_pair[1], x, self.sigma
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

    def covariate_with_sol(self, x, sigma):
        return np.dot(self.alpha(x), self.covariate_inner_models_with_sol(x, sigma))

    def alpha(self, x):
        # return np.linalg.solve(self.inner_models_cov_matrix(x),self.covariate_inner_models_with_sol(x,self.sigma))
        COV_mat = self.inner_models_cov_matrix(x)
        COV_Y = self.covariate_inner_models_with_sol(x, self.sigma)
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

    def covariate_models(model1, model2, x, sigma):
        if hasattr(model1, "models"):
            covs = np.array(
                list(
                    map(
                        lambda model: Aggregate.covariate_models(
                            model, model2, x, sigma
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
                        lambda model: Aggregate.covariate_models(
                            model, model1, x, sigma
                        ),
                        model2.models,
                    )
                )
            )
            return np.dot(model2.alpha(x), covs)
        return model1.covariate_with_other(model2, x, sigma)
