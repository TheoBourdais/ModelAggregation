import numpy as np
from sklearn.metrics import pairwise_distances
from functools import partial
from tqdm import tqdm
import itertools
import cvxpy as cp
import matplotlib.pyplot as plt
from sympy import symbols, Matrix, exp, diff, lambdify


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
        try:
            self.all_shared = np.concatenate(list(self.X_shared.values()))
        except:
            self.all_shared = np.empty((0, 2))
        self.K_mat = PDESolverLaplace.get_kernel_matrix(
            self.X_int,
            self.all_shared,
            self.X_boundary,
            self.sigma,
            nugget,
        )
        self.L = np.linalg.inv(np.linalg.cholesky(self.K_mat))
        self.K_inv = self.L.T @ self.L
        self.g_vec = np.array([g(x) for x in self.X_boundary])
        self.f_vec = np.array([f(x) for x in self.X_int])
        size_shared = self.all_shared.shape[0]
        vec = np.concatenate(
            [np.zeros(size_shared), self.g_vec, np.zeros(2 * size_shared), self.f_vec]
        )
        self.storage = MatVecStorage(
            self.K_inv, vec, self.X_int, self.X_shared, self.X_boundary
        )

    def finish_fit(self):
        self.a = np.concatenate(
            [self.shared_value[n]["dirac"] for n in self.neighbors]
            + [self.g_vec]
            + [self.shared_value[n]["dx"] for n in self.neighbors]
            + [self.shared_value[n]["dy"] for n in self.neighbors]
            + [self.f_vec]
        )
        self.storage.vec = self.a
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

    def __call__(self, x):
        return np.dot(
            self.coeff,
            PDESolverLaplace.get_kernel_vector(
                self.X_int,
                self.all_shared,
                self.X_boundary,
                self.sigma,
                x,
            ),
        )

    def laplacian(self, x):
        return np.dot(
            self.coeff,
            PDESolverLaplace.get_laplacian_kernel_vector(
                self.X_int,
                self.all_shared,
                self.X_boundary,
                self.sigma,
                x,
            ),
        )

    def __repr__(self) -> str:
        try:
            return self.name
        except:
            return "PDE Solver with K_mat : " + self.K_mat.__repr__()

    def joint_fit(models, f, g, nugget=1e-5):
        for model in models:
            model.setup_fit(f, g, nugget)
        operators = ["dirac", "dx", "dy"]
        pairs = {}
        begin = 0
        for operator in operators:
            for model in models:
                for other_model in model.neighbors:
                    if not (
                        (operator, other_model, model) in pairs
                        or (operator, model, other_model) in pairs
                    ):
                        pairs[(operator, model, other_model)] = (
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
            for operator in operators:
                for i, m1 in enumerate(model.neighbors):
                    begin1, end1 = model.storage.get_index(m1, operator)
                    begin_ext, end_ext = model.storage.get_index("ext", operator)
                    begin_int, end_int = model.storage.get_index("int", operator)
                    try:
                        indices1 = pairs[(operator, model, m1)]
                    except:
                        indices1 = pairs[(operator, m1, model)]
                    target[indices1[0] : indices1[1]] += (
                        model.K_inv[begin1:end1, begin_ext:end_ext] @ model.g_vec
                    )
                    target[indices1[0] : indices1[1]] += (
                        model.K_inv[begin1:end1, begin_int:end_int] @ model.f_vec
                    )

                    for j, m2 in enumerate(model.neighbors):
                        for operator2 in operators:
                            try:
                                indices2 = pairs[(operator2, model, m2)]
                            except:
                                indices2 = pairs[(operator2, m2, model)]
                            begin2, end2 = model.storage.get_index(m2, operator2)
                            matrix[
                                indices1[0] : indices1[1], indices2[0] : indices2[1]
                            ] += model.K_inv[begin1:end1, begin2:end2]

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
                model.shared_value[n] = {}
                for operator in operators:
                    try:
                        indices = pairs[(operator, model, n)]
                    except:
                        indices = pairs[(operator, n, model)]
                    model.shared_value[n][operator] = shared_value[
                        indices[0] : indices[1]
                    ]
            model.finish_fit()
        return mat_left, target, pairs

    def get_kernel_matrix(X_int, X_shared, X_ext, sigma, nugget):
        k2 = FasterGaussianKernel(sigma, X_int, X_shared, X_ext)
        dirac_mat = k2.get_dirac()
        dx1 = k2.get_dx()
        dy1 = k2.get_dy()
        lap1 = k2.get_lap()
        ddx1 = k2.get_dx1dx2()
        dydx1 = k2.get_dy1dx2()
        dxlap = k2.get_lap1dx2()
        ddy = k2.get_dy1dy2()
        dylap = k2.get_lap1dy2()
        laplap = k2.get_lap1lap2()
        res = [
            [dirac_mat, dx1.T, dy1.T, lap1.T],
            [dx1, ddx1, dydx1.T, dxlap.T],
            [dy1, dydx1, ddy, dylap.T],
            [lap1, dxlap, dylap, laplap],
        ]
        res = np.block(res)
        return res + np.eye(res.shape[0]) * nugget

    def get_kernel_vector(X_int, X_shared, X_ext, sigma, x):
        k2 = GaussianKernelVectorDirac(sigma, X_int, X_shared, X_ext, x)
        dirac_mat = k2.get_dirac()
        dx1 = k2.get_dx()
        dy1 = k2.get_dy()
        lap1 = k2.get_lap()
        return np.concatenate([dirac_mat, dx1, dy1, lap1])

    def get_laplacian_kernel_vector(X_int, X_shared, X_ext, sigma, x):
        k2 = GaussianKernelVectorLap(sigma, X_int, X_shared, X_ext, x)
        lap = k2.get_lap()
        dxlap = k2.get_lap1dx2()
        dylap = k2.get_lap1dy2()
        laplap = k2.get_lap1lap2()

        return np.concatenate([lap, dxlap, dylap, laplap])


class gaussianKernel:
    def __init__(self, s):
        self.sigma = s
        laplacian_1 = lambda f: -diff(f, x1, x1) - diff(f, y1, y1)
        laplacian_2 = lambda f: -diff(f, x2, x2) - diff(f, y2, y2)
        dx1 = lambda f: diff(f, x1)
        dx2 = lambda f: diff(f, x2)
        dy1 = lambda f: diff(f, y1)
        dy2 = lambda f: diff(f, y2)
        x1, x2, y1, y2, sigma = symbols("x_1 x_2 y_1 y_2 \sigma")
        p1 = Matrix([x1, y1])
        p2 = Matrix([x2, y2])
        self.gauss = exp(-(p1 - p2).dot(p1 - p2) / (2 * sigma**2))
        self.dxgauss = dx1(self.gauss)
        self.dygauss = dy1(self.gauss)
        self.lapgauss = laplacian_1(self.gauss)
        self.ddxgauss = dx1(dx2(self.gauss))
        self.dxygauss = dx2(dy1(self.gauss))
        self.dxlapgauss = dx2(self.lapgauss)
        self.ddygauss = dy2(dy1(self.gauss))
        self.dylapgauss = dy2(self.lapgauss)
        self.laplapgauss = laplacian_2(self.lapgauss)

        self.gauss = lambdify((x1, y1, x2, y2, sigma), self.gauss)
        self.dxgauss = lambdify((x1, y1, x2, y2, sigma), self.dxgauss)
        self.dygauss = lambdify((x1, y1, x2, y2, sigma), self.dygauss)
        self.lapgauss = lambdify((x1, y1, x2, y2, sigma), self.lapgauss)
        self.ddxgauss = lambdify((x1, y1, x2, y2, sigma), self.ddxgauss)
        self.dxygauss = lambdify((x1, y1, x2, y2, sigma), self.dxygauss)
        self.dxlapgauss = lambdify((x1, y1, x2, y2, sigma), self.dxlapgauss)
        self.ddygauss = lambdify((x1, y1, x2, y2, sigma), self.ddygauss)
        self.dylapgauss = lambdify((x1, y1, x2, y2, sigma), self.dylapgauss)
        self.laplapgauss = lambdify((x1, y1, x2, y2, sigma), self.laplapgauss)

    def evaluate(self, func, X1, X2):
        if X1.shape[0] * X2.shape[0] == 0:
            return np.zeros((X1.shape[0], X2.shape[0]))
        row_func = lambda x1: np.array(
            list(map(lambda x2: func(x1[0], x1[1], x2[0], x2[1], self.sigma), X2))
        )
        return np.stack(list(map(row_func, X1)))


class GaussianKernelVectorDirac:
    def __init__(self, s, X_int, X_shared, X_boundary, x):
        self.sigma = s
        self.x = x
        self.X_int = X_int
        self.X_shared = X_shared
        self.X_boundary = X_boundary
        self.X_dirac = np.concatenate([X_shared, X_boundary])
        self.size_dirac = self.X_dirac.shape[0]
        self.all = np.concatenate([X_shared, X_boundary, X_int])
        self.distances = pairwise_distances(self.all, Y=x) ** 2
        self.exp_d = np.exp(-self.distances / 2 / s**2)

    def get_dirac(self):
        return self.exp_d[: self.size_dirac, :]

    def get_dx(self):
        diff_mat = np.expand_dims(self.x[:, 0], 0) - np.expand_dims(
            self.X_shared[:, 0], 1
        )
        return self.exp_d[: self.X_shared.shape[0], :] * diff_mat / self.sigma**2

    def get_dy(self):
        diff_mat = np.expand_dims(self.x[:, 1], 0) - np.expand_dims(
            self.X_shared[:, 1], 1
        )
        return self.exp_d[: self.X_shared.shape[0], :] * diff_mat / self.sigma**2

    def get_lap(self):
        to_mult = (
            self.distances[-self.X_int.shape[0] :, :] - 2 * self.sigma**2
        ) / self.sigma**4
        return -self.exp_d[-self.X_int.shape[0] :, :] * to_mult


class GaussianKernelVectorLap:
    def __init__(self, s, X_int, X_shared, X_boundary, x):
        self.sigma = s
        self.x = x
        self.X_int = X_int
        self.X_shared = X_shared
        self.X_boundary = X_boundary
        self.X_dirac = np.concatenate([X_shared, X_boundary])
        self.size_dirac = self.X_dirac.shape[0]
        self.all = np.concatenate([X_shared, X_boundary, X_int])
        self.distances = pairwise_distances(self.all, Y=x) ** 2
        self.exp_d = np.exp(-self.distances / 2 / s**2)

    def get_lap(self):
        to_mult = (
            self.distances[: self.size_dirac, :] - 2 * self.sigma**2
        ) / self.sigma**4
        return -self.exp_d[: self.size_dirac, :] * to_mult

    def get_lap1dx2(self):
        lap_like = (
            self.distances[: self.X_shared.shape[0], :] - 4 * self.sigma**2
        ) / self.sigma**6
        diff = np.expand_dims(self.X_shared[:, 0], 1) - np.expand_dims(self.x[:, 0], 0)
        return self.exp_d[: self.X_shared.shape[0], :] * lap_like * diff

    def get_lap1dy2(self):
        lap_like = (
            self.distances[: self.X_shared.shape[0], :] - 4 * self.sigma**2
        ) / self.sigma**6
        diff = np.expand_dims(self.X_shared[:, 1], 1) - np.expand_dims(self.x[:, 1], 0)
        return self.exp_d[: self.X_shared.shape[0], :] * lap_like * diff

    def get_lap1lap2(self):
        dist = (
            (self.distances[-self.X_int.shape[0] :, :] - 4 * self.sigma**2) ** 2
            - 8 * self.sigma**4
        ) / self.sigma**8
        return self.exp_d[-self.X_int.shape[0] :, :] * dist


class FasterGaussianKernel:
    def __init__(self, s, X_int, X_shared, X_boundary):
        self.sigma = s
        self.X_int = X_int
        self.X_shared = X_shared
        self.X_boundary = X_boundary
        self.X_dirac = np.concatenate([X_shared, X_boundary])
        self.size_dirac = self.X_dirac.shape[0]
        self.all = np.concatenate([X_shared, X_boundary, X_int])
        self.distances = pairwise_distances(self.all) ** 2
        self.exp_d = np.exp(-self.distances / 2 / s**2)

    def get_dirac(self):
        return self.exp_d[: self.size_dirac, : self.size_dirac]

    def get_dx(self):
        diff_mat = np.expand_dims(self.X_dirac[:, 0], 0) - np.expand_dims(
            self.X_shared[:, 0], 1
        )
        return (
            self.exp_d[: self.X_shared.shape[0], : self.size_dirac]
            * diff_mat
            / self.sigma**2
        )

    def get_dy(self):
        diff_mat = np.expand_dims(self.X_dirac[:, 1], 0) - np.expand_dims(
            self.X_shared[:, 1], 1
        )
        return (
            self.exp_d[: self.X_shared.shape[0], : self.size_dirac]
            * diff_mat
            / self.sigma**2
        )

    def get_lap(self):
        to_mult = (
            self.distances[-self.X_int.shape[0] :, : self.size_dirac]
            - 2 * self.sigma**2
        ) / self.sigma**4
        return -self.exp_d[-self.X_int.shape[0] :, : self.size_dirac] * to_mult

    def get_dx1dx2(self):
        diff_mat = np.expand_dims(self.X_shared[:, 0], 0) - np.expand_dims(
            self.X_shared[:, 0], 1
        )
        to_mult = (self.sigma**2 - diff_mat**2) / self.sigma**4
        return to_mult * self.exp_d[: self.X_shared.shape[0], : self.X_shared.shape[0]]

    def get_dy1dx2(self):
        diff_mat_x = np.expand_dims(self.X_shared[:, 0], 0) - np.expand_dims(
            self.X_shared[:, 0], 1
        )
        diff_mat_y = np.expand_dims(self.X_shared[:, 1], 0) - np.expand_dims(
            self.X_shared[:, 1], 1
        )
        return (
            -self.exp_d[: self.X_shared.shape[0], : self.X_shared.shape[0]]
            * diff_mat_x
            * diff_mat_y
            / self.sigma**4
        )

    def get_lap1dx2(self):
        lap_like = (
            self.distances[-self.X_int.shape[0] :, : self.X_shared.shape[0]]
            - 4 * self.sigma**2
        ) / self.sigma**6
        diff = np.expand_dims(self.X_int[:, 0], 1) - np.expand_dims(
            self.X_shared[:, 0], 0
        )
        return -(
            self.exp_d[-self.X_int.shape[0] :, : self.X_shared.shape[0]]
            * lap_like
            * diff
        )

    def get_dy1dy2(self):
        diff_mat = np.expand_dims(self.X_shared[:, 1], 0) - np.expand_dims(
            self.X_shared[:, 1], 1
        )
        to_mult = (self.sigma**2 - diff_mat**2) / self.sigma**4
        return to_mult * self.exp_d[: self.X_shared.shape[0], : self.X_shared.shape[0]]

    def get_lap1dy2(self):
        lap_like = (
            self.distances[-self.X_int.shape[0] :, : self.X_shared.shape[0]]
            - 4 * self.sigma**2
        ) / self.sigma**6
        diff = np.expand_dims(self.X_int[:, 1], 1) - np.expand_dims(
            self.X_shared[:, 1], 0
        )
        return -(
            self.exp_d[-self.X_int.shape[0] :, : self.X_shared.shape[0]]
            * lap_like
            * diff
        )

    def get_lap1lap2(self):
        dist = (
            (
                self.distances[-self.X_int.shape[0] :, -self.X_int.shape[0] :]
                - 4 * self.sigma**2
            )
            ** 2
            - 8 * self.sigma**4
        ) / self.sigma**8
        return self.exp_d[-self.X_int.shape[0] :, -self.X_int.shape[0] :] * dist


class MatVecStorage:
    def __init__(self, mat, vec, X_int, X_shared_dict, X_boundary) -> None:
        self.mat = mat
        self.vec = vec
        self.X_int = X_int
        self.X_shared_dict = X_shared_dict
        self.X_boundary = X_boundary
        self.indices = {}

    def select_from_mat(self, axis, begin, end):
        assert axis in ["x", "y", "xy"]
        if axis == "xy":
            return self.mat[begin:end, begin:end]
        if axis == "y":
            return self.mat[:, begin:end]
        if axis == "x":
            return self.mat[begin:end, :]

    def get_index(self, who, operator):
        try:
            begin, end = self.indices[(who, operator)]
        except KeyError:
            if who == "ext":
                begin = sum([x.shape[0] for x in self.X_shared_dict.values()])
                end = begin + self.X_boundary.shape[0]
            elif who == "int":
                begin = (
                    3 * sum([x.shape[0] for x in self.X_shared_dict.values()])
                    + self.X_boundary.shape[0]
                )
                end = begin + self.X_int.shape[0]
            else:
                if operator == "dirac":
                    begin = 0
                elif operator == "dx":
                    begin = (
                        sum([x.shape[0] for x in self.X_shared_dict.values()])
                        + self.X_boundary.shape[0]
                    )
                elif operator == "dy":
                    begin = (
                        2 * sum([x.shape[0] for x in self.X_shared_dict.values()])
                        + self.X_boundary.shape[0]
                    )
                else:
                    raise f"operator uknown: {operator}"

                for n, X in self.X_shared_dict.items():
                    if n != who:
                        begin += X.shape[0]
                    else:
                        break
                end = begin + self.X_shared_dict[who].shape[0]

            self.indices[(who, operator)] = (begin, end)
        return (begin, end)

    def get_mat(self, who, operator, axis):
        begin, end = self.get_index(who, operator)
        return self.select_from_mat(axis, begin, end)

    def get_vec(self, who, operator):
        begin, end = self.get_index(who, operator)
        return self.vec[begin:end]


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
