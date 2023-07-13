import numpy as np
from sklearn.metrics import pairwise_distances
from functools import partial
from tqdm import tqdm
from random import shuffle
from kernel import Differentiable_matern_kernel, Differentiable_gaussian_kernel


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
        try:
            self.all_shared = np.concatenate(list(self.X_shared.values()))
        except:
            self.all_shared = np.empty((0, 2))
        self.K_mat, B, D = PDESolver.get_kernel_matrix_fast_gaussian(
            self.X_int,
            self.all_shared,
            self.X_boundary,
            self.sigma,
            nugget,
            laplaceBool=False,
            shared_laplacian=True,
        )
        chol = np.linalg.cholesky(np.block([[self.K_mat, B.T], [B, D]]))

        if self.all_shared.shape[0] == 0:
            self.L = np.linalg.inv(chol)
            self.L_shared_lap = np.linalg.inv(chol)
        else:
            self.L_shared_lap = np.linalg.inv(chol)
            self.L = self.L_shared_lap[
                : -self.all_shared.shape[0], : -self.all_shared.shape[0]
            ]

        self.K_inv = self.L.T @ self.L
        """S = D - B @ self.K_inv @ B.T
        print(np.linalg.eigvalsh())
        print(np.linalg.eigvalsh(S))
        cholS = np.linalg.cholesky(S)
        self.L_shared_lap = np.block(
            [[chol, np.zeros_like(B.T)], [B @ (self.L.T), cholS]]
        )"""

        """self.L_shared_lap = np.linalg.inv(np.linalg.cholesky(self.K_mat_shared_lap))
        if self.all_shared.shape[0] == 0:
            self.K_mat = self.K_mat_shared_lap
            self.L = self.L_shared_lap
        else:
            self.K_mat = self.K_mat_shared_lap[
                : -self.all_shared.shape[0], : -self.all_shared.shape[0]
            ]
            self.L = np.linalg.inv(np.linalg.cholesky(self.K_mat))"""

        self.K_mat_laplace, _, _ = PDESolver.get_kernel_matrix_fast_gaussian(
            self.X_int,
            self.all_shared,
            self.X_boundary,
            self.sigma,
            nugget,
            laplaceBool=True,
            shared_laplacian=False,
        )
        self.K_inv_laplace = np.linalg.inv(self.K_mat_laplace)

        self.g_vec = np.array([g(x) for x in self.X_boundary])
        self.f_vec = np.array([f(x) for x in self.X_int])
        self.f_vec_shared = np.array([f(x) for x in self.all_shared])
        size_shared = self.all_shared.shape[0]
        vec = np.concatenate(
            [
                np.zeros(size_shared),
                np.zeros_like(self.g_vec),
                np.zeros_like(self.f_vec),
                np.zeros(2 * size_shared),
                np.zeros_like(self.f_vec),
            ]
        )
        self.storage = MatVecStorageLaplace(
            self.K_inv_laplace, vec, self.X_int, self.X_shared, self.X_boundary
        )

    def fit_interior(self, f, g, tau, dtau, use_shared, share_laplacian, nugget=1e-5):
        if not hasattr(self, "g_vec"):
            self.setup_fit(f, g, nugget)
        z_shared, dx_shared, dy_shared, L, shared_laplacian = self.get_shared_values(
            use_shared, share_laplacian, tau
        )
        self.gauss_newton_solution = PDESolver.gauss_newton(
            x_int=self.X_int,
            x_ext=self.X_boundary,
            z_shared=z_shared,
            dx_shared=dx_shared,
            dy_shared=dy_shared,
            L=L,
            f_vec=self.f_vec,
            g_vec=self.g_vec,
            tau=tau,
            dtau=dtau,
            shared_laplacian=shared_laplacian,
        )

    def finish_fit(self):
        self.a = np.concatenate(
            [self.shared_value[n]["dirac"] for n in self.neighbors]
            + [self.g_vec]
            + [self.gauss_newton_solution["z"]]
            + [self.shared_value[n]["dx"] for n in self.neighbors]
            + [self.shared_value[n]["dy"] for n in self.neighbors]
            + [self.gauss_newton_solution["z_lap"]]
        )
        self.coeff = self.K_inv @ self.a

    def get_shared_values(self, use_shared, share_laplacian, tau):
        if use_shared:
            z_shared = np.concatenate(
                [self.shared_value[n]["dirac"] for n in self.neighbors]
            )
            dx_shared = np.concatenate(
                [self.shared_value[n]["dx"] for n in self.neighbors]
            )
            dy_shared = np.concatenate(
                [self.shared_value[n]["dy"] for n in self.neighbors]
            )
            lap_shared = self.f_vec_shared - tau(z_shared)
            if share_laplacian:
                return (
                    z_shared,
                    dx_shared,
                    dy_shared,
                    self.L_shared_lap,
                    lap_shared,
                )
            return (
                z_shared,
                dx_shared,
                dy_shared,
                self.L,
                np.empty((0)),
            )

        K, _, _ = PDESolver.get_kernel_matrix_fast_gaussian(
            self.X_int,
            np.empty((0, 2)),
            self.X_boundary,
            self.sigma,
            self.nugget,
            laplaceBool=False,
            shared_laplacian=False,
        )
        return (
            np.empty((0)),
            np.empty((0)),
            np.empty((0)),
            np.linalg.inv(np.linalg.cholesky(K)),
            np.empty((0)),
        )

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

    def setup_joint_fit(models):
        pairs = {}
        operators = ["dirac", "dx", "dy"]
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
        total_size = begin
        for model in models:
            matrix = np.zeros((total_size, total_size))
            for operator1 in operators:
                for operator2 in operators:
                    for m1 in model.neighbors:
                        for m2 in model.neighbors:
                            begin1, end1 = model.storage.get_index(m1, operator1)
                            begin2, end2 = model.storage.get_index(m2, operator2)
                            try:
                                indices1 = pairs[(operator1, model, m1)]
                            except:
                                indices1 = pairs[(operator1, m1, model)]
                            try:
                                indices2 = pairs[(operator2, model, m2)]
                            except:
                                indices2 = pairs[(operator2, m2, model)]
                            matrix[
                                indices1[0] : indices1[1], indices2[0] : indices2[1]
                            ] = model.K_inv_laplace[begin1:end1, begin2:end2]
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
            for other_model in model.neighbors:
                for operator in ["dirac", "dx", "dy"]:
                    begin, end = model.storage.get_index(other_model, operator)
                    begin_ext, end_ext = model.storage.get_index("ext", operator)
                    begin_int, end_int = model.storage.get_index("int", operator)
                    try:
                        indices1 = pairs[(operator, model, other_model)]
                    except:
                        indices1 = pairs[(operator, other_model, model)]
                    target[indices1[0] : indices1[1]] += (
                        model.K_inv_laplace[begin:end, begin_ext:end_ext] @ model.g_vec
                    )
                    target[indices1[0] : indices1[1]] += (
                        model.K_inv_laplace[begin:end, begin_int:end_int]
                        @ model.gauss_newton_solution["z_lap"]
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
                model.shared_value[n] = {}
                for operator in ["dirac", "dx", "dy"]:
                    try:
                        indices = pairs[(operator, model, n)]
                    except:
                        indices = pairs[(operator, n, model)]
                    model.shared_value[n][operator] = shared_value[
                        indices[0] : indices[1]
                    ]

            model.finish_fit()

    def joint_fit(
        models, f, g, tau, dtau, nugget=1e-5, tol=1e-6, max_eval=100, show_progress=True
    ):
        dz = {}
        for model in models:
            if not hasattr(model, "g_vec"):
                model.setup_fit(f, g, nugget)
            (
                z_shared,
                dx_shared,
                dy_shared,
                L,
                shared_laplacian,
            ) = model.get_shared_values(False, False, tau)
            model.gauss_newton_solution, dz[model] = PDESolver.gauss_newton_step(
                x_int=model.X_int,
                x_ext=model.X_boundary,
                z=np.zeros(model.X_int.shape[0]),
                z_shared=z_shared,
                dx_shared=dx_shared,
                dy_shared=dy_shared,
                L=L,
                f_vec=model.f_vec,
                g_vec=model.g_vec,
                tau=tau,
                dtau=dtau,
                shared_laplacian=shared_laplacian,
            )
        PDESolver.joint_fit_boundaries(models)
        if show_progress:
            progress = tqdm()
        dz_norm = 10
        eval = 0
        while dz_norm > tol and eval < max_eval:
            indexes = [k for k in range(len(models))]
            shuffle(indexes)
            for model in [models[k] for k in indexes]:
                (
                    z_shared,
                    dx_shared,
                    dy_shared,
                    L,
                    shared_laplacian,
                ) = model.get_shared_values(
                    True, False, tau
                )  # (True, (dz_norm < 100 * tol), tau)
                model.gauss_newton_solution, dz[model] = PDESolver.gauss_newton_step(
                    x_int=model.X_int,
                    x_ext=model.X_boundary,
                    z=model.gauss_newton_solution["z"],
                    z_shared=z_shared,
                    dx_shared=dx_shared,
                    dy_shared=dy_shared,
                    L=L,
                    f_vec=model.f_vec,
                    g_vec=model.g_vec,
                    tau=tau,
                    dtau=dtau,
                    shared_laplacian=shared_laplacian,
                )

            dz_norm = np.max([np.linalg.norm(dz[model], np.inf) for m in models])
            if show_progress:
                progress.set_description(f"Current residual {dz_norm:.3e}")
                progress.update()
            PDESolver.joint_fit_boundaries(models)
            eval += 1
        if show_progress:
            progress.close()
        """indexes = [k for k in range(len(models))]
        shuffle(indexes)
        for model in [models[k] for k in indexes]:
            (
                z_shared,
                dx_shared,
                dy_shared,
                L,
                shared_laplacian,
            ) = model.get_shared_values(True, True, tau)
            model.gauss_newton_solution, dz[model] = PDESolver.gauss_newton_step(
                x_int=model.X_int,
                x_ext=model.X_boundary,
                z=model.gauss_newton_solution["z"],
                z_shared=z_shared,
                dx_shared=dx_shared,
                dy_shared=dy_shared,
                L=L,
                f_vec=model.f_vec,
                g_vec=model.g_vec,
                tau=tau,
                dtau=dtau,
                shared_laplacian=shared_laplacian,
            )
        PDESolver.joint_fit_boundaries(models)"""

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
            self.coeff,
            PDESolver.get_kernel_vector(
                self.X_int,
                self.all_shared,
                self.X_boundary,
                self.sigma,
                x,
                laplaceBool=False,
            ),
        )

    def laplacian(self, x):
        return np.dot(
            self.coeff,
            PDESolver.get_laplacian_kernel_vector(
                self.X_int,
                self.all_shared,
                self.X_boundary,
                self.sigma,
                x,
                laplaceBool=False,
            ),
        )

    def __repr__(self) -> str:
        try:
            return self.name
        except:
            return "PDE Solver with K_mat : " + self.K_mat.__repr__()

    def differential_matrix(z, N_int, N_shared, N_ext, dtau, share_laplacian):
        return np.block(
            [
                [np.zeros((N_ext + N_shared, N_int))],
                [-np.eye(N_int)],
                [np.zeros((2 * N_shared, N_int))],
                [np.diag(dtau(z))],
                [np.zeros((int(share_laplacian) * N_shared, N_int))],
            ]
        )

    def gauss_newton(
        x_int,
        x_ext,
        z_shared,
        dx_shared,
        dy_shared,
        L,
        f_vec,
        g_vec,
        tau,
        dtau,
        shared_laplacian,
    ):
        z = np.zeros(x_int.shape[0])
        dz = 3 * np.ones_like(z)
        res = {}
        progress = tqdm()
        while np.linalg.norm(dz, np.inf) > 1e-6:
            res, dz = PDESolver.gauss_newton_step(
                x_int,
                x_ext,
                z,
                z_shared,
                dx_shared,
                dy_shared,
                L,
                f_vec,
                g_vec,
                tau,
                dtau,
                shared_laplacian,
            )
            progress.set_description(
                f"Current residual {np.linalg.norm(dz,np.inf):.3e}"
            )
            progress.update()
        progress.close()
        return res

    def gauss_newton_step(
        x_int,
        x_ext,
        z,
        z_shared,
        dx_shared,
        dy_shared,
        L,
        f_vec,
        g_vec,
        tau,
        dtau,
        shared_laplacian,
    ):
        H = PDESolver.differential_matrix(
            z,
            x_int.shape[0],
            z_shared.shape[0],
            x_ext.shape[0],
            dtau,
            share_laplacian=not (shared_laplacian.shape[0] == 0),
        )
        target = L @ np.concatenate(
            [z_shared, g_vec, z, dx_shared, dy_shared, f_vec - tau(z), shared_laplacian]
        )
        mat = L @ H
        dz = np.linalg.lstsq(mat, target, rcond=None)[0]
        z += dz
        return {
            "z": z,
            "z_lap": f_vec - tau(z),
            "alpha": np.concatenate(
                [z_shared, g_vec, z, dx_shared, dy_shared, f_vec - tau(z)]
            ),
        }, dz

    def get_kernel_matrix_matern(
        X_int, X_shared, X_ext, l, nu, nugget, laplaceBool, shared_laplacian
    ):
        k = Differentiable_matern_kernel(nu, l)
        if laplaceBool:
            X_dirac = np.concatenate([X_shared, X_ext])
        else:
            X_dirac = np.concatenate([X_shared, X_ext, X_int])

        dirac_mat = k.apply("kappa", X_dirac, X_dirac)
        dx1 = k.apply("D_x1_kappa", X_shared, X_dirac)
        dy1 = k.apply("D_x2_kappa", X_shared, X_dirac)
        lap1 = k.apply("Delta_x_kappa", X_int, X_dirac)
        ddx1 = k.apply("D_x1_D_y1_kappa", X_shared, X_shared)
        dydx1 = k.apply("D_x2_D_y1_kappa", X_shared, X_shared)
        dxlap = k.apply("Delta_x_D_y1_kappa", X_int, X_shared)
        ddy = k.apply("D_x2_D_y2_kappa", X_shared, X_shared)
        dylap = k.apply("Delta_x_D_y2_kappa", X_int, X_shared)
        laplap = k.apply("Delta_x_Delta_y_kappa", X_int, X_int)

        res = [
            [dirac_mat, dx1.T, dy1.T, lap1.T],
            [dx1, ddx1, dydx1.T, dxlap.T],
            [dy1, dydx1, ddy, dylap.T],
            [lap1, dxlap, dylap, laplap],
        ]
        res = np.block(res)

        if not shared_laplacian:
            return res + np.eye(res.shape[0]) * nugget, None, None

        lap_shared = k.apply("Delta_x_kappa", X_shared, X_dirac)
        dxlap_shared = k.apply("Delta_x_D_y1_kappa", X_shared, X_shared)
        dylap_shared = k.apply("Delta_x_D_y2_kappa", X_shared, X_shared)
        laplap_shared_shared = k.apply("Delta_x_Delta_y_kappa", X_shared, X_shared)
        laplap_int_shared = k.apply("Delta_x_Delta_y_kappa", X_shared, X_int)

        V = np.concatenate(
            [lap_shared, dxlap_shared, dylap_shared, laplap_int_shared], axis=1
        )
        # res_shared = np.block([[res, V.T], [V, laplap_shared_shared]])
        return (
            res + np.eye(res.shape[0]) * nugget,
            V,
            laplap_shared_shared + np.eye(laplap_shared_shared.shape[0]) * nugget,
        )

    def get_kernel_matrix_gaussian(
        X_int, X_shared, X_ext, sigma, nugget, laplaceBool, shared_laplacian
    ):
        k = Differentiable_gaussian_kernel(sigma)
        if laplaceBool:
            X_dirac = np.concatenate([X_shared, X_ext])
        else:
            X_dirac = np.concatenate([X_shared, X_ext, X_int])

        dirac_mat = k.apply("kappa", X_dirac, X_dirac)
        dx1 = k.apply("D_x1_kappa", X_shared, X_dirac)
        dy1 = k.apply("D_x2_kappa", X_shared, X_dirac)
        lap1 = -k.apply("Delta_x_kappa", X_int, X_dirac)
        ddx1 = k.apply("D_x1_D_y1_kappa", X_shared, X_shared)
        dydx1 = k.apply("D_x2_D_y1_kappa", X_shared, X_shared)
        dxlap = -k.apply("Delta_x_D_y1_kappa", X_int, X_shared)
        ddy = k.apply("D_x2_D_y2_kappa", X_shared, X_shared)
        dylap = -k.apply("Delta_x_D_y2_kappa", X_int, X_shared)
        laplap = k.apply("Delta_x_Delta_y_kappa", X_int, X_int)

        res = [
            [dirac_mat, dx1.T, dy1.T, lap1.T],
            [dx1, ddx1, dydx1.T, dxlap.T],
            [dy1, dydx1, ddy, dylap.T],
            [lap1, dxlap, dylap, laplap],
        ]
        res = np.block(res)

        if not shared_laplacian:
            return res + np.eye(res.shape[0]) * nugget, None, None

        lap_shared = -k.apply("Delta_x_kappa", X_shared, X_dirac)
        dxlap_shared = -k.apply("Delta_x_D_y1_kappa", X_shared, X_shared)
        dylap_shared = -k.apply("Delta_x_D_y2_kappa", X_shared, X_shared)
        laplap_shared_shared = k.apply("Delta_x_Delta_y_kappa", X_shared, X_shared)
        laplap_int_shared = k.apply("Delta_x_Delta_y_kappa", X_shared, X_int)

        V = np.concatenate(
            [lap_shared, dxlap_shared, dylap_shared, laplap_int_shared], axis=1
        )
        # res_shared = np.block([[res, V.T], [V, laplap_shared_shared]])
        return (
            res + np.eye(res.shape[0]) * nugget,
            V,
            laplap_shared_shared + np.eye(laplap_shared_shared.shape[0]) * nugget,
        )

    def get_kernel_matrix_fast_gaussian(
        X_int, X_shared, X_ext, sigma, nugget, laplaceBool, shared_laplacian
    ):
        k2 = FasterGaussianKernel(sigma, X_int, X_shared, X_ext, laplaceBool)
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

        if not shared_laplacian:
            return res + np.eye(res.shape[0]) * nugget, None, None

        lap_shared = k2.get_lap_of_shared()
        dxlap_shared = k2.get_lap1dx2_of_shared()
        dylap_shared = k2.get_lap1dy2_of_shared()
        laplap_shared_shared = k2.get_lap1lap2_of_shared_shared()
        laplap_int_shared = k2.get_lap1lap2_of_int_shared()

        V = np.concatenate(
            [lap_shared.T, dxlap_shared, dylap_shared, laplap_int_shared.T], axis=1
        )
        # res_shared = np.block([[res, V.T], [V, laplap_shared_shared]])
        return (
            res + np.eye(res.shape[0]) * nugget,
            V,
            laplap_shared_shared + np.eye(laplap_shared_shared.shape[0]) * nugget,
        )

    def get_kernel_vector_matern(X_int, X_shared, X_ext, l, nu, x, laplaceBool):
        k = Differentiable_matern_kernel(nu, l)
        if laplaceBool:
            X_dirac = np.concatenate([X_shared, X_ext])
        else:
            X_dirac = np.concatenate([X_shared, X_ext, X_int])
        dirac_mat = k.apply("kappa", X_dirac, x)
        dx1 = k.apply("D_x1_kappa", X_shared, x)
        dy1 = k.apply("D_x2_kappa", X_shared, x)
        lap1 = -k.apply("Delta_x_kappa", X_int, x)
        return np.concatenate([dirac_mat, dx1, dy1, lap1])

    def get_laplacian_kernel_vector_matern(
        X_int, X_shared, X_ext, l, nu, x, laplaceBool
    ):
        k = Differentiable_matern_kernel(nu, l)
        if laplaceBool:
            X_dirac = np.concatenate([X_shared, X_ext])
        else:
            X_dirac = np.concatenate([X_shared, X_ext, X_int])
        lap = -k.apply("Delta_x_kappa", x, X_dirac).T
        dxlap = -k.apply("Delta_x_D_y1_kappa", x, X_shared).T
        dylap = -k.apply("Delta_x_D_y2_kappa", x, X_shared).T
        laplap = k.apply("Delta_x_Delta_y_kappa", x, X_int).T
        return np.concatenate([lap, dxlap, dylap, laplap])

    def get_kernel_vector(X_int, X_shared, X_ext, sigma, x, laplaceBool):
        k2 = GaussianKernelVectorDirac(sigma, X_int, X_shared, X_ext, x, laplaceBool)
        dirac_mat = k2.get_dirac()
        dx1 = k2.get_dx()
        dy1 = k2.get_dy()
        lap1 = k2.get_lap()
        return np.concatenate([dirac_mat, dx1, dy1, lap1])

    def get_laplacian_kernel_vector(X_int, X_shared, X_ext, sigma, x, laplaceBool):
        k2 = GaussianKernelVectorLap(sigma, X_int, X_shared, X_ext, x, laplaceBool)
        lap = k2.get_lap()
        dxlap = k2.get_lap1dx2()
        dylap = k2.get_lap1dy2()
        laplap = k2.get_lap1lap2()
        return np.concatenate([lap, dxlap, dylap, laplap])


class GaussianKernelVectorDirac:
    def __init__(self, s, X_int, X_shared, X_boundary, x, laplaceBool):
        self.sigma = s
        self.x = x
        self.X_int = X_int
        self.X_shared = X_shared
        self.X_boundary = X_boundary
        if laplaceBool:
            self.X_dirac = np.concatenate([X_shared, X_boundary])
        else:
            self.X_dirac = np.concatenate([X_shared, X_boundary, X_int])
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
    def __init__(self, s, X_int, X_shared, X_boundary, x, laplaceBool):
        self.sigma = s
        self.x = x
        self.X_int = X_int
        self.X_shared = X_shared
        self.X_boundary = X_boundary
        if laplaceBool:
            self.X_dirac = np.concatenate([X_shared, X_boundary])
        else:
            self.X_dirac = np.concatenate([X_shared, X_boundary, X_int])
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
    def __init__(self, s, X_int, X_shared, X_boundary, laplaceBool):
        self.sigma = s
        self.X_int = X_int
        self.X_shared = X_shared
        self.X_boundary = X_boundary
        if laplaceBool:
            self.X_dirac = np.concatenate([X_shared, X_boundary])
        else:
            self.X_dirac = np.concatenate([X_shared, X_boundary, X_int])
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

    def get_lap_of_shared(self):
        to_mult = (
            self.distances[: self.X_dirac.shape[0], : self.X_shared.shape[0]]
            - 2 * self.sigma**2
        ) / self.sigma**4
        return -self.exp_d[: self.X_dirac.shape[0], : self.X_shared.shape[0]] * to_mult

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

    def get_lap1dx2_of_shared(self):
        lap_like = (
            self.distances[: self.X_shared.shape[0], : self.X_shared.shape[0]]
            - 4 * self.sigma**2
        ) / self.sigma**6
        diff = np.expand_dims(self.X_shared[:, 0], 1) - np.expand_dims(
            self.X_shared[:, 0], 0
        )
        return -(
            self.exp_d[: self.X_shared.shape[0], : self.X_shared.shape[0]]
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

    def get_lap1dy2_of_shared(self):
        lap_like = (
            self.distances[: self.X_shared.shape[0], : self.X_shared.shape[0]]
            - 4 * self.sigma**2
        ) / self.sigma**6
        diff = np.expand_dims(self.X_shared[:, 1], 1) - np.expand_dims(
            self.X_shared[:, 1], 0
        )
        return -(
            self.exp_d[: self.X_shared.shape[0], : self.X_shared.shape[0]]
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

    def get_lap1lap2_of_int_shared(self):
        dist = (
            (
                self.distances[-self.X_int.shape[0] :, : self.X_shared.shape[0]]
                - 4 * self.sigma**2
            )
            ** 2
            - 8 * self.sigma**4
        ) / self.sigma**8
        return self.exp_d[-self.X_int.shape[0] :, : self.X_shared.shape[0]] * dist

    def get_lap1lap2_of_shared_shared(self):
        dist = (
            (
                self.distances[: self.X_shared.shape[0], : self.X_shared.shape[0]]
                - 4 * self.sigma**2
            )
            ** 2
            - 8 * self.sigma**4
        ) / self.sigma**8
        return self.exp_d[: self.X_shared.shape[0], : self.X_shared.shape[0]] * dist


class MatVecStorageLaplace:
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
