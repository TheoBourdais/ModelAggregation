import numpy as np
from scipy.fft import dstn, idstn
import scipy.sparse as sp
import scipy.linalg
from scipy.interpolate import RegularGridInterpolator
from sklearn.metrics import pairwise_distances


def laplace_spectral(N, N_target):
    """Solve the Poisson equation
    \Delta u = f
    on the unit square with Dirichlet boundary conditions
    u = 0
    using the discrete sine transform.
    """
    # Create the grid
    x = np.linspace(0, 1, N + 1, endpoint=True)
    y = np.linspace(0, 1, N + 1, endpoint=True)
    X, Y = np.meshgrid(x[1:-1], y[1:-1])

    # Create the wave numbers
    kx = np.pi * np.arange(1, N)
    ky = np.pi * np.arange(1, N)
    KX, KY = np.meshgrid(kx, ky)
    denominator = KX**2 + KY**2
    x_target = np.linspace(0, 1, N_target + 1, endpoint=True)
    y_target = np.linspace(0, 1, N_target + 1, endpoint=True)
    X_target, Y_target = np.meshgrid(x_target, y_target)

    def solver(f):
        F = f(X, Y)
        Fhat = dstn(F, type=1)
        Uhat = Fhat / denominator
        U = idstn(-Uhat, type=1)
        res = np.pad(U, ((1, 1), (1, 1)), "constant")

        return RegularGridInterpolator((x, y), res.T)((X_target, Y_target))

    return solver


def laplace_fdm(N, N_target):
    L = 1
    h = L / (N)

    A = make_mat(N - 1, h, N, h)
    x = np.linspace(0, L, N + 1, endpoint=True)
    y = np.linspace(0, L, N + 1, endpoint=True)
    X, Y = np.meshgrid(x[1:-1], y[1:-1])

    factorized_A = sp.linalg.factorized(A)

    x_target = np.linspace(0, L, N_target + 1, endpoint=True)
    y_target = np.linspace(0, L, N_target + 1, endpoint=True)
    X_target, Y_target = np.meshgrid(x_target, y_target)

    def solver(f):
        F = f(X, Y)
        F_flat = F.flatten()

        U_flat = factorized_A(F_flat)
        U = np.zeros((N + 1, N + 1))
        U[1:N, 1:N] = -U_flat.reshape((N - 1, N - 1))
        interp = RegularGridInterpolator((x, y), U.T)

        return interp((X_target, Y_target))

    return solver


def send_to_boundary(X):
    index = np.random.randint(0, X.shape[1], size=X.shape[0])
    X[np.arange(X.shape[0]), index] = (X[np.arange(X.shape[0]), index] > 0.5).astype(
        np.float64
    )
    return X


class GaussianKernel:
    def __init__(self, s, X_int, X_boundary):
        self.sigma = s
        self.X_int = X_int
        self.X_boundary = X_boundary
        self.size_dirac = X_boundary.shape[0]
        self.all = np.concatenate([X_boundary, X_int])
        self.distances = pairwise_distances(self.all) ** 2
        self.exp_d = np.exp(-self.distances / 2 / s**2)

    def get_dirac(self):
        return self.exp_d[: self.size_dirac, : self.size_dirac]

    def get_lap(self):
        to_mult = (
            self.distances[-self.X_int.shape[0] :, : self.size_dirac]
            - 2 * self.sigma**2
        ) / self.sigma**4
        return -self.exp_d[-self.X_int.shape[0] :, : self.size_dirac] * to_mult

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

    def K(self):
        KXX = self.get_dirac()
        lapKXX = self.get_lap()
        lap1lap2KXX = self.get_lap1lap2()
        return np.block([[KXX, lapKXX.T], [lapKXX, lap1lap2KXX]])

    def Kx(self, x):
        dist = pairwise_distances(self.all, x) ** 2
        dirac = np.exp(-dist / 2 / self.sigma**2)[: self.size_dirac]
        to_mult_lap = (dist[-self.X_int.shape[0] :] - 2 * self.sigma**2) / self.sigma**4
        lap = -np.exp(-dist[-self.X_int.shape[0] :] / 2 / self.sigma**2) * to_mult_lap
        return np.concatenate([dirac, lap], axis=0)


def laplace_GP(N, N_target, sigma=0.1, nugget=1e-7):
    L = 1
    X_int = np.random.rand(int(N * 0.85), 2)
    X_bnd = np.random.rand(int(N * 0.15), 2)
    X_bnd = send_to_boundary(X_bnd)

    kernel = GaussianKernel(sigma, X_int, X_bnd)
    K = kernel.K() + nugget * np.eye(N)
    cho_factor = scipy.linalg.cho_factor(K, lower=True)

    x = np.linspace(0, L, N_target + 1, endpoint=True)  # exclude boundary points
    y = np.linspace(0, L, N_target + 1, endpoint=True)
    X, Y = np.meshgrid(x, y)
    Kx = kernel.Kx(np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1))
    V = scipy.linalg.cho_solve(cho_factor, Kx)[int(N * 0.15) :]

    def solver(f):
        F = f(X_int[:, 0], X_int[:, 1])
        U = V.T @ F
        return -U.reshape((N_target + 1, N_target + 1))

    return solver


def make_mat(Nx, hx, Ny, hy):
    # Calculate total number of grid points

    # Create grid

    N2 = (Nx) * (Ny - 1)
    # Construct the finite difference matrix for the Laplacian
    A = sp.diags(
        [
            -1 / hy**2,
            -1 / hx**2,
            2 * (1 / hx**2 + 1 / hy**2),
            -1 / hx**2,
            -1 / hy**2,
        ],
        [-Nx, -1, 0, 1, Nx],
        shape=(N2, N2),
    )
    coo = A.tocoo()

    row1 = (Nx) * np.arange(1, Ny - 1) - 1
    col1 = row1 + 1
    row2 = (Nx) * np.arange(1, Ny - 1)
    col2 = row2 - 1
    data = np.concatenate([coo.data, np.ones(2 * (Ny - 2)) / hx**2])
    cols = np.concatenate([coo.col, col1, col2], dtype=np.int32)
    rows = np.concatenate([coo.row, row1, row2], dtype=np.int32)
    return sp.csc_matrix((data, (rows, cols)), shape=(N2, N2))


def interpolate_two_blocks(L1, L, block1, block2, N):
    N1x = block1.shape[1] - 1
    N1y = block1.shape[0] - 1
    N2x = block2.shape[1] - 1
    N2y = block2.shape[0] - 1
    x1 = np.linspace(0, L1, N1x + 1, endpoint=True)
    y1 = np.linspace(0, L, N1y + 1, endpoint=True)
    x2 = np.linspace(L1, L, N2x + 1, endpoint=True)
    y2 = np.linspace(0, L, N2y + 1, endpoint=True)
    left = RegularGridInterpolator((x1, y1), block1.T)
    right = RegularGridInterpolator((x2, y2), block2.T)
    x = np.linspace(0, L, N + 1, endpoint=True)
    y = np.linspace(0, L, N + 1, endpoint=True)
    if N1x > N2x:
        X1, Y1 = np.meshgrid(x[x <= L1], y)
        X2, Y2 = np.meshgrid(x[x > L1], y)
        res_left = left((X1, Y1))
        res_right = right((X2, Y2))
        return np.concatenate([res_left, res_right], axis=1).reshape((N + 1, N + 1))
    else:
        X1, Y1 = np.meshgrid(x[x < L1], y)
        X2, Y2 = np.meshgrid(x[x >= L1], y)
        res_left = left((X1, Y1))
        res_right = right((X2, Y2))
        return np.concatenate([res_left, res_right], axis=1).reshape((N + 1, N + 1))


def make_left_interpolation(L1, N1, N1y, N2, N2y, h1, h1y, h2, h2y):
    j_left = np.floor((L1 - h2 - h1) / h1).astype(np.int32)
    i_left = np.floor((h2y * np.arange(N2y - 1)) / h1y).astype(np.int32)

    def index_to_pos(i, j):
        return (j + 1) * h1, i * h1y

    def interpolation(x1, y1, x2, y2, x, y):
        assert (
            np.all(x1 <= x2)
            and np.all(y1 <= y2)
            and np.all(x >= x1)
            and np.all(x <= x2)
            and np.all(y >= y1)
            and np.all(y <= y2)
        )
        return np.stack(
            [
                (x2 - x) * (y2 - y),
                (x - x1) * (y2 - y),
                (x2 - x) * (y - y1),
                (x - x1) * (y - y1),
            ]
        ) / (h1 * h1y)

    indexes_top_left = j_left + i_left * (N1)
    valid_top_left = indexes_top_left < N1 * (N1y - 1)
    positions_top_left = index_to_pos(i_left, j_left * np.ones_like(i_left))

    indexes_top_right = indexes_top_left + 1
    valid_top_right = indexes_top_right < N1 * (N1y - 1)

    indexes_bottom_left = j_left + (i_left + 1) * (N1)
    valid_bottom_left = indexes_bottom_left < N1 * (N1y - 1)

    indexes_bottom_right = indexes_bottom_left + 1
    valid_bottom_right = indexes_bottom_right < N1 * (N1y - 1)
    positions_bottom_right = index_to_pos(
        i_left + 1, (j_left + 1) * np.ones_like(i_left)
    )

    xy = ((L1 - h2) * np.ones_like(i_left), h2y * np.arange(N2y - 1))

    (
        data_top_left,
        data_top_right,
        data_bottom_left,
        data_bottom_right,
    ) = interpolation(
        positions_top_left[0],
        positions_top_left[1],
        positions_bottom_right[0],
        positions_bottom_right[1],
        xy[0],
        xy[1],
    )

    data_top_left = data_top_left[valid_top_left]
    indexes_top_left = indexes_top_left[valid_top_left]

    data_top_right = data_top_right[valid_top_right]
    indexes_top_right = indexes_top_right[valid_top_right]

    data_bottom_left = data_bottom_left[valid_bottom_left]
    indexes_bottom_left = indexes_bottom_left[valid_bottom_left]

    data_bottom_right = data_bottom_right[valid_bottom_right]
    indexes_bottom_right = indexes_bottom_right[valid_bottom_right]

    data = np.concatenate(
        [data_top_left, data_top_right, data_bottom_left, data_bottom_right]
    )
    data = np.abs(data)
    cols = np.concatenate(
        [indexes_top_left, indexes_top_right, indexes_bottom_left, indexes_bottom_right]
    )
    rows = N1 * (N1y - 1) + np.concatenate(
        [
            (N2) * np.arange(N2y - 1)[valid]
            for valid in [
                valid_top_left,
                valid_top_right,
                valid_bottom_left,
                valid_bottom_right,
            ]
        ]
    )
    return -data / h2**2, rows, cols


def make_right_interpolation(L1, N1, N1y, N2, N2y, h1, h1y, h2, h2y):
    j_left = np.floor((h1) / h2).astype(np.int32)
    i_left = np.floor((h1y * np.arange(N1y - 1)) / h2y).astype(np.int32)

    def index_to_pos(i, j):
        return L1 + j * h2, i * h2y

    def interpolation(x1, y1, x2, y2, x, y):
        assert (
            np.all(x1 <= x2)
            and np.all(y1 <= y2)
            and np.all(x >= x1)
            and np.all(x <= x2)
            and np.all(y >= y1)
            and np.all(y <= y2)
        )
        return np.stack(
            [
                (x2 - x) * (y2 - y),
                (x - x1) * (y2 - y),
                (x2 - x) * (y - y1),
                (x - x1) * (y - y1),
            ]
        ) / (h2 * h2y)

    indexes_top_left = j_left + i_left * (N2)
    valid_top_left = indexes_top_left < N2 * (N2y - 1)
    positions_top_left = index_to_pos(i_left, j_left * np.ones_like(i_left))

    indexes_top_right = indexes_top_left + 1
    valid_top_right = indexes_top_right < N2 * (N2y - 1)

    indexes_bottom_left = j_left + (i_left + 1) * (N2)
    valid_bottom_left = indexes_bottom_left < N2 * (N2y - 1)

    indexes_bottom_right = indexes_bottom_left + 1
    valid_bottom_right = indexes_bottom_right < N2 * (N2y - 1)
    positions_bottom_right = index_to_pos(
        i_left + 1, (j_left + 1) * np.ones_like(i_left)
    )

    xy = ((L1 + h1) * np.ones_like(i_left), h1y * np.arange(N1y - 1))

    (
        data_top_left,
        data_top_right,
        data_bottom_left,
        data_bottom_right,
    ) = interpolation(
        positions_top_left[0],
        positions_top_left[1],
        positions_bottom_right[0],
        positions_bottom_right[1],
        xy[0],
        xy[1],
    )

    data_top_left = data_top_left[valid_top_left]
    indexes_top_left = indexes_top_left[valid_top_left]

    data_top_right = data_top_right[valid_top_right]
    indexes_top_right = indexes_top_right[valid_top_right]

    data_bottom_left = data_bottom_left[valid_bottom_left]
    indexes_bottom_left = indexes_bottom_left[valid_bottom_left]

    data_bottom_right = data_bottom_right[valid_bottom_right]
    indexes_bottom_right = indexes_bottom_right[valid_bottom_right]

    data = np.concatenate(
        [data_top_left, data_top_right, data_bottom_left, data_bottom_right]
    )
    data = np.abs(data)
    cols = N1 * (N1y - 1) + np.concatenate(
        [indexes_top_left, indexes_top_right, indexes_bottom_left, indexes_bottom_right]
    )
    rows = np.concatenate(
        [
            (N1) * np.arange(1, N1y)[valid] - 1
            for valid in [
                valid_top_left,
                valid_top_right,
                valid_bottom_left,
                valid_bottom_right,
            ]
        ]
    )
    return -data / h1**2, rows, cols


def laplace_fdm_inhomogeneus(N1, N2, L1, L2, N_target):
    L = 1
    assert L1 + L2 == L
    h1 = L1 / (N1)
    N1y = int(L / h1)
    h1y = L / (N1y)
    h2 = L2 / (N2)
    N2y = int(L / h2)
    h2y = L / (N2y)

    A1 = make_mat(Nx=N1, hx=h1, Ny=N1y, hy=h1y)

    A2 = make_mat(Nx=N2, hx=h2, Ny=N2y, hy=h2y)

    A = sp.block_diag([A1, A2])
    data_left, rows_left, cols_left = make_left_interpolation(
        L1, N1, N1y, N2, N2y, h1, h1y, h2, h2y
    )

    data_right, rows_right, cols_right = make_right_interpolation(
        L1, N1, N1y, N2, N2y, h1, h1y, h2, h2y
    )
    A = sp.coo_matrix(A)
    A.data = np.concatenate([A.data, data_left, data_right])
    A.row = np.concatenate([A.row, rows_left, rows_right])
    A.col = np.concatenate([A.col, cols_left, cols_right])
    A = A.tocsc()

    factorized_A = sp.linalg.factorized(A)

    x1 = np.linspace(0, L1, N1 + 1, endpoint=True)[1:]
    y1 = np.linspace(0, L, N1y + 1, endpoint=True)[1:-1]
    X1, Y1 = np.meshgrid(x1, y1)

    x2 = np.linspace(L1, L, N2 + 1, endpoint=True)[:-1]
    y2 = np.linspace(0, L, N2y + 1, endpoint=True)[1:-1]
    X2, Y2 = np.meshgrid(x2, y2)

    def solver(f):

        F1 = f(X1, Y1).flatten()
        F2 = f(X2, Y2).flatten()

        F_flat = np.concatenate([F1, F2])
        U_flat = -factorized_A(F_flat)
        Uleft = U_flat[: F1.shape[0]].reshape((N1y - 1, N1))
        Uright = U_flat[F1.shape[0] :].reshape((N2y - 1, N2))

        Uleft, Uright = np.pad(Uleft, ((1, 1), (1, 0))), np.pad(
            Uright, ((1, 1), (0, 1))
        )
        return interpolate_two_blocks(L1, L, Uleft, Uright, N_target)

    return solver
