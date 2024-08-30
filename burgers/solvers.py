import numpy as np
import scipy.linalg
from tqdm import tqdm


class GridArray:
    def __init__(self, X_span, T_span, data) -> None:
        assert tuple(data.shape[-2:]) == (
            len(T_span),
            len(X_span),
        ), f"Data shape {data.shape} does not match span shapes {len(T_span), len(X_span)}"
        self.X_span = X_span
        self.T_span = T_span
        self.data = data

    def locate_idx(span, values):
        idx = np.searchsorted(span, values)
        try:
            assert np.all(idx < span.shape[0]) and np.allclose(
                span[idx], values
            ), f"Values {values} are not in the span"
        except:
            # Handle case where numerical error causes values to be slightly off
            idx[span[idx] != values] -= 1
            assert np.all(idx < span.shape[0]) and np.allclose(
                span[idx], values
            ), f"Values {values} are not in the span"
        return idx

    def __getitem__(self, xt):
        x, t = xt
        x_idx = GridArray.locate_idx(self.X_span, x)
        t_idx = GridArray.locate_idx(self.T_span, t)
        try:
            return self.data[..., t_idx, x_idx]
        except:
            t_idx, x_idx = np.meshgrid(t_idx, x_idx, indexing="ij")
            final_shape = (*self.data.shape[:-2], *t_idx.shape)
            t_idx = t_idx.flatten()
            x_idx = x_idx.flatten()
            return self.data[..., t_idx, x_idx].reshape(final_shape)

    # implement shape property
    @property
    def shape(self):
        return self.data.shape

    # make iterable by iterating over first axis
    def __iter__(self):
        return iter(self.data)


class BurgersSolver:
    def __init__(self, nx, T, boundary="periodic") -> None:
        self.nx = nx
        self.T = T
        self.boundary = boundary
        self.dx = 1 / (nx - 1)

    def get_dt_nt(self, u0, nu, clf=0.85, diff=0.35):
        u_max = np.max(np.abs(u0))
        dt_cfl = clf * self.dx / u_max
        dt_diff = diff * self.dx**2 / nu
        print(dt_cfl, dt_diff)
        dt = min(dt_cfl, dt_diff)
        nt = int(self.T / dt) + 1
        dt = self.T / (nt - 1)
        return dt, nt

    def is_stable(self, u0, nu, nt):
        u_max = np.max(np.abs(u0))
        dt = self.T / (nt - 1)
        clf = u_max / self.dx * dt
        diff = nu * dt / self.dx**2
        print(clf, diff)
        return clf < 1 and diff < 0.5

    def __call__(self, u0, nu, nt=None):
        res = self.solve(u0=u0, nu=nu, nt=nt)
        # replace nan and inf with random values
        # infinite_mask = np.logical_not(np.isfinite(res))
        # res[infinite_mask] = (np.random.rand(*res.shape) * 1e6)[infinite_mask]
        return res


class BurgersExplicit(BurgersSolver):
    def __init__(self, nx, T, boundary="periodic") -> None:
        super().__init__(nx, T, boundary)

    def solve(self, u0, nu, nt=None):
        if nt is None:
            dt, nt = self.get_dt_nt(u0, nu)
        else:
            dt = self.T / (nt - 1)
        u = np.zeros((nt, self.nx))
        u[0, :] = u0.copy()
        for n in range(nt - 1):
            u_left = np.roll(u[n, :], 1)  # Shift right to get u[i-1]
            u_right = np.roll(u[n, :], -1)  # Shift left to get u[i+1]

            # Update all points vectorized using rolled arrays for periodic boundary conditions
            u[n + 1, :] = (
                u[n, :]
                - dt * u[n, :] * (u_right - u_left) / self.dx / 2
                + nu * dt / self.dx**2 * (u_right - 2 * u[n, :] + u_left)
            )
            if self.boundary == "constant":
                u[n + 1, 0] = u0[0]
                u[n + 1, -1] = u0[-1]
        return u


def solve_ldl(L_D_perm, b):
    L, D, perm = L_D_perm
    b = b[perm]
    try:
        y = scipy.linalg.solve_triangular(L, b, lower=True)
        x = scipy.linalg.solve_triangular(L.T, y / np.diag(D), lower=False)
    except ValueError:
        # return an array with same shape and nan values
        x = np.empty_like(b)
        x[:] = np.nan
    return x


class BurgersImplicit(BurgersSolver):
    def __init__(self, nx, T, boundary="periodic") -> None:
        super().__init__(nx, T, boundary)

    def make_A(self, nu, dt):
        a = -dt * nu / self.dx**2
        b = 1 - 2 * a
        A = np.zeros((self.nx, self.nx))
        np.fill_diagonal(A, b)
        np.fill_diagonal(A[:, 1:], a)
        np.fill_diagonal(A[1:, :], a)
        if self.boundary == "periodic":
            A[0, -1] = a
            A[-1, 0] = a
        if self.boundary == "constant":
            A[0, 0] = 1
            A[-1, -1] = 1
            A[0, 1] = 0
            A[-1, -2] = 0
        return A

    def solve(self, u0, nu, nt=None):
        if nt is None:
            dt, nt = self.get_dt_nt(u0, nu)
        else:
            dt = self.T / (nt - 1)
        u = np.zeros((nt, self.nx))
        A = self.make_A(nu, dt)
        L_D_perm = scipy.linalg.ldl(A)
        u[0, :] = u0.copy()
        for n in range(nt - 1):
            du_dx = (np.roll(u[n, :], -1) - np.roll(u[n, :], 1)) / (2 * self.dx)
            rhs = u[n, :] - dt * u[n, :] * du_dx
            if self.boundary == "constant":
                rhs[0] = u0[0]  # u(t, 0) = u(0, 0)
                rhs[-1] = u0[-1]  # u(t, 1) = u(0, 1)
            u[n + 1, :] = solve_ldl(L_D_perm, rhs)
        return u


class BurgersSpectral(BurgersSolver):
    def __init__(self, nx, T, boundary="periodic", verbose=False) -> None:
        super().__init__(nx, T, boundary)
        self.k = np.fft.fftfreq(nx, d=self.dx) * 2 * np.pi
        self.verbose = verbose

    def solve(self, u0, nu, nt=None):
        if nt is None:
            dt, nt = self.get_dt_nt(u0, nu)
        else:
            dt = self.T / (nt - 1)
        u = np.zeros((nt, self.nx))
        u[0, :] = u0.copy()
        if self.verbose:
            pbar = tqdm(range(1, nt))
        else:
            pbar = range(1, nt)
        for n in pbar:
            un_hat = np.fft.fft(u[n - 1, :])
            nonlinear_term = -0.5j * self.k * np.fft.fft(np.fft.ifft(un_hat).real ** 2)
            un_hat = (un_hat + dt * nonlinear_term) / (1 + nu * dt * self.k**2)
            u[n, :] = np.fft.ifft(un_hat).real
        return u


class BurgersFVM(BurgersSolver):
    def __init__(self, nx, T, boundary="periodic") -> None:
        super().__init__(nx, T, boundary)

    def solve(self, u0, nu, nt=None):
        if nt is None:
            dt, nt = self.get_dt_nt(u0, nu)
        else:
            dt = self.T / (nt - 1)
        u = np.zeros((nt, self.nx))
        u[0, :] = u0.copy()
        for n in range(1, nt):
            # Compute convective flux using a difference of squares (assumes periodic BC)
            conv_flux = (
                0.5 * (np.roll(u[n - 1], -1) ** 2 - np.roll(u[n - 1], 1) ** 2) / 2
            )
            # Compute diffusive flux using central differences (assumes periodic BC)
            diff_flux = (
                nu
                * (np.roll(u[n - 1], 1) + np.roll(u[n - 1], -1) - 2 * u[n - 1])
                / self.dx
            )
            # Update velocity field using discrete conservation law
            u[n] = u[n - 1] - dt / self.dx * (conv_flux - diff_flux)
            if self.boundary == "constant":
                u[n, 0] = u0[0]
                u[n, -1] = u0[-1]
        return u


class BurgersFVMSave(BurgersSolver):
    def __init__(self, nx, T, boundary="periodic") -> None:
        super().__init__(nx, T, boundary)

    def solve(self, u0, nu, nt=None):
        if nt is None:
            dt, nt = self.get_dt_nt(u0, nu)
        else:
            dt = self.T / (nt - 1)
        u = np.zeros((nt, self.nx))
        u[0, :] = u0.copy()
        for n in range(1, nt):
            # Compute convective flux using a difference of squares (assumes periodic BC)
            conv_flux = (
                0.5 * (np.roll(u[n - 1], -1) ** 2 - np.roll(u[n - 1], 1) ** 2) / 2
            )
            # Compute diffusive flux using central differences (assumes periodic BC)
            diff_flux = (
                nu
                * (np.roll(u[n - 1], 1) + np.roll(u[n - 1], -1) - 2 * u[n - 1])
                / self.dx
            )
            # Update velocity field using discrete conservation law
            u[n] = u[n - 1] - dt / self.dx * (conv_flux - diff_flux)
            if self.boundary == "constant":
                u[n, 0] = u0[0]
                u[n, -1] = u0[-1]
        return u


class BurgersLaxWendroff(BurgersSolver):
    def __init__(self, nx, T, boundary="periodic") -> None:
        super().__init__(nx, T, boundary)

    def solve(self, u0, nu, nt=None):
        if nt is None:
            dt, nt = self.get_dt_nt(u0, nu)
        else:
            dt = self.T / (nt - 1)
        u = np.zeros((nt, self.nx))
        u[0, :] = u0.copy()
        for n in range(1, nt):
            advection = (
                u[n - 1] * (np.roll(u[n - 1], -1) - np.roll(u[n - 1], 1)) / 2 / self.dx
            )
            diffusion = (
                nu
                * (np.roll(u[n - 1], 1) + np.roll(u[n - 1], -1) - 2 * u[n - 1])
                / self.dx**2
            )
            u[n] = u[n - 1] - dt * (advection - diffusion)
            if self.boundary == "constant":
                u[n, 0] = u0[0]
                u[n, -1] = u0[-1]  # Boundary conditions

        return u


class BurgersTVDOld(BurgersSolver):
    def __init__(self, nx, T, boundary="periodic") -> None:
        super().__init__(nx, T, boundary)

    def flux(u):
        return 0.5 * u**2

    def minmod(a, b):
        assert a.shape == b.shape
        res = np.where(np.abs(a) < np.abs(b), a, b)
        res[a * b <= 0] = 0
        return res

    def solve(self, u0, nu, nt=None):
        if nt is None:
            dt, nt = self.get_dt_nt(u0, nu)
        else:
            dt = self.T / (nt - 1)
        u = np.zeros((nt, self.nx))
        u[0, :] = u0.copy()
        for t in range(1, nt):

            # Minmod slope calculation
            slopes = BurgersTVDOld.minmod(
                (u[t - 1] - np.roll(u[t - 1], 1)), (np.roll(u[t - 1], -1) - u[t - 1])
            )

            # Reconstruct values at interfaces
            u_minus = u[t - 1] - 0.5 * slopes  # u[i-1/2]^+
            u_plus = u[t - 1] + 0.5 * slopes  # u[i+1/2]^-
            mean_flux = 0.5 * (
                BurgersTVDOld.flux(u_plus) + BurgersTVDOld.flux(np.roll(u_minus, -1))
            )

            # Viscous term discretization
            u_xx = (
                np.roll(u[t - 1], -1) - 2 * u[t - 1] + np.roll(u[t - 1], 1)
            ) / self.dx**2

            # Update formula with viscosity
            u[t] = (
                u[t - 1]
                - dt / self.dx * (mean_flux - np.roll(mean_flux, 1))
                + nu * dt * u_xx
            )
        return u


class BurgersTVD(BurgersSolver):
    def __init__(self, nx, T, boundary="periodic") -> None:
        super().__init__(nx, T, boundary)

    def flux(self, u):
        return 0.5 * u**2

    def minmod(self, a, b):
        assert a.shape == b.shape
        res = np.where(np.abs(a) < np.abs(b), a, b)
        res[a * b <= 0] = 0
        return res

    def lax_friedrichs_flux(self, u_left, u_right):
        """Local Lax-Friedrichs flux approximation"""
        alpha = np.maximum(np.abs(u_left), np.abs(u_right))
        return 0.5 * (self.flux(u_left) + self.flux(u_right)) - 0.5 * alpha * (
            u_right - u_left
        )

    def solve(self, u0, nu, nt=None):
        if nt is None:
            dt, nt = self.get_dt_nt(u0, nu)
        else:
            dt = self.T / (nt - 1)
        u = np.zeros((nt, self.nx))
        u[0, :] = u0.copy()
        for t in range(1, nt):
            u_left = np.roll(u[t - 1], 1)  # u[i-1]
            u_right = np.roll(u[t - 1], -1)  # u[i+1]

            slopes = self.minmod(u[t - 1] - u_left, u_right - u[t - 1])

            u_minus = u[t - 1] - 0.5 * slopes
            u_plus = u[t - 1] + 0.5 * slopes
            fluxes = self.lax_friedrichs_flux(u_plus, np.roll(u_minus, -1))

            u[t] = u[t - 1] - dt / self.dx * (fluxes - np.roll(fluxes, 1))
            u[t] += (
                nu
                * dt
                / self.dx**2
                * (np.roll(u[t - 1], 1) - 2 * u[t - 1] + np.roll(u[t - 1], -1))
            )
        return u


class BurgersRiemann(BurgersSolver):
    """Code borrowed from https://zingale.github.io/comp_astro_tutorial/advection_euler/burgers/burgers-methods.html"""

    def __init__(self, nx, T, order=2, boundary="periodic") -> None:
        assert order in [1, 2]
        self.order = order
        super().__init__(nx, T, boundary)

    def flux_update(self, u):
        """compute -div{F} for linear advection"""

        # slope -- we'll do the minmod limiter

        # dl = u_i - u_{i-1}
        dl = u - np.roll(u, 1)
        # dr = u_{i+1} - u_i
        dr = np.roll(u, -1) - u

        d1 = np.where(np.fabs(dl) < np.fabs(dr), dl, dr)
        du = np.where(dl * dr > 0.0, d1, 0.0)

        # compute the left and right interface states
        # Note that there are 1 more interfaces than zones

        # u_{i-1/2,R} = u_i - 1/2 du_i
        ur = u - 0.5 * du

        # u_{i-1/2,L} = u_{i-1} + 1/2 du_{i-1}
        ul = np.roll(u, 1) + 0.5 * np.roll(du, 1)

        # now do the Riemann problem

        S = 0.5 * (ul + ur)
        ushock = np.where(S > 0.0, ul, ur)
        ushock = np.where(S == 0.0, 0.0, ushock)

        # rarefaction solution
        urare = np.where(ur <= 0.0, ur, 0.0)
        urare = np.where(ul >= 0.0, ul, urare)

        # if we are compressive, then we are a shock
        us = np.where(ul > ur, ushock, urare)

        flux_diff = (0.5 * us**2 - 0.5 * np.roll(us, -1) ** 2) / self.dx

        return flux_diff

    def solve(self, u0, nu, nt=None):
        if nt is None:
            dt, nt = self.get_dt_nt(u0, nu)
        else:
            dt = self.T / (nt - 1)
        u = np.zeros((nt, self.nx))
        u[0, :] = u0.copy()
        for t in range(1, nt):
            k1 = self.flux_update(u[t - 1])
            """advection = (
                nu
                * (np.roll(u[t - 1], -1) - 2 * u[t - 1] + np.roll(u[t - 1], 1))
                / self.dx**2
            )"""
            if self.order == 1:
                u[t] = u[t - 1] + dt * (k1)  # + advection)
                continue

            utemp = u[t - 1] + 0.5 * dt * (k1)  # + advection)

            k2 = self.flux_update(utemp)
            """advection = (
                nu * (np.roll(utemp, -1) - 2 * utemp + np.roll(utemp, 1)) / self.dx**2
            )"""

            u[t] = u[t - 1] + dt * (k2)  # + advection)
        return u
