import jax.numpy as jnp
import numpy as onp
from jax import grad, jit, vmap
from jax.config import config
from tensorflow_probability.substrates.jax.math import bessel_kve
from jax.scipy.special import gammaln
from scipy.special import gamma

config.update("jax_enable_x64", True)

from functools import partial  # for jit to make codes faster


class Differentiable_kernel(object):
    def __init__(self):
        pass

    @partial(jit, static_argnums=(0,))
    def D_x1_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 0)(x1, x2, y1, y2)
        return val

    @partial(jit, static_argnums=(0,))
    def D_x2_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 1)(x1, x2, y1, y2)
        return val

    @partial(jit, static_argnums=(0,))
    def DD_x2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 1), 1)(x1, x2, y1, y2)
        return val

    @partial(jit, static_argnums=(0,))
    def D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 2)(x1, x2, y1, y2)
        return val

    @partial(jit, static_argnums=(0,))
    def D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 3)(x1, x2, y1, y2)
        return val

    @partial(jit, static_argnums=(0,))
    def DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 3), 3)(x1, x2, y1, y2)
        return val

    @partial(jit, static_argnums=(0,))
    def D_x1_D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 0), 2)(x1, x2, y1, y2)
        return val

    @partial(jit, static_argnums=(0,))
    def D_x1_D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 0), 3)(x1, x2, y1, y2)
        return val

    @partial(jit, static_argnums=(0,))
    def D_x1_DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(self.kappa, 0), 3), 3)(x1, x2, y1, y2)
        return val

    @partial(jit, static_argnums=(0,))
    def D_x2_D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 1), 3)(x1, x2, y1, y2)
        return val

    @partial(jit, static_argnums=(0,))
    def D_x2_D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(self.D_x2_kappa, 2)(x1, x2, y1, y2)
        return val

    @partial(jit, static_argnums=(0,))
    def D_x2_DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(self.kappa, 1), 3), 3)(x1, x2, y1, y2)
        return val

    @partial(jit, static_argnums=(0,))
    def DD_x2_DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(grad(self.kappa, 1), 1), 3), 3)(x1, x2, y1, y2)
        return val

    @partial(jit, static_argnums=(0,))
    def Delta_x_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 0), 0)(x1, x2, y1, y2)
        val += grad(grad(self.kappa, 1), 1)(x1, x2, y1, y2)
        return val

    @partial(jit, static_argnums=(0,))
    def Delta_y_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 2), 2)(x1, x2, y1, y2)
        val += grad(grad(self.kappa, 3), 3)(x1, x2, y1, y2)
        return val

    @partial(jit, static_argnums=(0,))
    def Delta_x_Delta_y_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.Delta_y_kappa, 0), 0)(x1, x2, y1, y2)
        val += grad(grad(self.Delta_y_kappa, 1), 1)(x1, x2, y1, y2)
        return val

    @partial(jit, static_argnums=(0,))
    def Delta_x_D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(self.Delta_x_kappa, 2)(x1, x2, y1, y2)
        return val

    @partial(jit, static_argnums=(0,))
    def Delta_x_D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(self.Delta_x_kappa, 3)(x1, x2, y1, y2)
        return val

    def apply(self, f_to_apply, X, Y):
        assert X.shape[1] == 2 and Y.shape[1] == 2
        X_tiled = jnp.transpose(jnp.tile(X, (Y.shape[0], 1, 1)), (1, 0, 2))
        Y_tiled = jnp.tile(Y, (X.shape[0], 1, 1))

        f = getattr(self, f_to_apply)
        X1 = X_tiled[:, :, 0].flatten()
        X2 = X_tiled[:, :, 1].flatten()
        Y1 = Y_tiled[:, :, 0].flatten()
        Y2 = Y_tiled[:, :, 1].flatten()
        val = vmap(f)(X1, X2, Y1, Y2)
        res = onp.reshape(val, (X.shape[0], Y.shape[0]))
        return onp.nan_to_num(res, nan=1.0)


class Differentiable_gaussian_kernel(Differentiable_kernel):
    def __init__(self, sigma):
        self.sigma = sigma

    @partial(jit, static_argnums=(0,))
    def kappa(self, x1, x2, y1, y2):
        return jnp.exp(-(1 / (2 * self.sigma**2)) * ((x1 - y1) ** 2 + (x2 - y2) ** 2))


class Differentiable_matern_kernel(Differentiable_kernel):
    def __init__(self, nu, l):
        self.nu = nu
        self.l = l
        self.mult_Z = jnp.sqrt(2 * nu) / l
        self.scale = 1 / 2 ** (nu - 1) / gamma(nu)

    @partial(jit, static_argnums=(0,))
    def kappa(self, x1, x2, y1, y2):
        z = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2) * self.mult_Z
        return z**self.nu * bessel_kve(self.nu, z) * jnp.exp(-z) * self.scale
