r"""
Each of the two equations describes a flow in one compartment of a porous
medium. The equations are based on the Darcy flow and the i-th compartment is
defined in :math:`\Omega_{i}`.

.. math::
    \int_{\Omega_{i}} K^{i} \nabla p^{i} \cdot \nabla q^{i}+\int_{\Omega_{i}}
    \sum_{j} \bar{G}\alpha_{k} \left( p^{i}-p^{j} \right)q^{i}
    = \int_{\Omega_{i}} f^{i} q^{i},
.. math::
    \forall q^{i} \in Q^{i}, \quad i,j=1,2 \quad \mbox{and} \quad i\neq j,

where :math:`K^{i}` is the local permeability of the i-th compartment,
:math:`\bar{G}\alpha_{k} = G^{i}_{j}` is the perfusion coefficient
related to the compartments :math:`i` and :math:`j`, :math:`f^i` are
sources or sinks which represent the external flow into the i-th
compartment and :math:`p^{i}` is the pressure in the i-th compartment.
"""

from __future__ import absolute_import
from sfepy.base.base import Struct
import numpy as nm
from sfepy import data_dir
import matplotlib.pyplot as plt

nm.random.seed(1)

filename_mesh = "simple_mesh.mesh"

materials = {"mat": ("mat_fun")}
eps = 1e-5
regions = {
    "Omega": "all",
    "left": (f"vertices in x < {eps}", "facet"),
    "right": (f"vertices in x > {1-eps}", "facet"),
    "bottom": (f"vertices in y < {eps}", "facet"),
    "top": (f"vertices in y > {1-eps}", "facet"),
}

fields = {"pressure": ("real", 1, "Omega", 1)}

variables = {"p": ("unknown field", "pressure"), "q": ("test field", "pressure", "p")}

ebcs = {
    "P1": ("left", {"p.all": 0.0}),
    "P2": ("right", {"p.all": 0.0}),
    "P3": ("bottom", {"p.all": 0.0}),
    "P4": ("top", {"p.all": 0.0}),
}

equations = {
    "komp1": """dw_diffusion.5.Omega(mat.K, q, p)
              = dw_integrate.5.Omega(mat.f, q)""",
}

solvers = {
    "ls": ("ls.scipy_direct", {}),
    "newton": (
        "nls.newton",
        {
            "i_max": 1,
            "eps_a": 1e-6,
            "eps_r": 1.0,
        },
    ),
}

gaussian_coeffs = nm.random.normal(size=(100, 100))


def eval_sample(x, start_freq=1):
    k = nm.linspace(
        start_freq, start_freq + gaussian_coeffs.shape[0] - 1, gaussian_coeffs.shape[0]
    )
    sinuses = (
        nm.sin(nm.pi * k[None, :] * x[:, :1])[:, :, None]
        * nm.sin(nm.pi * k[None, :] * x[:, 1:])[:, None, :]
    )

    coeffs = nm.pi**2 * (k[None, :] ** 2 + k[:, None] ** 2)
    vals = nm.sum(sinuses * gaussian_coeffs / coeffs, axis=(1, 2))
    return nm.where(vals >= 0, 12, 3)


x = nm.linspace(0, 1, 100)
y = nm.linspace(0, 1, 100)
xv, yv = nm.meshgrid(x, y)
X = nm.stack([xv.flatten(), yv.flatten()], axis=-1)
Z = eval_sample(X, 1).reshape((100, 100))
plt.imshow(Z)
plt.colorbar()
plt.savefig(
    f"/Users/theobourdais/Desktop/Caltech/Research/model_aggregation/ModelAggregation/PDEsolvers/input_coeff_{nm.random.rand()}.png"
)
plt.close()


def mat_fun(ts, coors, mode=None, **kwargs):
    if mode == "qp":
        nqp, dim = coors.shape
        assert dim == 2
        vals = eval_sample(coors, start_freq=1)
        K = nm.eye(dim, dtype=nm.float64)
        K2 = nm.tile(K, (nqp, 1, 1))
        out = {
            "K": K * vals[:, None, None],
            "f": nm.ones((nqp, 1, 1), dtype=nm.float64),
        }

        return out


functions = {
    "mat_fun": (mat_fun,),
}

# options = {
#    "post_process_hook": "postproc",
# }


def postproc(out, pb, state, extend=False):
    alpha = pb.evaluate("ev_integrate_mat.5.Omega(mat.G_alfa, p1)", mode="el_avg")
    out["alpha"] = Struct(
        name="output_data",
        mode="cell",
        data=alpha.reshape(alpha.shape[0], 1, 1, 1),
        dofs=None,
    )
    return out
