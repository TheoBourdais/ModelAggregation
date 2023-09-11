from __future__ import absolute_import
from sfepy.examples.linear_elasticity.its2D_1 import *

from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson
from sfepy.discrete.fem.utils import refine_mesh
from sfepy.base.base import Struct
from sfepy import data_dir

# Fix the mesh file name if you run this file outside the SfePy directory.
filename_mesh = "/Users/theobourdais/Desktop/mesh2D_1.mesh"


output_dir = "."  # set this to a valid directory you have write access to

young = 2000.0  # Young's modulus [MPa]
poisson = 0.4  # Poisson's ratio

options = {
    "output_dir": output_dir,
}

regions = {
    "Omega": "all",
    "Top": ("vertex 5", "vertex"),
    "Right": ("vertex 3", "vertex"),
    "Left": ("vertex 1", "vertex"),
}

materials = {
    "Asphalt": ({"D": stiffness_from_youngpoisson(2, young, poisson)},),
    "Load": ({".val": [0.0, -1000.0]},),
}

fields = {
    "displacement": ("real", "vector", "Omega", 1),
}

equations = {
    "balance_of_forces": """dw_lin_elastic.2.Omega(Asphalt.D, v, u)
      = dw_point_load.0.Top(Load.val, v)""",
}

variables = {
    "u": ("unknown field", "displacement", 0),
    "v": ("test field", "displacement", "u"),
}

ebcs = {
    "fix_right": ("Right", {"u.all": 0.0}),
    "fix_left": ("Left", {"u.all": 0.0}),
}

solvers = {
    "ls": ("ls.scipy_direct", {}),
    "newton": (
        "nls.newton",
        {
            "i_max": 1,
            "eps_a": 1e-6,
        },
    ),
}


def stress_strain(out, pb, state, extend=False):
    """
    Calculate and output strain and stress for given displacements.
    """

    ev = pb.evaluate
    strain = ev("ev_cauchy_strain.2.Omega(u)", mode="el_avg")
    stress = ev(
        "ev_cauchy_stress.2.Omega(Asphalt.D, u)", mode="el_avg", copy_materials=False
    )

    out["cauchy_strain"] = Struct(
        name="output_data", mode="cell", data=strain, dofs=None
    )
    out["cauchy_stress"] = Struct(
        name="output_data", mode="cell", data=stress, dofs=None
    )

    return out


asphalt = materials["Asphalt"][0]
asphalt.update({"D": stiffness_from_youngpoisson(2, young, poisson)})
options.update(
    {
        "post_process_hook": "stress_strain",
    }
)
