import numpy as nm
from sfepy import data_dir
import meshio
import glob
import pandas as pd

f = 5 * 1e9  # change this to 2.4 or 5 if you want to simulate frequencies
# typically used by your Wifi
c0 = 3e8  # m/s
k = 2 * nm.pi * f / c0

max_len = 100

for name in [
    "acc2D_0",
    "acc2D_left",
    "acc2D_right",
    "acc2D_top",
    "acc2D_center",
]:
    seed = len(
        glob.glob(
            f"/Users/theobourdais/Desktop/Caltech/Research/model_aggregation/ModelAggregation/PDEsolvers/SfePy/sims/{name}_*.vtk"
        )
    )
    print(f"there are {seed} results already for {name}")
    if seed == max_len:
        continue
    else:
        break
print("-" * 20 + f"\n computing simulation {seed} for {name}\n" + "-" * 20)


filename_mesh = f"/Users/theobourdais/Desktop/{name}.mesh"

regions = {
    "Omega": "all",
    "Water": "cells by random_right",
    "Air": "cells by random_left",
    "Source": "cells by ball",
    "Gamma": ("vertices of surface", "facet"),
}

# air and walls have different material parameters, hence the 1. and 2.4 factors
materials = {
    "material": (
        {
            "kn_square": {"Air": (k * 1.0) ** 2, "Water": (k * 1.333) ** 2},
        },
    ),
    "boundary": ({"kn": 1j * k * 1.333},),
    "source_func": "source_func",
}
m = meshio.read(filename_mesh)
min_x, max_x = min(m.points[:, 0]), max(m.points[:, 0])
min_y, max_y = min(m.points[:, 1]), max(m.points[:, 1])
# sample uniform point in the domain


path = "/Users/theobourdais/Desktop/Caltech/Research/model_aggregation/ModelAggregation/PDEsolvers/SfePy/sims/"
random_inputs = dict(pd.read_csv(path + "random_inputs.csv").loc[seed])
random_point = nm.array([random_inputs["point_x"], random_inputs["point_y"]])[None, :]
u = nm.array([random_inputs["u_x"], random_inputs["u_y"]])[:, None]


def random_left(coors, domain=None):
    proj = (coors - random_point) @ u
    flag = nm.where((proj < 0))[0]

    return flag


def random_right(coors, domain=None):
    # same as above but for the right side
    proj = (coors - random_point) @ u
    flag = nm.where((proj > 0))[0]
    return flag


has_empty_region = random_left(m.points).size == 0 or random_right(m.points).size == 0
while has_empty_region:
    random_point = (
        nm.array([min_x + max_x, min_y + max_y]) / 2
        + nm.array([max_x - min_x, max_y - min_y]) * (nm.random.rand(1, 2) - 0.5) / 2
    )
    # sample uniform 2d unit vector
    theta = 2 * nm.pi * nm.random.rand(1)
    u = nm.array([nm.cos(theta), nm.sin(theta)])

    df = pd.read_csv(path + "random_inputs.csv")
    df.loc[seed, "point_x"] = random_point[0, 0]
    df.loc[seed, "point_y"] = random_point[0, 1]
    df.loc[seed, "u_x"] = u[0, 0]
    df.loc[seed, "u_y"] = u[1, 0]
    df.to_csv(path + "random_inputs.csv", index=False)

    # save the random point and vector for later use in text file
    def random_left(coors, domain=None):
        proj = (coors - random_point) @ u
        flag = nm.where((proj < 0))[0]

        return flag

    def random_right(coors, domain=None):
        # same as above but for the right side
        proj = (coors - random_point) @ u
        flag = nm.where((proj > 0))[0]
        return flag

    has_empty_region = (
        random_left(m.points).size == 0 or random_right(m.points).size == 0
    )


def ball(coors, domain=None):
    x, y = coors[:, 0], coors[:, 1]
    c1 = nm.array([0.4, 0.5])
    r1 = nm.linalg.norm(coors - c1, axis=1)
    # same with c2 at 2,2
    c2 = nm.array([2.5, 0.3])
    r2 = nm.linalg.norm(coors - c2, axis=1)
    # same with c3 at 0.5,2
    c3 = nm.array([1.9, 1.9])
    r3 = nm.linalg.norm(coors - c3, axis=1)
    r = 0.1
    flag = nm.where((r1 < r) | (r2 < r) | (r3 < r))[0]
    return flag


def source_func(ts, coors, mode=None, **kwargs):
    """The source function for the antenna."""
    epsilon = 0.1
    x_center_1 = nm.array([0.4, 0.5])
    x_center_2 = nm.array([2.5, 0.3])
    x_center_3 = nm.array([1.9, 1.9])
    if mode == "qp":
        dists_1 = nm.abs(nm.sum(nm.square(coors - x_center_1), axis=1))
        dists_2 = nm.abs(nm.sum(nm.square(coors - x_center_2), axis=1))
        dists_3 = nm.abs(nm.sum(nm.square(coors - x_center_3), axis=1))
        val = (
            3
            / nm.pi
            / epsilon**2
            * nm.where((1 - dists_1 / epsilon) > 0, (1 - dists_1 / epsilon), 0)
        )
        val += (
            3
            / nm.pi
            / epsilon**2
            * nm.where((1 - dists_2 / epsilon) > 0, (1 - dists_2 / epsilon), 0)
        )
        val += (
            3
            / nm.pi
            / epsilon**2
            * nm.where((1 - dists_3 / epsilon) > 0, (1 - dists_3 / epsilon), 0)
        )

        return {"val": val[:, nm.newaxis, nm.newaxis]}


functions = {
    "source_func": (source_func,),
    "random_left": (random_left,),
    "random_right": (random_right,),
    "ball": (ball,),
}

fields = {
    "electric_field": ("complex", 1, "Omega", 1),
}

variables = {
    "E": ("unknown field", "electric_field", 1),
    "q": ("test field", "electric_field", "E"),
}

ebcs = {}

integrals = {
    "i": 2,
}

equations = {
    "Helmholtz equation": """- dw_laplace.i.Omega(q, E)
        + dw_dot.i.Omega(material.kn_square, q, E)
        + dw_dot.i.Gamma(boundary.kn, q, E)
        = dw_integrate.i.Source(source_func.val, q)
        """
}

solvers = {
    "ls": ("ls.auto_direct", {}),
    "newton": (
        "nls.newton",
        {
            "i_max": 1,
        },
    ),
}

options = {
    # "refinement_level": 3,
}
