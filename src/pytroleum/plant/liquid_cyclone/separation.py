import numpy as np
from scipy.integrate import solve_ivp

from numpy.typing import NDArray

from pytroleum.plant.liquid_cyclone.flowsheets import FlowSheet
from pytroleum.plant.liquid_cyclone.interfaces import Design, VelocityField

INTEGRATION_TIME_CAP = 60*60
MAX_INTEGRATION_STEP = 0.05
INTEGRATION_METHOD = "LSODA"

# Implicit methods work better for this problem, there are multiple in scipy,
# any of them works fine, LSODA is set as default


def droplet_motion_equations(t, Y, drop_diameter: float,
                             setup: tuple[FlowSheet, Design, VelocityField]):
    flowsheet, design, velocity_field = setup

    radial_velocity = (
        velocity_field.radial_component(Y, flowsheet, design) +
        velocity_field.drop_slip_velocity(Y, drop_diameter, flowsheet, design))

    if _wall_collision(t, Y, drop_diameter, setup) < 0:
        breakpoint()

    axial_velocity = velocity_field.axial_component(Y, flowsheet, design)

    dY_dt = radial_velocity, axial_velocity

    return dY_dt


def _wall_collision(
        t, Y, drop_diameter: float, setup: tuple[FlowSheet, Design, VelocityField]):
    _, design, _ = setup
    r, z = Y
    return design.wall(z)-r


def _top_plane_collision(
        t, Y, drop_diameter: float, setup: tuple[FlowSheet, Design, VelocityField]):
    _, z = Y
    return z


# ADjusting events
_wall_collision.terminal = True
_wall_collision.direction = 0

_top_plane_collision.terminal = True
_top_plane_collision.direction = 0


def drop_trajectroy(drop_diameter: float, setup: tuple[FlowSheet, Design, VelocityField]):

    _, design, velocity_field = setup

    z0 = design.model_length
    r0 = design.wall(z0)*velocity_field.ndim_reversal_radius

    solution = solve_ivp(
        fun=droplet_motion_equations,
        t_span=[0, -INTEGRATION_TIME_CAP],
        y0=[r0, z0],
        method=INTEGRATION_METHOD,
        events=[_wall_collision, _top_plane_collision],
        args=(drop_diameter, setup),
        max_step=MAX_INTEGRATION_STEP)

    return solution
