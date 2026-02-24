import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import brentq
from scipy.special import erf

from numpy.typing import NDArray

from pytroleum.plant.liquid_cyclone.flowsheets import FlowSheet, FlowSpec
from pytroleum.plant.liquid_cyclone.interfaces import Design, VelocityField


INTEGRATION_TIME_CAP = 60*60
MAX_INTEGRATION_STEP = 1.0
INTEGRATION_METHOD = "LSODA"

ROOTFINDING_MAX_DROP_SIZE = 100e-6

_SIZE_DISTRIBUTION_AUXILIARY_CONSTANT = 0.394
PROBABILITY_DENSITY_ZERO_TOLERANCE = 0.01e-6
NUMBER_OF_DROPS_FOR_GRADE_EFFICIENCY = 50

# Implicit methods work better for this problem, there are multiple in scipy,
# any of them works fine, LSODA is set as default

# region grade efficiency


def droplet_motion_equations(t, Y, drop_diameter: float,
                             setup: tuple[FlowSheet, Design, VelocityField]):
    flowsheet, design, velocity_field = setup

    radial_velocity = (
        velocity_field.radial_component(Y, flowsheet, design) +
        velocity_field.drop_slip_velocity(Y, drop_diameter, flowsheet, design))

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


def drop_trajectroy(drop_diameter: float, setup: tuple[FlowSheet, Design, VelocityField],
                    full_output: bool = False):

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

    if full_output:
        return solution
    else:
        return solution.y


def drop_final_point(
        drop_diameter: float, setup: tuple[FlowSheet, Design, VelocityField]):
    # convenience function
    r, z = drop_trajectroy(drop_diameter, setup)
    return r[-1], z[-1]


def _largest_drop_objective(
        drop_diameter: float, setup: tuple[FlowSheet, Design, VelocityField]):

    drop_diameter = float(drop_diameter)
    _, design, _ = setup
    r, z = drop_final_point(drop_diameter, setup)

    f = np.sqrt(r**2+z**2) - design.wall(0)

    return f


def solve_largest_drop(setup: tuple[FlowSheet, Design, VelocityField]):

    # should improve rootfinding routines

    _, design, _ = setup

    # fsolve fails becuase there is no distinct zero-crossing, curve is tangent
    # to x-axis
    largest_diameter = brentq(
        f=lambda d: _largest_drop_objective(d, setup),
        a=0, b=ROOTFINDING_MAX_DROP_SIZE)

    return largest_diameter


def compute_grade_efficiency(
        drop_diameter: float, setup: tuple[FlowSheet, Design, VelocityField]):

    flowsheet, design, velocity_filed = setup
    Q_in = flowsheet.flow_rate[FlowSpec.INLET]

    r_final, _ = drop_final_point(drop_diameter, setup)
    ndim_r_final = r_final/design.wall(0)

    integral = (velocity_filed._ndim_profile_dotprod_polynom(1) -
                velocity_filed._ndim_profile_dotprod_polynom(ndim_r_final))

    G = 1 - 2*np.pi*integral/Q_in

    return G


def build_grade_efficiency_curve(setup: tuple[FlowSheet, Design, VelocityField],
                                 num_of_drops=NUMBER_OF_DROPS_FOR_GRADE_EFFICIENCY):

    largest_drop_size = solve_largest_drop(setup)
    drop_diameters = np.linspace(
        0, largest_drop_size, num_of_drops)  # type: ignore

    grade_efficiency = []
    for d in drop_diameters:
        grade_efficiency.append(compute_grade_efficiency(d, setup))

    # recast into numpy array
    grade_efficiency = np.array(grade_efficiency)

    return drop_diameters, grade_efficiency


def extract_d50(drop_diameters, grade_efficiency):
    # conveninece interpolation function
    d50 = np.interp(0.5, grade_efficiency, drop_diameters)
    return d50

# region size distribution

# TODO :
# implement distribution functions, evaluate volumetric efficiency


def _auxiliary_zeta(drop_diameter, percentiles: tuple[float, float, float]):

    d50, d90, d100 = percentiles
    a = (d100-d50)/d50

    zeta = np.log(a*drop_diameter/(d100-drop_diameter))

    return zeta


def _auxiliary_delta(percentiles):

    d50, d90, d100 = percentiles
    nu90 = d90/(d100-d90)
    nu50 = d50/(d100-d50)

    delta = _SIZE_DISTRIBUTION_AUXILIARY_CONSTANT/np.log10(nu90/nu50)

    return delta


def probability_density_func(drop_diameter, percentiles):
    _, _, d100 = percentiles

    drop_diameter = np.atleast_1d(drop_diameter)

    is_valid_diameter = (
        (drop_diameter < d100)*(drop_diameter > PROBABILITY_DENSITY_ZERO_TOLERANCE))
    valid_diameters = drop_diameter[is_valid_diameter]

    zeta = np.zeros_like(drop_diameter)

    zeta[is_valid_diameter] = _auxiliary_zeta(valid_diameters, percentiles)
    delta = _auxiliary_delta(percentiles)

    pdf = np.zeros_like(drop_diameter)

    pdf[is_valid_diameter] = (
        delta*d100 /
        (drop_diameter[is_valid_diameter]*(d100-drop_diameter[is_valid_diameter])) *
        np.exp(-(delta*zeta[is_valid_diameter])**2) / np.sqrt(np.pi))

    return pdf


def cumulative_distribution_func(drop_diameter, percentiles):

    _, _, d100 = percentiles

    is_valid_diameter = (
        (drop_diameter < d100)*(drop_diameter > PROBABILITY_DENSITY_ZERO_TOLERANCE))
    valid_diameters = drop_diameter[is_valid_diameter]

    zeta = np.zeros_like(drop_diameter)

    zeta[is_valid_diameter] = _auxiliary_zeta(valid_diameters, percentiles)
    delta = _auxiliary_delta(percentiles)

    cdf = np.zeros_like(drop_diameter)
    cdf[is_valid_diameter] = 1 - (1-erf(zeta[is_valid_diameter]*delta))/2

    # After 100-th percentile this should be 1
    cdf[drop_diameter > d100] = 1

    # TODO : fix right boundary

    return cdf


def evaluate_total_efficiency(
        setup: tuple[FlowSheet, Design, VelocityField],
        percentiles: tuple[float, float, float],
        margin_factor=5):

    # Evaluate overlapping part
    d, G = build_grade_efficiency_curve(setup)
    pdf = probability_density_func(d, percentiles)

    I_overlap = np.trapezoid(G*pdf, d)

    # Evaluate pdf-only part
    I_pdf_only = quad(
        lambda size: probability_density_func(size, percentiles),
        d[-1], margin_factor*percentiles[-1])[0]
    # upper limit is deduced from 100-th percentile, which is stored last,
    # and quad return tuple with result and error estimation, we only need result,
    # that's why we index by 0

    # Total efficiency is a sum of two integrals above
    efficiency = I_overlap + I_pdf_only

    return efficiency


if __name__ == "__main__":
    # Sanity checks on distribution functions

    import matplotlib.pyplot as plt

    TO_MICROMETERS = 1e6

    percentiles = (20e-6, 30e-6, 120e-6)
    diameter = np.linspace(0, 500e-6, 500)
    # TODO :
    # resolve limits for d = 0 and d = d100

    pdf = probability_density_func(diameter, percentiles)
    cdf = cumulative_distribution_func(diameter, percentiles)

    diameter = diameter*TO_MICROMETERS

    fig, ax = plt.subplots()
    ax.set_title("Probability Density Function")
    ax.set_xlabel(r"d [$\mu m$]")
    ax.set_ylabel(r"$f(d)$")
    ax.set_xlim((0, diameter[-1]))
    ax.plot(diameter, pdf)
    ax.grid(True)

    fig, ax = plt.subplots()
    ax.set_title("Cumulative Distribution Function")
    ax.set_xlabel(r"d [$\mu m$]")
    ax.set_ylabel(r"$F(d)$")
    ax.set_xlim((0, diameter[-1]))
    ax.plot(diameter, cdf)
    ax.grid(True)
