"""
Hydrocyclone inverse problem: find body diameter Dc
for a given flow rate Q, phase properties, and concentration.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import fsolve

from pytroleum.plant.solid_cyclone.geometry import (
    CycloneDesign,
    HydrocycloneDiameters,
    CYCLONE_CONE_ANGLE_MIN,
    CYCLONE_CONE_ANGLE_MAX,
)
from pytroleum.plant.solid_cyclone.inputs import SizeDistribution
from pytroleum.plant.solid_cyclone.models import BaseHydrocyclone
from pytroleum.plant.solid_cyclone.efficiency import (
    calculate_reduced_grade_efficiency,
    calculate_reduced_total_efficiency,
    calculate_total_efficiency,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOL_RELATIVE = 1e-4  # allowable discrepancy between forward and inverse problems
_V_IN_INITIAL = 9.0   # m/s — typical velocity for initial approximation

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _validate_cone_angle(design: CycloneDesign) -> None:
    """Validate cone angle before solving the inverse problem."""
    if not (CYCLONE_CONE_ANGLE_MIN <= design.cone_angle <= CYCLONE_CONE_ANGLE_MAX):
        raise ValueError(
            f"angle = {design.cone_angle:.1f}° out of valid range "
            f"[{CYCLONE_CONE_ANGLE_MIN}°, {CYCLONE_CONE_ANGLE_MAX}°]."
        )


def _initial_Dc(Q: float, design: CycloneDesign) -> float:
    """
    Initial Dc approximation for fsolve.

    Derived from v_in = Q / (pi*Di^2/4) at the typical velocity _V_IN_INITIAL:
      Di0 = sqrt(4*Q / (v_in*pi)),  Dc0 = Di0 / (Di/Dc)
    """
    Di_Dc_ratio = design.diameter_proportions[HydrocycloneDiameters.I]
    Di0 = np.sqrt(4.0 * Q / (_V_IN_INITIAL * np.pi))

    return Di0 / Di_Dc_ratio


def _residual_cut_size(
        Dc: float,
        cut_size_target: float,
        hydrocyclone: BaseHydrocyclone,
) -> float:
    """Residual for problem 1: f(Dc) = d50'(Dc, Q) - d50'_target."""
    hydrocyclone.design = CycloneDesign(
        Dc, hydrocyclone.design.diameter_proportions,
        hydrocyclone.design.length_proportions,
        hydrocyclone.design.cone_angle)

    hydrocyclone.calculate_from_flow_rate()

    return hydrocyclone.reduced_cut_size - cut_size_target


def _residual_efficiency(
        Dc: float,
        efficiency_target: float,
        hydrocyclone: BaseHydrocyclone,
        size_dist: SizeDistribution,
) -> NDArray | np.floating:
    """Residual for problem 2: f(Dc) = E_T(Dc, Q) - E_T_target."""
    hydrocyclone.design = CycloneDesign(
        Dc, hydrocyclone.design.diameter_proportions,
        hydrocyclone.design.length_proportions,
        hydrocyclone.design.cone_angle)

    hydrocyclone.calculate_from_flow_rate()

    reduced_grade_efficiency = calculate_reduced_grade_efficiency(
        size_dist.particle_diameters, hydrocyclone.reduced_cut_size,
        'plitt', hydrocyclone.m, hydrocyclone.alpha)

    reduced_total_efficiency = calculate_reduced_total_efficiency(
        size_dist.particle_diameters, reduced_grade_efficiency, size_dist.k, size_dist.n)

    total_efficiency = calculate_total_efficiency(
        reduced_total_efficiency, hydrocyclone.water_flow_ratio)

    return total_efficiency - efficiency_target


# ---------------------------------------------------------------------------
# Public inverse problem functions
# ---------------------------------------------------------------------------

def find_Dc_by_cut_size(
        cut_size_target: float,
        hydrocyclone: BaseHydrocyclone,
        Dc0: float | None = None,
) -> float:
    """
    Problem 1. Find Dc such that d50'(Dc, Q) = cut_size_target.
    Solves: f(Dc) = d50'(Dc, Q) - cut_size_target = 0
    """
    _validate_cone_angle(hydrocyclone.design)

    if Dc0 is None:
        Dc0 = _initial_Dc(
            hydrocyclone.conditions.feed_volumetric_flow_rate, hydrocyclone.design)

    Dc_solution = fsolve(
        _residual_cut_size, x0=Dc0,
        args=(cut_size_target, hydrocyclone),
    )[0]

    return Dc_solution


def find_Dc_by_efficiency(
        efficiency_target: float,
        hydrocyclone: BaseHydrocyclone,
        size_dist: SizeDistribution,
        Dc0: float | None = None,
) -> float:
    """
    Problem 2. Find Dc such that E_T(Dc, Q) = efficiency_target.

    Solves: f(Dc) = E_T(Dc, Q) - efficiency_target = 0
    """
    _validate_cone_angle(hydrocyclone.design)

    if Dc0 is None:
        Dc0 = _initial_Dc(
            hydrocyclone.conditions.feed_volumetric_flow_rate, hydrocyclone.design)

    Dc_solution = fsolve(
        _residual_efficiency, x0=Dc0,
        args=(efficiency_target, hydrocyclone, size_dist),
    )[0]

    return Dc_solution
