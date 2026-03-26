"""
Hydrocyclone inverse problem: find body diameter Dc
for a given flow rate Q, phase properties, and concentration.
"""

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import fsolve

from pytroleum.plant.solid_cyclone.geometry import (
    CycloneDesign,
    HydrocycloneDiameters,
    CYCLONE_CONE_ANGLE_MIN,
    CYCLONE_CONE_ANGLE_MAX,
)
from pytroleum.plant.solid_cyclone.inputs import (
    PhysicalProperties,
    OperationConditions,
    SizeDistribution,
)
from pytroleum.plant.solid_cyclone.models import BaseHydrocyclone
from pytroleum.plant.solid_cyclone.efficiency import (
    calculate_reduced_grade_efficiency,
    calculate_reduced_total_efficiency,
    calculate_total_efficiency,
)
from pytroleum.plant.solid_cyclone.utils import _minor_divider, _major_divider


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


def _compute_efficiencies(
        hydrocyclone: BaseHydrocyclone,
        size_dist: SizeDistribution,
        model: Literal['plitt', 'lynch_rao'] = 'plitt',
) -> tuple[NDArray | np.floating, NDArray | np.floating]:
    """Calculate reduced E_T' and total E_T efficiencies."""
    reduced_grade_efficiency = calculate_reduced_grade_efficiency(
        size_dist.particle_diameters,
        hydrocyclone.reduced_cut_size,
        model,
        hydrocyclone.m,
        hydrocyclone.alpha,
    )

    reduced_total_efficiency = calculate_reduced_total_efficiency(
        size_dist.particle_diameters, reduced_grade_efficiency,
        size_dist.k, size_dist.n)

    total_efficiency = calculate_total_efficiency(
        reduced_total_efficiency, hydrocyclone.water_flow_ratio)

    return reduced_total_efficiency, total_efficiency


def _residual_cut_size(
        Dc: float,
        cut_size_target: float,
        hydrocyclone: BaseHydrocyclone,
        properties: PhysicalProperties,
) -> float:
    """Residual for problem 1: f(Dc) = d50'(Dc, Q) - d50'_target."""
    hydrocyclone.design = CycloneDesign(
        Dc, hydrocyclone.design.diameter_proportions,
        hydrocyclone.design.length_proportions,
        hydrocyclone.design.cone_angle)

    hydrocyclone.calculate_from_flow_rate(properties)

    return hydrocyclone.reduced_cut_size - cut_size_target


def _residual_efficiency(
        Dc: float,
        efficiency_target: float,
        hydrocyclone: BaseHydrocyclone,
        properties: PhysicalProperties,
        size_dist: SizeDistribution,
) -> NDArray | np.floating:
    """Residual for problem 2: f(Dc) = E_T(Dc, Q) - E_T_target."""
    hydrocyclone.design = CycloneDesign(
        Dc, hydrocyclone.design.diameter_proportions,
        hydrocyclone.design.length_proportions,
        hydrocyclone.design.cone_angle)

    hydrocyclone.calculate_from_flow_rate(properties)

    # NOTE Здесь нам нужна только полная эффективность и у нас уже есть
    # NOTE calculate_total_efficiency, зачем считать и возвращать в _
    # NOTE приведённую эффективность?
    _, total_efficiency = _compute_efficiencies(hydrocyclone, size_dist)

    return total_efficiency - efficiency_target


# ---------------------------------------------------------------------------
# Public inverse problem functions
# ---------------------------------------------------------------------------

def find_Dc_by_cut_size(
        cut_size_target: float,
        hydrocyclone: BaseHydrocyclone,
        properties: PhysicalProperties,
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
        args=(cut_size_target, hydrocyclone, properties),
    )[0]

    return Dc_solution


def find_Dc_by_efficiency(
        efficiency_target: float,
        hydrocyclone: BaseHydrocyclone,
        properties: PhysicalProperties,
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
        args=(efficiency_target, hydrocyclone, properties, size_dist),
    )[0]

    return Dc_solution


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from pytroleum.plant.solid_cyclone.geometry import (
        RIETEMA_DIAMETER_PROPORTIONS,
        RIETEMA_LENGTH_PROPORTIONS,
        RIETEMA_CONE_ANGLE,
    )
    from pytroleum.plant.solid_cyclone.inputs import PhysicalProperties
    from pytroleum.plant.solid_cyclone.models import RietemaHydrocyclone

    properties = PhysicalProperties(solid_density=1500)

    conditions = OperationConditions(
        feed_volumetric_concentration=0.00033,
        mode='Q',
        feed_volumetric_flow_rate=12.0 / (1000 * 60),
    )
    size_dist = SizeDistribution(
        # type: ignore[call-overload]
        particle_diameters=np.linspace(1e-6, 200e-6, 500),
        k=10.9918e-6,
        n=0.9187,
    )

    rietema_hydrocyclone = RietemaHydrocyclone(
        '', CycloneDesign(10e-3, RIETEMA_DIAMETER_PROPORTIONS,
                          RIETEMA_LENGTH_PROPORTIONS, RIETEMA_CONE_ANGLE),
        conditions)

    # Task 1: find Dc for target d50'
    cut_size_target = 5e-6
    print("TASK 1: FIND Dc FOR TARGET REDUCED CUT SIZE d50'")
    _minor_divider()
    find_Dc_by_cut_size(
        cut_size_target=cut_size_target,
        hydrocyclone=rietema_hydrocyclone,
        properties=properties,
    )
    rietema_hydrocyclone.design.summary()
    et1_reduced, et1_total = _compute_efficiencies(
        rietema_hydrocyclone, size_dist)
    print(
        f"Volumetric flow rate  Q = "
        f"{rietema_hydrocyclone.conditions.feed_volumetric_flow_rate*6e4:.3f} L/min")
    print(
        f"Pressure drop ΔP = "
        f"{rietema_hydrocyclone.conditions.pressure_drop/1e3:.2f} kPa")
    print(f"Water flow ratio Rw = {rietema_hydrocyclone.water_flow_ratio:.4f}")
    print(f"Reduced cut size d50'= {rietema_hydrocyclone.reduced_cut_size*1e6:.2f} µm"
          f" (target: {cut_size_target*1e6:.2f} µm)")
    print(f"Reduced total efficiency E_T'= {et1_reduced*100:.1f} %")
    print(f"Total efficiency E_T = {et1_total*100:.2f} %")
    reduced_cut_size_1 = rietema_hydrocyclone.reduced_cut_size

    print()
    _major_divider()
    print()

    # Task 2: find Dc for target E_T
    efficiency_target = 0.9
    print("TASK 2: FIND Dc FOR TARGET TOTAL EFFICIENCY E_T")
    _minor_divider()
    find_Dc_by_efficiency(
        efficiency_target=efficiency_target,
        hydrocyclone=rietema_hydrocyclone,
        properties=properties,
        size_dist=size_dist,
    )
    rietema_hydrocyclone.design.summary()
    et2_reduced, et2_total = _compute_efficiencies(
        rietema_hydrocyclone, size_dist)
    print(
        f"Volumetric flow rate Q = "
        f"{rietema_hydrocyclone.conditions.feed_volumetric_flow_rate*6e4:.3f} L/min")
    print(
        f"Pressure drop ΔP = "
        f"{rietema_hydrocyclone.conditions.pressure_drop/1e3:.2f} kPa")
    print(f"Water flow ratio Rw = {rietema_hydrocyclone.water_flow_ratio:.4f}")
    print(
        f"Reduced cut size d50' = "
        f"{rietema_hydrocyclone.reduced_cut_size*1e6:.2f} µm")
    print(f"Reduced total efficiency E_T'= {et2_reduced*100:.1f} %")
    print(f"Total efficiency E_T  = {et2_total*100:.2f} %"
          f"  (target: {efficiency_target*100:.2f} %)")

    print()
    _major_divider()
    print()

    # Verification
    print("VERIFICATION OF TASK 1 (d50')")
    _minor_divider()
    rel_err1 = abs(reduced_cut_size_1 - cut_size_target) / cut_size_target
    print(f"d50' (found)  = {reduced_cut_size_1*1e6:.4f} µm")
    print(f"d50' (target) = {cut_size_target*1e6:.4f} µm")
    print(f"Relative error: {rel_err1:.2e}")
    if rel_err1 <= TOL_RELATIVE:
        print("Task 1 converged")
    else:
        print("Task 1 did NOT converge — error exceeds tolerance")

    print()
    _major_divider()
    print()

    print("VERIFICATION OF TASK 2 (E_T)")
    _minor_divider()
    rel_err2 = abs(et2_total - efficiency_target) / efficiency_target
    print(f"E_T (found)  = {et2_total*100:.2f} %")
    print(f"E_T (target) = {efficiency_target*100:.2f} %")
    print(f"Relative error: {rel_err2:.2e}")
    if rel_err2 <= TOL_RELATIVE:
        print("Task 2 converged")
    else:
        print("Task 2 did NOT converge — error exceeds tolerance")

    print()
    _major_divider()
    print()
