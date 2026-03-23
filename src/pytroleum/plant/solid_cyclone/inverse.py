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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOL_RELATIVE = 1e-4  # allowable discrepancy between forward and inverse problems
_V_IN_INITIAL = 9.0   # m/s — typical velocity for initial approximation

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _validate_cone_angle(cone_angle: float) -> None:
    # NOTE у нас есть dataclass для хранения информации о геометрии гидроциклона,
    # NOTE зачем тогда работать со словарями, которые делают то же самое?
    # NOTE можно просто передавать объект класса, который хранит геометрические параметры
    """Validate cone angle before solving the inverse problem."""
    if not (CYCLONE_CONE_ANGLE_MIN <= cone_angle <= CYCLONE_CONE_ANGLE_MAX):
        raise ValueError(
            f"angle = {cone_angle:.1f}° out of valid range "
            f"[{CYCLONE_CONE_ANGLE_MIN}°, {CYCLONE_CONE_ANGLE_MAX}°]."
        )


def _initial_Dc(Q: float, diameter_proportions: NDArray) -> float:
    # NOTE здесь тоже можно просто передавать объект класса, который хранит
    # NOTE геометрические параметры, в нём уже есть информация о Di_Dc_ratio
    """
    Initial Dc approximation for fsolve.

    Derived from v_in = Q / (pi*Di^2/4) at the typical velocity _V_IN_INITIAL:
      Di0 = sqrt(4*Q / (v_in*pi)),  Dc0 = Di0 / (Di/Dc)
    """
    Di_Dc_ratio = diameter_proportions[HydrocycloneDiameters.I]
    Di0 = np.sqrt(4.0 * Q / (_V_IN_INITIAL * np.pi))
    return Di0 / Di_Dc_ratio


def _compute_efficiencies(
        hydrocyclone: BaseHydrocyclone,
        size_dist: SizeDistribution,
) -> tuple[NDArray | np.floating, NDArray | np.floating]:
    """Calculate reduced E_T' and total E_T efficiencies."""
    reduced_grade_efficiency = calculate_reduced_grade_efficiency(
        size_dist.particle_diameters,
        hydrocyclone.reduced_cut_size,
        'plitt',
        hydrocyclone.m,
        hydrocyclone.alpha,
    )
    reduced_total_efficiency = calculate_reduced_total_efficiency(
        size_dist.particle_diameters, reduced_grade_efficiency,
        size_dist.k, size_dist.n)
    total_efficiency = calculate_total_efficiency(
        reduced_total_efficiency, hydrocyclone.water_flow_ratio)

    # NOTE насколько часто нам нужно считать сразу обе эффективности?
    # NOTE функции для их расчёта по отдельности уже есть в отдельном модуле,
    # NOTE смысл в таком оборачивании есть только если нам нужно очень часто
    # NOTE считать сразу обе эффективности
    return reduced_total_efficiency, total_efficiency


def _residual_cut_size(
        Dc: float,
        cut_size_target: float,
        conditions: OperationConditions,
        diameter_proportions: NDArray,
        length_proportions: NDArray,
        cone_angle: float,
        hydrocyclone_cls: type[BaseHydrocyclone],
        properties: PhysicalProperties,
) -> float:
    """Residual for problem 1: f(Dc) = d50'(Dc, Q) - d50'_target."""
    hydrocyclone = hydrocyclone_cls(
        '', CycloneDesign(Dc, diameter_proportions, length_proportions, cone_angle))
    hydrocyclone.calculate_from_flow_rate(
        properties,
        conditions.feed_volumetric_flow_rate,
        conditions.feed_volumetric_concentration,
    )
    return hydrocyclone.reduced_cut_size - cut_size_target


def _residual_efficiency(
        Dc: float,
        efficiency_target: float,
        conditions: OperationConditions,

        # NOTE это уже лежит в датаклассе с геометрией
        diameter_proportions: NDArray,
        length_proportions: NDArray,
        cone_angle: float,

        # NOTE зачем мы делаем функцию, которой нужно передавать класс?
        hydrocyclone_cls: type[BaseHydrocyclone],

        properties: PhysicalProperties,
        size_dist: SizeDistribution,
) -> NDArray | np.floating:
    # NOTE половина передаваемой информации в сигнатуре вызова этой функции уже содержится
    # NOTE в описанных датаклассах (для геометрии, рабочих параметров) - почему не
    # NOTE передавать объекты этих датаклассов и не работать с ними?
    """Residual for problem 2: f(Dc) = E_T(Dc, Q) - E_T_target."""

    hydrocyclone = hydrocyclone_cls(
        '', CycloneDesign(Dc, diameter_proportions, length_proportions, cone_angle))
    # NOTE такая функция может работать с уже собранным гидроциклоном, нужно только
    # NOTE предусмотреть возможность переназначить размеры

    hydrocyclone.calculate_from_flow_rate(
        properties,
        conditions.feed_volumetric_flow_rate,
        conditions.feed_volumetric_concentration,
    )
    _, total_efficiency = _compute_efficiencies(hydrocyclone, size_dist)
    return total_efficiency - efficiency_target


def _assemble_output(
        hydrocyclone: BaseHydrocyclone,
        size_dist: SizeDistribution,
) -> dict:
    """Assemble output dict: geometry + hydraulics + efficiency."""
    from pytroleum.plant.solid_cyclone.geometry import (
        HydrocycloneDiameters, HydrocycloneLengths,
    )
    reduced_total_efficiency, total_efficiency = _compute_efficiencies(
        hydrocyclone, size_dist)

    d = hydrocyclone.design.diameters
    le = hydrocyclone.design.lengths

    return {
        'Dc': d[HydrocycloneDiameters.C],
        'Di': d[HydrocycloneDiameters.I],
        'Do': d[HydrocycloneDiameters.O],
        'Du': d[HydrocycloneDiameters.U],
        'L': le[HydrocycloneLengths.T],
        'Lc': le[HydrocycloneLengths.C],
        'vortex_finder_length': le[HydrocycloneLengths.V],
        'angle': hydrocyclone.design.cone_angle,
        'feed_volumetric_flow_rate': hydrocyclone.feed_volumetric_flow_rate,
        'pressure_drop': hydrocyclone.pressure_drop,
        'water_flow_ratio': hydrocyclone.water_flow_ratio,
        'Re': hydrocyclone.Re,
        'Eu': hydrocyclone.Eu,
        'reduced_cut_size': hydrocyclone.reduced_cut_size,
        'alpha': hydrocyclone.alpha,
        'm': hydrocyclone.m,
        'reduced_total_efficiency': reduced_total_efficiency,
        'total_efficiency': total_efficiency,
    }  # NOTE зачем нам словарь, в котором лежит всё сразу?


# ---------------------------------------------------------------------------
# Public inverse problem functions
# ---------------------------------------------------------------------------

def find_Dc_by_cut_size(
        cut_size_target: float,
        conditions: OperationConditions,

        # NOTE это уже лежит в датаклассе с геометрией
        diameter_proportions: NDArray,
        length_proportions: NDArray,
        cone_angle: float,

        # NOTE зачем мы делаем функцию, которой нужно передавать класс?
        hydrocyclone_cls: type[BaseHydrocyclone],

        properties: PhysicalProperties,
        size_dist: SizeDistribution,
        Dc0: float | None = None,
) -> dict:
    """
    Problem 1. Find Dc such that d50'(Dc, Q) = cut_size_target.

    Solves: f(Dc) = d50'(Dc, Q) - cut_size_target = 0
    """
    _validate_cone_angle(cone_angle)

    if Dc0 is None:
        Dc0 = _initial_Dc(conditions.feed_volumetric_flow_rate,
                          diameter_proportions)

    Dc_solution = fsolve(
        _residual_cut_size, x0=Dc0,
        args=(cut_size_target, conditions, diameter_proportions,
              length_proportions, cone_angle, hydrocyclone_cls, properties),
    )[0]

    hydrocyclone = hydrocyclone_cls(
        '', CycloneDesign(Dc_solution,
                          diameter_proportions,
                          length_proportions,
                          cone_angle))
    hydrocyclone.calculate_from_flow_rate(
        properties,
        conditions.feed_volumetric_flow_rate,
        conditions.feed_volumetric_concentration,
    )

    return _assemble_output(hydrocyclone, size_dist)


def find_Dc_by_efficiency(
        efficiency_target: float,
        conditions: OperationConditions,

        # NOTE это уже лежит в датаклассе с геометрией
        diameter_proportions: NDArray,
        length_proportions: NDArray,
        cone_angle: float,

        # NOTE зачем мы делаем функцию, которой нужно передавать класс?
        hydrocyclone_cls: type[BaseHydrocyclone],

        properties: PhysicalProperties,
        size_dist: SizeDistribution,
        Dc0: float | None = None,
) -> dict:
    """
    Problem 2. Find Dc such that E_T(Dc, Q) = efficiency_target.

    Solves: f(Dc) = E_T(Dc, Q) - efficiency_target = 0
    """
    _validate_cone_angle(cone_angle)

    if Dc0 is None:
        Dc0 = _initial_Dc(conditions.feed_volumetric_flow_rate,
                          diameter_proportions)

    Dc_solution = fsolve(
        _residual_efficiency, x0=Dc0,
        args=(efficiency_target, conditions, diameter_proportions,
              length_proportions, cone_angle, hydrocyclone_cls, properties, size_dist),
    )[0]

    hydrocyclone = hydrocyclone_cls(
        '', CycloneDesign(Dc_solution,
                          diameter_proportions,
                          length_proportions,
                          cone_angle))
    hydrocyclone.calculate_from_flow_rate(
        properties,
        conditions.feed_volumetric_flow_rate,
        conditions.feed_volumetric_concentration,
    )

    return _assemble_output(hydrocyclone, size_dist)


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
        particle_diameters=np.linspace(1e-6, 200e-6, 500),  # type: ignore[call-overload]
        k=10.9918e-6,
        n=0.9187,
    )

    # Task 1: find Dc for target d50'
    cut_size_target = 5e-6
    print("TASK 1: FIND Dc FOR TARGET REDUCED CUT SIZE d50'")
    print("-" * 60)
    res1 = find_Dc_by_cut_size(
        cut_size_target=cut_size_target,
        conditions=conditions,
        diameter_proportions=RIETEMA_DIAMETER_PROPORTIONS,
        length_proportions=RIETEMA_LENGTH_PROPORTIONS,
        cone_angle=RIETEMA_CONE_ANGLE,
        hydrocyclone_cls=RietemaHydrocyclone,
        properties=properties,
        size_dist=size_dist,
    )
    RietemaHydrocyclone(
        'Rietema',
        CycloneDesign(res1['Dc'], RIETEMA_DIAMETER_PROPORTIONS,
                      RIETEMA_LENGTH_PROPORTIONS, RIETEMA_CONE_ANGLE),
    ).design.summary()
    print(
        f"Volumetric flow rate  Q = {res1['feed_volumetric_flow_rate']*6e4:.3f} L/min")
    print(f"Pressure drop ΔP = {res1['pressure_drop']/1e3:.2f} kPa")
    print(f"Water flow ratio Rw = {res1['water_flow_ratio']:.4f}")
    print(f"Reduced cut size d50'= {res1['reduced_cut_size']*1e6:.2f} µm"
          f"  (target: {cut_size_target*1e6:.2f} µm)")
    print(
        f"Reduced total efficiency E_T'= {res1['reduced_total_efficiency']*100:.1f} %")
    print(f"Total efficiency E_T = {res1['total_efficiency']*100:.2f} %")

    print("\n" + "=" * 60 + "\n")

    # Task 2: find Dc for target E_T
    efficiency_target = 0.9
    print("TASK 2: FIND Dc FOR TARGET TOTAL EFFICIENCY E_T")
    print("-" * 60)
    res2 = find_Dc_by_efficiency(
        efficiency_target=efficiency_target,
        conditions=conditions,
        diameter_proportions=RIETEMA_DIAMETER_PROPORTIONS,
        length_proportions=RIETEMA_LENGTH_PROPORTIONS,
        cone_angle=RIETEMA_CONE_ANGLE,
        hydrocyclone_cls=RietemaHydrocyclone,
        properties=properties,
        size_dist=size_dist,
    )
    RietemaHydrocyclone(
        'Rietema',
        CycloneDesign(res2['Dc'], RIETEMA_DIAMETER_PROPORTIONS,
                      RIETEMA_LENGTH_PROPORTIONS, RIETEMA_CONE_ANGLE),
    ).design.summary()
    print(
        f"Volumetric flow rate Q = {res2['feed_volumetric_flow_rate']*6e4:.3f} L/min")
    print(f"Pressure drop ΔP = {res2['pressure_drop']/1e3:.2f} kPa")
    print(f"Water flow ratio Rw = {res2['water_flow_ratio']:.4f}")
    print(f"Reduced cut size d50' = {res2['reduced_cut_size']*1e6:.2f} µm")
    print(
        f"Reduced total efficiency E_T'= {res2['reduced_total_efficiency']*100:.1f} %")
    print(f"Total efficiency E_T  = {res2['total_efficiency']*100:.2f} %"
          f"  (target: {efficiency_target*100:.2f} %)")

    print("\n" + "=" * 60 + "\n")

    # Verification
    print("VERIFICATION OF TASK 1 (d50')")
    print("-" * 60)
    hc_check1 = RietemaHydrocyclone(
        'Rietema',
        CycloneDesign(res1['Dc'], RIETEMA_DIAMETER_PROPORTIONS,
                      RIETEMA_LENGTH_PROPORTIONS, RIETEMA_CONE_ANGLE),
    )
    hc_check1.calculate_from_flow_rate(
        properties,
        conditions.feed_volumetric_flow_rate,
        conditions.feed_volumetric_concentration,
    )

    d50_inverse = res1['reduced_cut_size']
    d50_direct = hc_check1.reduced_cut_size
    rel_err1 = abs(d50_direct - d50_inverse) / d50_inverse
    print(f"d50' (inverse) = {d50_inverse*1e6:.4f} µm")
    print(f"d50' (direct)  = {d50_direct*1e6:.4f} µm")
    print(f"Relative error: {rel_err1:.2e}")
    if rel_err1 <= TOL_RELATIVE:
        print("Task 1 converged")
    else:
        print("Task 1 did NOT converge — error exceeds tolerance")

    print("\n" + "=" * 60 + "\n")

    print("VERIFICATION OF TASK 2 (E_T)")
    print("-" * 60)
    hc_check2 = RietemaHydrocyclone(
        'Rietema',
        CycloneDesign(res2['Dc'], RIETEMA_DIAMETER_PROPORTIONS,
                      RIETEMA_LENGTH_PROPORTIONS, RIETEMA_CONE_ANGLE),
    )
    hc_check2.calculate_from_flow_rate(
        properties,
        conditions.feed_volumetric_flow_rate,
        conditions.feed_volumetric_concentration,
    )
    _, et_direct = _compute_efficiencies(hc_check2, size_dist)

    et_inverse = res2['total_efficiency']
    rel_err2 = abs(et_direct - et_inverse) / et_inverse
    print(f"E_T (inverse) = {et_inverse*100:.2f} %")
    print(f"E_T (direct)  = {et_direct*100:.2f} %")
    print(f"Relative error: {rel_err2:.2e}")
    if rel_err2 <= TOL_RELATIVE:
        print("Task 2 converged")
    else:
        print("Task 2 did NOT converge — error exceeds tolerance")

    print("\n" + "=" * 60 + "\n")
