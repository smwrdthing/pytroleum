"""
Обратная задача гидроциклона: поиск диаметра корпуса Dc
при заданном расходе Q, свойствах фаз и концентрации.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import fsolve

from pytroleum.plant.solid_cyclone.geometry import (
    build_from_ratios,
    CYCLONE_CONE_ANGLE_MIN,
    CYCLONE_CONE_ANGLE_MAX,
)
from pytroleum.plant.solid_cyclone.inputs import (
    PhysicalProperties,
    OperatingConditions,
    SizeDistribution,
)
from pytroleum.plant.solid_cyclone.models import BaseHydrocyclone
from pytroleum.plant.solid_cyclone.efficiency import (
    calculate_reduced_grade_efficiency,
    calculate_reduced_total_efficiency,
    calculate_total_efficiency,
)


# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

TOL_RELATIVE = 1e-4  # допустимое расхождение прямой и обратной задач
_V_IN_INITIAL = 9.0   # м/с — типовая скорость для начального приближения

# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def _validate_cone_angle(ratios: dict[str, float]) -> None:
    # NOTE у нас есть dataclass для хранения инфрмации о геометрии гидроциклона,
    # NOTE зачем тогда работать со словарями, которые делают то же самое?
    # NOTE можно просто передавать объект класса, который хранит геометрические параметры
    """Проверка угла конуса перед решением обратной задачи."""
    angle = ratios['angle']
    if not (CYCLONE_CONE_ANGLE_MIN <= angle <= CYCLONE_CONE_ANGLE_MAX):
        raise ValueError(
            f"angle = {angle:.1f}° вне допустимого диапазона "
            f"[{CYCLONE_CONE_ANGLE_MIN}°, {CYCLONE_CONE_ANGLE_MAX}°]."
        )


def _initial_Dc(Q: float, Di_Dc_ratio: float) -> float:
    # NOTE здесь тоже можно просто передавать объект класса, который хранит
    # NOTE геометрические параметры, в нём уже есть информация о Di_Dc_ratio
    """
    Начальное приближение Dc для fsolve.

    Выводится из условия v_in = Q / (π·Di²/4) при типовой скорости _V_IN_INITIAL:
      Di0 = sqrt(4·Q / (v_in·π)),  Dc0 = Di0 / (Di/Dc)
    """
    Di0 = np.sqrt(4.0 * Q / (_V_IN_INITIAL * np.pi))
    return Di0 / Di_Dc_ratio


def _compute_efficiencies(
        results: dict[str, float],  # NOTE см. заметку по словарю в models
        size_dist: SizeDistribution,
) -> tuple[NDArray | np.floating, NDArray | np.floating]:
    """Расчёт приведённой E_T' и полной E_T эффективностей."""
    reduced_grade_efficiency = calculate_reduced_grade_efficiency(
        size_dist.particle_diameters,
        results['reduced_cut_size'],
        'plitt',
        results['m'],
        results['alpha'],
    )
    reduced_total_efficiency = calculate_reduced_total_efficiency(
        size_dist.particle_diameters, size_dist.k, size_dist.n,
        reduced_grade_efficiency)
    total_efficiency = calculate_total_efficiency(
        reduced_total_efficiency, results['water_flow_ratio'])

    # NOTE насколько часто нам нужно считать сразу обе эффективности?
    # NOTE функции для их расчёта по отдельности уже есть в отдельном модуле,
    # NOTE смысл в таком оборачивании есть только если нам нужно очень часто
    # NOTE считать сразу обе эффективности
    return reduced_total_efficiency, total_efficiency


def _residual_cut_size(
        Dc: float,
        cut_size_target: float,
        conditions: OperatingConditions,
        ratios: dict[str, float],
        hydrocyclone_cls: type[BaseHydrocyclone],
        properties: PhysicalProperties,
) -> float:
    """Невязка задачи 1: f(Dc) = d₅₀'(Dc, Q) - d₅₀'_target."""
    hydrocyclone = build_from_ratios(Dc, ratios, hydrocyclone_cls)
    results = hydrocyclone.calculate_from_flow_rate(
        properties,
        conditions.feed_volumetric_flow_rate,
        conditions.feed_volumetric_concentration,
    )
    return results['reduced_cut_size'] - cut_size_target


def _residual_efficiency(
        Dc: float,
        efficiency_target: float,
        conditions: OperatingConditions,

        # NOTE это уже лежит в датаклассе с геометрией
        ratios: dict[str, float],

        # NOTE зачем мы делаем функцию, которой нужно передавать класс?
        hydrocyclone_cls: type[BaseHydrocyclone],

        properties: PhysicalProperties,
        size_dist: SizeDistribution,
) -> NDArray | np.floating:
    # NOTE половина передаваемой информации в сигнатуре вызова этой функции уже содержится
    # NOTE в описанных датаклассах (для геометрии, рабочих параметров) - почему не
    # NOTE передавть объекты этих датаклассов и не работать с ними?
    """Невязка задачи 2: f(Dc) = E_T(Dc, Q) - E_T_target."""

    hydrocyclone = build_from_ratios(Dc, ratios, hydrocyclone_cls)
    # NOTE такая функция может работать с уже собранным гидроциклоном, нужно только
    # NOTE предусмотреть возможность переназначить размеры

    results = hydrocyclone.calculate_from_flow_rate(
        properties,
        conditions.feed_volumetric_flow_rate,
        conditions.feed_volumetric_concentration,
    )
    _, total_efficiency = _compute_efficiencies(results, size_dist)
    return total_efficiency - efficiency_target


def _assemble_output(
        hydrocyclone: BaseHydrocyclone,
        results: dict[str, float],  # NOTE см. заметку по словарю в models
        size_dist: SizeDistribution,
) -> dict:
    """Сборка выходного словаря: геометрия + гидравлика + эффективность."""
    from pytroleum.plant.solid_cyclone.geometry import (
        HydrocycloneDiameters, HydrocycloneLengths, IDX_ANGLE,
    )
    reduced_total_efficiency, total_efficiency = _compute_efficiencies(
        results, size_dist)

    d = hydrocyclone.design.diameters
    le = hydrocyclone.design.lengths
    p = hydrocyclone.design.proportions

    geometry = {
        'Dc': d[HydrocycloneDiameters.C],
        'Di': d[HydrocycloneDiameters.I],
        'Do': d[HydrocycloneDiameters.O],
        'Du': d[HydrocycloneDiameters.U],
        'L': le[HydrocycloneLengths.T],
        'Lc': le[HydrocycloneLengths.C],
        'vortex_finder_length': le[HydrocycloneLengths.V],
        'angle': p[IDX_ANGLE],
    }

    return {
        **geometry,
        **results,
        'reduced_total_efficiency': reduced_total_efficiency,
        'total_efficiency': total_efficiency,
    }  # NOTE зачем нам словарь, в котором лежит всё сразу?


# ---------------------------------------------------------------------------
# Публичные функции обратной задачи
# ---------------------------------------------------------------------------

def find_Dc_by_cut_size(
        cut_size_target: float,
        conditions: OperatingConditions,

        # NOTE это уже лежит в датаклассе с геометрией
        ratios: dict[str, float],

        # NOTE зачем мы делаем функцию, которой нужно передавать класс?
        hydrocyclone_cls: type[BaseHydrocyclone],

        properties: PhysicalProperties,
        size_dist: SizeDistribution,
        Dc0: float | None = None,
) -> dict:
    """
    Задача 1. Найти Dc, при котором d₅₀'(Dc, Q) = cut_size_target.

    Решается: f(Dc) = d₅₀'(Dc, Q) - cut_size_target = 0
    """
    _validate_cone_angle(ratios)

    if Dc0 is None:
        Dc0 = _initial_Dc(
            conditions.feed_volumetric_flow_rate, ratios['Di/Dc'])

    Dc_solution = fsolve(
        _residual_cut_size, x0=Dc0,
        args=(cut_size_target, conditions, ratios,
              hydrocyclone_cls, properties),
    )[0]

    hydrocyclone = build_from_ratios(Dc_solution, ratios, hydrocyclone_cls)
    results = hydrocyclone.calculate_from_flow_rate(
        properties,
        conditions.feed_volumetric_flow_rate,
        conditions.feed_volumetric_concentration,
    )

    return _assemble_output(hydrocyclone, results, size_dist)


def find_Dc_by_efficiency(
        efficiency_target: float,
        conditions: OperatingConditions,

        # NOTE это уже лежит в датаклассе с геометрией
        ratios: dict[str, float],

        # NOTE зачем мы делаем функцию, которой нужно передавать класс?
        hydrocyclone_cls: type[BaseHydrocyclone],

        properties: PhysicalProperties,
        size_dist: SizeDistribution,
        Dc0: float | None = None,
) -> dict:
    """
    Задача 2. Найти Dc, при котором E_T(Dc, Q) = efficiency_target.

    Решается: f(Dc) = E_T(Dc, Q) - efficiency_target = 0
    """
    _validate_cone_angle(ratios)

    if Dc0 is None:
        Dc0 = _initial_Dc(
            conditions.feed_volumetric_flow_rate, ratios['Di/Dc'])

    Dc_solution = fsolve(
        _residual_efficiency, x0=Dc0,
        args=(efficiency_target, conditions, ratios,
              hydrocyclone_cls, properties, size_dist),
    )[0]

    hydrocyclone = build_from_ratios(Dc_solution, ratios, hydrocyclone_cls)
    results = hydrocyclone.calculate_from_flow_rate(
        properties,
        conditions.feed_volumetric_flow_rate,
        conditions.feed_volumetric_concentration,
    )

    return _assemble_output(hydrocyclone, results, size_dist)


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from pytroleum.plant.solid_cyclone.inputs import PhysicalProperties
    from pytroleum.plant.solid_cyclone.models import RietemaHydrocyclone

    properties = PhysicalProperties(solid_density=1500)

    # NOTE этот словарь дублирует информацию из массива в geometry
    ratios_rietema = {
        'Di/Dc': 0.20,
        'Do/Dc': 0.25,
        'Du/Dc': 0.15,
        'L/Dc': 4.5,
        'l/Dc': 0.40,
        'angle': 15.0,
    }

    conditions = OperatingConditions(
        feed_volumetric_concentration=0.00033,
        mode='Q',
        feed_volumetric_flow_rate=12.0 / (1000 * 60),
    )
    size_dist = SizeDistribution(
        particle_diameters=np.linspace(1e-6, 200e-6, 500),
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
        ratios=ratios_rietema,
        hydrocyclone_cls=RietemaHydrocyclone,
        properties=properties,
        size_dist=size_dist,
    )
    build_from_ratios(res1['Dc'], ratios_rietema,
                      RietemaHydrocyclone).design.summary()
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
        ratios=ratios_rietema,
        hydrocyclone_cls=RietemaHydrocyclone,
        properties=properties,
        size_dist=size_dist,
    )
    build_from_ratios(res2['Dc'], ratios_rietema,
                      RietemaHydrocyclone).design.summary()
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
    hc_check1 = build_from_ratios(
        res1['Dc'], ratios_rietema, RietemaHydrocyclone)
    check1 = hc_check1.calculate_from_flow_rate(
        properties,
        conditions.feed_volumetric_flow_rate,
        conditions.feed_volumetric_concentration,
    )

    d50_inverse = res1['reduced_cut_size']
    d50_direct = check1['reduced_cut_size']
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
    hc_check2 = build_from_ratios(
        res2['Dc'], ratios_rietema, RietemaHydrocyclone)
    check2 = hc_check2.calculate_from_flow_rate(
        properties,
        conditions.feed_volumetric_flow_rate,
        conditions.feed_volumetric_concentration,
    )
    _, et_direct = _compute_efficiencies(check2, size_dist)

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
