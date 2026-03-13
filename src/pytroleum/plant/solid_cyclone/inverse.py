"""
Обратная задача гидроциклона: поиск диаметра корпуса Dc
при заданном расходе Q, свойствах фаз и концентрации.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import fsolve

from pytroleum.plant.solid_cyclone.geometry import (
    build_geometry,
    CYCLONE_CONE_ANGLE_MIN,
    CYCLONE_CONE_ANGLE_MAX,
)
from pytroleum.plant.solid_cyclone.properties import PhysicalProperties
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
    """Проверка угла конуса перед решением обратной задачи."""
    angle = ratios['angle']
    if not (CYCLONE_CONE_ANGLE_MIN <= angle <= CYCLONE_CONE_ANGLE_MAX):
        raise ValueError(
            f"angle = {angle:.1f}° вне допустимого диапазона "
            f"[{CYCLONE_CONE_ANGLE_MIN}°, {CYCLONE_CONE_ANGLE_MAX}°]."
        )


def _build_from_ratios(
        Dc: float,
        ratios: dict[str, float],
        hydrocyclone_cls: type[BaseHydrocyclone],
) -> BaseHydrocyclone:
    """Создание экземпляра гидроциклона из диаметра корпуса и словаря пропорций."""
    geometry = build_geometry(
        hydrocyclone_diameter=Dc,
        feed_inlet_diameter=Dc * ratios['Di/Dc'],
        overflow_diameter=Dc * ratios['Do/Dc'],
        underflow_diameter=Dc * ratios['Du/Dc'],
        hydrocyclone_length=Dc * ratios['L/Dc'],
        vortex_finder_length=Dc * ratios['l/Dc'],
        angle=ratios['angle'],
    )
    return hydrocyclone_cls('', geometry)


def _initial_Dc(Q: float, Di_Dc_ratio: float) -> float:
    """
    Начальное приближение Dc для fsolve.

    Выводится из условия v_in = Q / (π·Di²/4) при типовой скорости _V_IN_INITIAL:
      Di0 = sqrt(4·Q / (v_in·π)),  Dc0 = Di0 / (Di/Dc)
    """
    Di0 = np.sqrt(4.0 * Q / (_V_IN_INITIAL * np.pi))
    return Di0 / Di_Dc_ratio


def _compute_efficiencies(
        results: dict[str, float],
        particle_diameters: NDArray,
        k: float,
        n: float,
) -> tuple[float, float]:
    """Расчёт приведённой E_T' и полной E_T эффективностей."""
    reduced_grade_efficiency = calculate_reduced_grade_efficiency(
        particle_diameters,
        results['reduced_cut_size'],
        'plitt',
        results['m'],
        results['alpha'],
    )
    reduced_total_efficiency = calculate_reduced_total_efficiency(
        particle_diameters, k, n, reduced_grade_efficiency)
    total_efficiency = calculate_total_efficiency(
        reduced_total_efficiency, results['water_flow_ratio'])
    return reduced_total_efficiency, total_efficiency


def _residual_cut_size(
        Dc: float,
        cut_size_target: float,
        feed_volumetric_flow_rate: float,
        ratios: dict[str, float],
        hydrocyclone_cls: type[BaseHydrocyclone],
        properties: PhysicalProperties,
        feed_volumetric_concentration: float,
) -> float:
    """Невязка задачи 1: f(Dc) = d₅₀'(Dc, Q) - d₅₀'_target."""
    # NOTE Возможно здесь и в других местах получится сделать dependency injection
    # NOTE (то есть передавать готовый, собранный гидроцкилон, а не пересоздавать его)
    # NOTE тогда в функции надо будет менять размеры у готового гидроциклона
    # NOTE
    # NOTE Такие вещи имеют смысл, когда инициализация объектов дорогая и занмимет много
    # NOTE времени
    hydrocyclone = _build_from_ratios(Dc, ratios, hydrocyclone_cls)
    results = hydrocyclone.calculate_from_flow_rate(
        properties, feed_volumetric_flow_rate, feed_volumetric_concentration)
    return results['reduced_cut_size'] - cut_size_target


def _residual_efficiency(
        Dc: float,
        efficiency_target: float,
        feed_volumetric_flow_rate: float,
        ratios: dict[str, float],
        hydrocyclone_cls: type[BaseHydrocyclone],
        properties: PhysicalProperties,
        feed_volumetric_concentration: float,
        particle_diameters: NDArray,
        k: float,
        n: float,
) -> float:
    """Невязка задачи 2: f(Dc) = E_T(Dc, Q) - E_T_target."""
    hydrocyclone = _build_from_ratios(Dc, ratios, hydrocyclone_cls)
    results = hydrocyclone.calculate_from_flow_rate(
        properties, feed_volumetric_flow_rate, feed_volumetric_concentration)
    _, total_efficiency = _compute_efficiencies(
        results, particle_diameters, k, n)
    return total_efficiency - efficiency_target


def _assemble_output(
        Dc_solution: float,
        ratios: dict[str, float],
        results: dict[str, float],
        particle_diameters: NDArray,
        k: float,
        n: float,
) -> dict:
    """Сборка выходного словаря: геометрия + гидравлика + эффективность."""
    reduced_total_efficiency, total_efficiency = _compute_efficiencies(
        results, particle_diameters, k, n)

    Dc = Dc_solution
    Di = Dc * ratios['Di/Dc']
    Do = Dc * ratios['Do/Dc']
    Du = Dc * ratios['Du/Dc']
    total_length = Dc * ratios['L/Dc']
    vortex_finder_length = Dc * ratios['l/Dc']

    angle_rad = np.radians(ratios['angle'])
    Lc = total_length - (Dc - Du) / (2 * np.tan(angle_rad / 2))

    geometry = {
        'Dc': Dc,
        'Di': Di,
        'Do': Do,
        'Du': Du,
        'L': total_length,
        'Lc': Lc,
        'vortex_finder_length': vortex_finder_length,
        'angle': ratios['angle'],
    }

    return {
        **geometry,
        **results,
        'reduced_total_efficiency': reduced_total_efficiency,
        'total_efficiency': total_efficiency,
    }


# ---------------------------------------------------------------------------
# Публичные функции обратной задачи
# ---------------------------------------------------------------------------

def find_Dc_by_cut_size(
        cut_size_target: float,
        feed_volumetric_flow_rate: float,
        ratios: dict[str, float],
        hydrocyclone_cls: type[BaseHydrocyclone],
        properties: PhysicalProperties,
        feed_volumetric_concentration: float,
        particle_diameters: NDArray,
        k: float,
        n: float,
        Dc0: float | None = None,
) -> dict:
    """
    Задача 1. Найти Dc, при котором d₅₀'(Dc, Q) = cut_size_target.

    Решается: f(Dc) = d₅₀'(Dc, Q) - cut_size_target = 0
    """
    _validate_cone_angle(ratios)

    if Dc0 is None:
        Dc0 = _initial_Dc(feed_volumetric_flow_rate, ratios['Di/Dc'])

    Dc_solution = fsolve(
        _residual_cut_size, x0=Dc0,
        args=(cut_size_target, feed_volumetric_flow_rate,
              ratios, hydrocyclone_cls, properties, feed_volumetric_concentration),
    )[0]

    hydrocyclone = _build_from_ratios(Dc_solution, ratios, hydrocyclone_cls)
    results = hydrocyclone.calculate_from_flow_rate(
        properties, feed_volumetric_flow_rate, feed_volumetric_concentration)

    return _assemble_output(Dc_solution, ratios, results, particle_diameters, k, n)


def find_Dc_by_efficiency(
        efficiency_target: float,
        feed_volumetric_flow_rate: float,
        ratios: dict[str, float],
        hydrocyclone_cls: type[BaseHydrocyclone],
        properties: PhysicalProperties,
        feed_volumetric_concentration: float,
        particle_diameters: NDArray,
        k: float,
        n: float,
        Dc0: float | None = None,
) -> dict:
    """
    Задача 2. Найти Dc, при котором E_T(Dc, Q) = efficiency_target.

    Решается: f(Dc) = E_T(Dc, Q) - efficiency_target = 0
    """
    _validate_cone_angle(ratios)

    if Dc0 is None:
        Dc0 = _initial_Dc(feed_volumetric_flow_rate, ratios['Di/Dc'])

    Dc_solution = fsolve(
        _residual_efficiency, x0=Dc0,
        args=(efficiency_target, feed_volumetric_flow_rate,
              ratios, hydrocyclone_cls, properties, feed_volumetric_concentration,
              particle_diameters, k, n),
    )[0]

    hydrocyclone = _build_from_ratios(Dc_solution, ratios, hydrocyclone_cls)
    results = hydrocyclone.calculate_from_flow_rate(
        properties, feed_volumetric_flow_rate, feed_volumetric_concentration)

    return _assemble_output(Dc_solution, ratios, results, particle_diameters, k, n)


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from pytroleum.plant.solid_cyclone.properties import PhysicalProperties
    from pytroleum.plant.solid_cyclone.models import RietemaHydrocyclone

    properties = PhysicalProperties(solid_density=1500)
    feed_volumetric_concentration = 0.00033

    ratios_rietema = {
        'Di/Dc': 0.20,
        'Do/Dc': 0.25,
        'Du/Dc': 0.15,
        'L/Dc': 4.5,
        'l/Dc': 0.40,
        'angle': 15.0,
    }

    Q = 12.0 / (1000 * 60)
    k = 10.9918e-6
    n = 0.9187
    particle_diameters = np.linspace(1e-6, 200e-6, 500)  # от 1 мкм до 200 мкм

    # Задача 1: найти Dc для d₅₀'
    cut_size_target = 6.888e-6
    print("ЗАДАЧА 1: ПОИСК Dc ПО ЦЕЛЕВОМУ ОТСЕЧНОМУ РАЗМЕРУ d₅₀'")
    print("-" * 60)
    res1 = find_Dc_by_cut_size(
        cut_size_target=cut_size_target,
        feed_volumetric_flow_rate=Q,
        ratios=ratios_rietema,
        hydrocyclone_cls=RietemaHydrocyclone,
        properties=properties,
        feed_volumetric_concentration=feed_volumetric_concentration,
        particle_diameters=particle_diameters,
        k=k, n=n,
    )
    print(f"Dc = {res1['Dc']*1e3:.2f} мм")
    print(f"Di = {res1['Di']*1e3:.2f} мм")
    print(f"Do = {res1['Do']*1e3:.2f} мм")
    print(f"Du = {res1['Du']*1e3:.2f} мм")
    print(f"L = {res1['L']*1e3:.2f} мм")
    print(f"Lc = {res1['Lc']*1e3:.2f} мм")
    print(f"l = {res1['vortex_finder_length']*1e3:.2f} мм")
    print(f"θ = {res1['angle']:.1f} °")
    print(f"Q = {res1['feed_volumetric_flow_rate']*6e4:.3f} л/мин")
    print(f"ΔP = {res1['pressure_drop']/1e3:.2f} кПа")
    print(f"Rw = {res1['water_flow_ratio']:.4f}")
    print(
        f"d50'= {res1['reduced_cut_size']*1e6:.2f} мкм  "
        f"(цель: {cut_size_target*1e6:.2f} мкм)")
    print(f"E_T' = {res1['reduced_total_efficiency']*100:.1f} %")
    print(f"E_T = {res1['total_efficiency']*100:.2f} %")

    print("\n" + "=" * 60 + "\n")

    # Задача 2: найти Dc для E_T
    efficiency_target = 0.9
    print("ЗАДАЧА 2: ПОИСК Dc ПО ЦЕЛЕВОЙ ПОЛНОЙ ЭФФЕКТИВНОСТИ E_T")
    print("-" * 60)
    res2 = find_Dc_by_efficiency(
        efficiency_target=efficiency_target,
        feed_volumetric_flow_rate=Q,
        ratios=ratios_rietema,
        hydrocyclone_cls=RietemaHydrocyclone,
        properties=properties,
        feed_volumetric_concentration=feed_volumetric_concentration,
        particle_diameters=particle_diameters,
        k=k, n=n,
    )
    print(f"Dc = {res2['Dc']*1e3:.2f} мм")
    print(f"Di = {res2['Di']*1e3:.2f} мм")
    print(f"Do = {res2['Do']*1e3:.2f} мм")
    print(f"Du = {res2['Du']*1e3:.2f} мм")
    print(f"L = {res2['L']*1e3:.2f} мм")
    print(f"Lc = {res2['Lc']*1e3:.2f} мм")
    print(f"l = {res2['vortex_finder_length']*1e3:.2f} мм")
    print(f"θ = {res2['angle']:.1f} °")
    print(f"Q = {res2['feed_volumetric_flow_rate']*6e4:.3f} л/мин")
    print(f"ΔP = {res2['pressure_drop']/1e3:.2f} кПа")
    print(f"Rw = {res2['water_flow_ratio']:.4f}")
    print(f"d50' = {res2['reduced_cut_size']*1e6:.2f} мкм")
    print(f"E_T' = {res2['reduced_total_efficiency']*100:.1f} %")
    print(
        f"E_T  = {res2['total_efficiency']*100:.2f} %  "
        f"(цель: {efficiency_target*100:.2f} %)")

    print("\n" + "=" * 60 + "\n")

    # Верификация
    print("ВЕРИФИКАЦИЯ ЗАДАЧИ 1 по d50'")
    print("-" * 60)
    hc_check1 = _build_from_ratios(
        res1['Dc'], ratios_rietema, RietemaHydrocyclone)
    check1 = hc_check1.calculate_from_flow_rate(
        properties, Q, feed_volumetric_concentration)

    d50_inverse = res1['reduced_cut_size']
    d50_direct = check1['reduced_cut_size']
    rel_err1 = abs(d50_direct - d50_inverse) / d50_inverse
    print(f"d50' (обратная) = {d50_inverse*1e6:.4f} мкм")
    print(f"d50' (прямая)   = {d50_direct*1e6:.4f} мкм")
    print(f"Относительная погрешность: {rel_err1:.2e}")
    if rel_err1 <= TOL_RELATIVE:
        print("Задача 1 сошлась")
    else:
        print("Задача 1 НЕ сошлась — погрешность превышает допуск")

    print("\n" + "=" * 60 + "\n")

    print("ВЕРИФИКАЦИЯ ЗАДАЧИ 2 по E_T")
    print("-" * 60)
    hc_check2 = _build_from_ratios(
        res2['Dc'], ratios_rietema, RietemaHydrocyclone)
    check2 = hc_check2.calculate_from_flow_rate(
        properties, Q, feed_volumetric_concentration)
    _, et_direct = _compute_efficiencies(check2, particle_diameters, k, n)

    et_inverse = res2['total_efficiency']
    rel_err2 = abs(et_direct - et_inverse) / et_inverse
    print(f"E_T (обратная) = {et_inverse*100:.2f} %")
    print(f"E_T (прямая)   = {et_direct*100:.2f} %")
    print(f"Относительная погрешность: {rel_err2:.2e}")
    if rel_err2 <= TOL_RELATIVE:
        print("Задача 2 сошлась")
    else:
        print("Задача 2 НЕ сошлась — погрешность превышает допуск")

    print("\n" + "=" * 60 + "\n")
