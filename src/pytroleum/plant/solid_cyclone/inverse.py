"""
Обратная задача гидроциклона: поиск диаметра корпуса Dc
при заданном расходе Q, свойствах фаз и концентрации.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import fsolve

from pytroleum.plant.solid_cyclone.geometry import (
    GeometryParameters,
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


_V_IN_INITIAL = 3.0  # м/с — только для начального приближения x0


def _validate_cone_angle(ratios: dict[str, float]) -> None:
    """
    Проверка угла конуса перед решением обратной задачи.

    Проверяется одно условие:
      angle ∈ [CYCLONE_CONE_ANGLE_MIN, CYCLONE_CONE_ANGLE_MAX]
    """
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

    geometry = GeometryParameters.from_named(
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
      Di0 = sqrt(4·Q / (v_in·π))
      Dc0 = Di0 / (Di/Dc)
    """
    Di0 = np.sqrt(4.0 * Q / (_V_IN_INITIAL * np.pi))
    return Di0 / Di_Dc_ratio


def _residual_cut_size(
        Dc: float,
        cut_size_target: float,
        feed_volumetric_flow_rate: float,
        ratios: dict[str, float],
        hydrocyclone_cls: type[BaseHydrocyclone],
        properties: PhysicalProperties,
        feed_volumetric_concentration: float,
) -> float:
    """
    Невязка задачи 1: f(Dc) = d₅₀'(Dc, Q) - d₅₀'_target.
    """
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


def _compute_efficiencies(
        results: dict[str, float],
        particle_diameters: NDArray,
        k: float,
        n: float,
) -> tuple[float, float]:
    """Расчёт E_T' и E_T"""
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
    """
    Невязка задачи 2: f(Dc) = E_T(Dc, Q) - E_T_target.
    """
    hydrocyclone = _build_from_ratios(Dc, ratios, hydrocyclone_cls)
    results = hydrocyclone.calculate_from_flow_rate(
        properties, feed_volumetric_flow_rate, feed_volumetric_concentration)
    _, total_efficiency = _compute_efficiencies(
        results, particle_diameters, k, n)
    return total_efficiency - efficiency_target


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

    # Абсолютные размеры геометрии, вычисленные из найденного Dc и пропорций.
    Dc = Dc_solution
    Di = Dc * ratios['Di/Dc']
    Do = Dc * ratios['Do/Dc']
    Du = Dc * ratios['Du/Dc']
    total_length = Dc * ratios['L/Dc']
    vortex_finder_length = Dc * ratios['l/Dc']

    # Вычисление длины цилиндрической части Lc
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


if __name__ == '__main__':
    from pytroleum.plant.solid_cyclone.properties import PhysicalProperties
    from pytroleum.plant.solid_cyclone.models import RietemaHydrocyclone

    properties = PhysicalProperties(solid_density=2650)
    feed_volumetric_concentration = 0.05

    ratios_rietema = {
        'Di/Dc': 0.20,
        'Do/Dc': 0.25,
        'Du/Dc': 0.15,
        'L/Dc': 4.5,
        'l/Dc': 0.40,
        'angle': 11.0,
    }

    Q = 12.0 / (1000 * 60)
    k = 50e-6
    n = 1.5
    particle_diameters = np.linspace(0, 1e-3, 500)

    # Задача 1: найти Dc для d₅₀'
    cut_size_target = 3.88e-6
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
    print(f"L  = {res1['L']*1e3:.2f} мм")
    print(f"Lc = {res1['Lc']*1e3:.2f} мм")
    print(f"vortex_finder_length = {res1['vortex_finder_length']*1e3:.2f} мм")
    print(f"angle = {res1['angle']:.1f} °")
    print(f"Q    = {res1['feed_volumetric_flow_rate']*6e4:.3f} л/мин")
    print(f"ΔP   = {res1['pressure_drop']/1e3:.2f} кПа")
    print(f"Rw   = {res1['water_flow_ratio']:.4f}")
    print(f"d50' = {res1['reduced_cut_size']*1e6:.2f} мкм "
          f"(цель: {cut_size_target*1e6:.2f} мкм)")
    print(f"E_T' = {res1['reduced_total_efficiency']*100:.1f} %")
    print(f"E_T  = {res1['total_efficiency']*100:.1f} %")

    print("\n" + "="*60 + "\n")

    # Задача 2: найти Dc для E_T
    efficiency_target = 0.98
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
    print(f"L  = {res2['L']*1e3:.2f} мм")
    print(f"Lc = {res2['Lc']*1e3:.2f} мм")
    print(f"vortex_finder_length = {res2['vortex_finder_length']*1e3:.2f} мм")
    print(f"angle = {res2['angle']:.1f} °")
    print(f"Q    = {res2['feed_volumetric_flow_rate']*6e4:.3f} л/мин")
    print(f"ΔP   = {res2['pressure_drop']/1e3:.2f} кПа")
    print(f"Rw   = {res2['water_flow_ratio']:.4f}")
    print(f"d50' = {res2['reduced_cut_size']*1e6:.2f} мкм")
    print(f"E_T' = {res2['reduced_total_efficiency']*100:.1f} %")
    print(
        f"E_T  = {res2['total_efficiency']*100:.1f} % "
        f"(цель: {efficiency_target*100:.1f} %)"
    )
    print("\n" + "="*60 + "\n")

    # NOTE Сюда же можно добавить проверку - после получения Dc провести расчёт
    # NOTE по известному диаметру и расходу гидроциклона (и такими же пропорциями)
    # NOTE
    # NOTE Для первого случая нужно убедиться, что d50 получается такой же
    # NOTE Для второго случая нужно убедиться, что эффективность получается такой же
    # NOTE
    # NOTE Если расчёт для Dc и Q даст такие же значения для d50 (или E), то задача
    # NOTE сошлась
    # NOTE
    # NOTE Сначала всё это целесообразнее сделать тут, потом можно перекинуть в
    # NOTE examples
