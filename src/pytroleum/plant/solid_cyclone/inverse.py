"""
Обратная задача гидроциклона: поиск диаметра корпуса Dc
при заданном расходе Q или перепаде давления ΔP.
"""
from typing import Literal

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


def _validate_geometry_params(geometry_params: dict[str, float]) -> None:
    """
    Проверка входных геометрических параметров перед решением обратной задачи.

    Проверяется одно условие:
      angle ∈ [CYCLONE_CONE_ANGLE_MIN, CYCLONE_CONE_ANGLE_MAX]
    """
    angle = geometry_params['angle']
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


def _residual_pressure_drop(
        Dc: float,
        pressure_drop_target: float,
        feed_volumetric_flow_rate: float,
        ratios: dict[str, float],
        hydrocyclone_cls: type[BaseHydrocyclone],
        properties: PhysicalProperties,
        feed_volumetric_concentration: float,
) -> float:
    """
    Невязка для режима 'Q': f(Dc) = ΔP(Dc, Q) - ΔP_target.
    """
    hydrocyclone = _build_from_ratios(Dc, ratios, hydrocyclone_cls)
    results = hydrocyclone.calculate_from_flow_rate(
        properties, feed_volumetric_flow_rate, feed_volumetric_concentration)
    return results['pressure_drop'] - pressure_drop_target


def _residual_flow_rate(
        Dc: float,
        feed_volumetric_flow_rate_target: float,
        pressure_drop: float,
        ratios: dict[str, float],
        hydrocyclone_cls: type[BaseHydrocyclone],
        properties: PhysicalProperties,
        feed_volumetric_concentration: float,
) -> float:
    """
    Невязка для режима 'delta_p': f(Dc) = Q(Dc, ΔP) - Q_target.
    """
    hydrocyclone = _build_from_ratios(Dc, ratios, hydrocyclone_cls)
    results = hydrocyclone.calculate_from_pressure_drop(
        properties, pressure_drop, feed_volumetric_concentration)
    return results['feed_volumetric_flow_rate'] - feed_volumetric_flow_rate_target


def find_Dc(
        operation_mode: Literal['Q', 'delta_p'],
        feed_volumetric_flow_rate: float,
        pressure_drop: float,
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
    Найти Dc при заданном режиме работы.

    Режим 'Q': решается f(Dc) = ΔP(Dc, Q) - pressure_drop = 0
    Режим 'delta_p': решается f(Dc) = Q(Dc, ΔP) - feed_volumetric_flow_rate = 0

    """
    _validate_geometry_params(ratios)

    if Dc0 is None:
        Dc0 = _initial_Dc(feed_volumetric_flow_rate, ratios['Di/Dc'])

    if operation_mode == 'Q':

        Dc_solution = fsolve(
            _residual_pressure_drop, x0=Dc0,
            args=(pressure_drop, feed_volumetric_flow_rate,
                  ratios, hydrocyclone_cls, properties, feed_volumetric_concentration),
        )[0]

        hydrocyclone = _build_from_ratios(
            Dc_solution, ratios, hydrocyclone_cls)
        results = hydrocyclone.calculate_from_flow_rate(
            properties, feed_volumetric_flow_rate, feed_volumetric_concentration)

    else:

        Dc_solution = fsolve(
            _residual_flow_rate, x0=Dc0,
            args=(feed_volumetric_flow_rate, pressure_drop,
                  ratios, hydrocyclone_cls, properties, feed_volumetric_concentration),
        )[0]

        hydrocyclone = _build_from_ratios(
            Dc_solution, ratios, hydrocyclone_cls)
        results = hydrocyclone.calculate_from_pressure_drop(
            properties, pressure_drop, feed_volumetric_concentration)

    # G'(d) — приведённая вероятность уноса по модели Plitt.
    reduced_grade_efficiency = calculate_reduced_grade_efficiency(
        particle_diameters,
        results['reduced_cut_size'],
        'plitt',
        results['m'],
        results['alpha'],
    )

    # E_T' — приведённая эффективность
    reduced_total_efficiency = calculate_reduced_total_efficiency(
        particle_diameters, k, n, reduced_grade_efficiency)

    # E_T — полная эффективность
    total_efficiency = calculate_total_efficiency(
        reduced_total_efficiency, results['water_flow_ratio'])

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
        'Lc': Lc,  # Добавлена длина цилиндрической части
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

    k = 50e-6
    n = 1.5
    particle_diameters = np.linspace(0, 1e-3, 500)

    # Режим 'delta_p': ΔP задан, найти Dc при котором Q = Q_target.
    print("РЕЖИМ 'delta_p': ПОИСК Dc ПО ЗАДАННОМУ ПЕРЕПАДУ ДАВЛЕНИЯ")
    print("-" * 60)
    res1 = find_Dc(
        operation_mode='delta_p',
        feed_volumetric_flow_rate=12.0 / (1000 * 60),
        pressure_drop=100e3,
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
    print(f"Lc = {res1['Lc']*1e3:.2f} мм")  # Добавлен вывод Lc
    print(f"vortex_finder_length = {res1['vortex_finder_length']*1e3:.2f} мм")
    print(f"angle = {res1['angle']:.1f} °")
    print(f"ΔP = {res1['pressure_drop']/1e3:.2f} кПа  (задан)")
    print(f"Q = {res1['feed_volumetric_flow_rate']*6e4:.3f} л/мин  (цель)")
    print(f"Rw = {res1['water_flow_ratio']:.4f}")
    print(f"d50'= {res1['reduced_cut_size']*1e6:.2f} мкм")
    print(f"E_T'= {res1['reduced_total_efficiency']*100:.1f} %")
    print(f"E_T = {res1['total_efficiency']*100:.1f} %")

    print("\n" + "="*60 + "\n")

    # Режим 'Q': Q задан, найти Dc при котором ΔP = pressure_drop_target.
    print("РЕЖИМ 'Q': ПОИСК Dc ПО ЗАДАННОМУ РАСХОДУ")
    print("-" * 60)
    res2 = find_Dc(
        operation_mode='Q',
        feed_volumetric_flow_rate=12 / (1000 * 60),
        pressure_drop=100e3,
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
    print(f"vortex_finder_length = {res2['vortex_finder_length']*1e3:.2f} мм")
    print(f"angle = {res2['angle']:.1f} °")
    print(f"ΔP = {res2['pressure_drop']/1e3:.2f} кПа  (цель)")
    print(f"Q = {res2['feed_volumetric_flow_rate']*6e4:.3f} л/мин  (задан)")
    print(f"Rw = {res2['water_flow_ratio']:.4f}")
    print(f"d50'= {res2['reduced_cut_size']*1e6:.2f} мкм")
    print(f"E_T'= {res2['reduced_total_efficiency']*100:.1f} %")
    print(f"E_T = {res2['total_efficiency']*100:.1f} %")

    print("\n" + "="*60 + "\n")
