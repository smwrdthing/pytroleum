"""
Модуль содержит функции для визуализации характеристик гидроциклонов:
распределение частиц, плотность распределения, приведённую эффективность разделения.
"""

from typing import Literal

from matplotlib.axes import Axes
from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt

from hydrocyclone.models import BaseHydrocyclone
from hydrocyclone.properties import PhysicalProperties
from hydrocyclone.configs import build_standard_configs
from hydrocyclone.efficiency import (
    calculate_cumulative_particle_size_distribution,
    calculate_reduced_grade_efficiency,
    calculate_reduced_total_efficiency,
    calculate_total_efficiency,
    _probability_density,
)

CUMULATIVE_DISTRIBUTION_COL = 0  # столбец 0 — кумулятивное распределение
PROBABILITY_DENSITY_COL = 1      # столбец 1 — плотность распределения
GRADE_EFFICIENCY_COL = 2         # столбец 2 — вероятность уноса


def _plot_cumulative_distribution(
    ax: Axes,
    hydrocyclone: BaseHydrocyclone,
    particle_diameters: NDArray,
    reduced_cut_size: float,
    k: float,
    n: float,
) -> None:
    """График кумулятивного распределения частиц по Розин-Раммлеру."""
    normalized_diameter = particle_diameters / reduced_cut_size
    y = calculate_cumulative_particle_size_distribution(
        particle_diameters, k, n)
    ax.plot(normalized_diameter, y, 'b-', linewidth=2)
    ax.set_xlabel("$d/d_{50}'$", fontsize=10)
    ax.set_ylabel('y(d)', fontsize=10)
    ax.set_xlim((0, 10))
    ax.set_ylim((0, 1.2))
    ax.grid(True, alpha=0.3)
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    ax.text(
        0.95, 0.05,
        f'{hydrocyclone.name}\nk={k*1e6:.0f} мкм\nn={n:.1f}',
        transform=ax.transAxes, fontsize=9,
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5),
    )


def _plot_probability_density(
    ax: Axes,
    hydrocyclone: BaseHydrocyclone,
    particle_diameters: NDArray,
    reduced_cut_size: float,
    k: float,
    n: float,
) -> None:
    """График плотности распределения частиц."""
    normalized_diameter = particle_diameters / reduced_cut_size
    dy_dd = _probability_density(particle_diameters, k, n)
    dy_dd_scaled = dy_dd * reduced_cut_size
    ax.plot(normalized_diameter, dy_dd_scaled, 'b-', linewidth=2)
    ax.set_xlabel("$d/d_{50}'$", fontsize=10)
    ax.set_ylabel("$dy/d(d/d_{50}')$", fontsize=10)
    ax.set_xlim((0, 10))
    ax.set_ylim((0, 0.3))
    ax.grid(True, alpha=0.3)
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    ax.text(
        0.95, 0.95,
        f'{hydrocyclone.name}\nk={k*1e6:.0f} мкм\nn={n:.1f}',
        transform=ax.transAxes, fontsize=9,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5),
    )


def _plot_grade_efficiency(
    ax: Axes,
    hydrocyclone: BaseHydrocyclone,
    particle_diameters: NDArray,
    results: dict[str, float],
    operation_mode: Literal['Q', 'delta_p'],
) -> NDArray:
    """График приведённой и полной вероятности уноса. Возвращает G'(d)."""
    reduced_cut_size = results['reduced_cut_size']
    water_flow_ratio = results['water_flow_ratio']
    m = results['m']
    alpha = results['alpha']

    normalized_diameter = particle_diameters / reduced_cut_size
    reduced_grade_efficiency = calculate_reduced_grade_efficiency(
        particle_diameters, reduced_cut_size, 'plitt', m, alpha)
    G = reduced_grade_efficiency * (1 - water_flow_ratio) + water_flow_ratio

    ax.plot(normalized_diameter, reduced_grade_efficiency, 'r-',
            linewidth=2, label="$G'(d/d_{50}')$")
    ax.plot(normalized_diameter, G, 'b--', linewidth=2, label="$G(d/d_{50}')$")
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    ax.set_xlabel("$d/d_{50}'$", fontsize=10)
    ax.set_ylabel("$G(d)$, $G'(d)$", fontsize=10)
    ax.set_xlim((0, 10))
    ax.set_ylim((0, 1.2))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    if operation_mode == 'Q':
        info_text = (
            f'{hydrocyclone.name}\n'
            f"$d_{{50}}'$={reduced_cut_size*1e6:.1f} мкм\n"
            f'm={m:.2f}\n'
            f'$R_w$={water_flow_ratio:.3f}\n'
            f'Q={results["feed_volumetric_flow_rate"]*1000*60:.1f} л/мин'
        )
    else:
        info_text = (
            f'{hydrocyclone.name}\n'
            f"$d_{{50}}'$={reduced_cut_size*1e6:.1f} мкм\n"
            f'm={m:.2f}\n'
            f'$R_w$={water_flow_ratio:.3f}\n'
            f'ΔP={results["pressure_drop"]/1000:.2f} кПа'
        )

    ax.text(
        0.95, 0.05, info_text,
        transform=ax.transAxes, fontsize=9,
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5),
    )

    return reduced_grade_efficiency


def _print_efficiency(
    hydrocyclone: BaseHydrocyclone,
    results: dict[str, float],
    particle_diameters: NDArray,
    k: float,
    n: float,
    reduced_grade_efficiency: NDArray,
) -> None:
    """Вывод эффективности разделения в консоль."""
    reduced_total_efficiency = calculate_reduced_total_efficiency(
        particle_diameters, k, n, reduced_grade_efficiency)
    total_efficiency = calculate_total_efficiency(
        reduced_total_efficiency, results['water_flow_ratio'])

    print(f"=== {hydrocyclone.name} ===")
    print(f"Q = {results['feed_volumetric_flow_rate']*1000*60:.2f} л/мин, "
          f"ΔP = {results['pressure_drop']/1000:.2f} кПа")
    print(f"E_T' = {reduced_total_efficiency*100:.1f}%")
    print(f"E_T  = {total_efficiency*100:.1f}%")
    print()


def _plot_row(
    axes_row,
    hydrocyclone: BaseHydrocyclone,
    results: dict[str, float],
    particle_diameters: NDArray,
    k: float,
    n: float,
    operation_mode: Literal['Q', 'delta_p'],
) -> None:
    """Рисует строку из трёх графиков для одного типа гидроциклона."""
    reduced_cut_size = results['reduced_cut_size']

    _plot_cumulative_distribution(axes_row[CUMULATIVE_DISTRIBUTION_COL],
                                  hydrocyclone, particle_diameters,
                                  reduced_cut_size,
                                  k,
                                  n)
    _plot_probability_density(axes_row[PROBABILITY_DENSITY_COL],
                              hydrocyclone,
                              particle_diameters,
                              reduced_cut_size,
                              k,
                              n)
    reduced_grade_efficiency = _plot_grade_efficiency(axes_row[GRADE_EFFICIENCY_COL],
                                                      hydrocyclone,
                                                      particle_diameters,
                                                      results,
                                                      operation_mode)
    _print_efficiency(
        hydrocyclone, results, particle_diameters, k, n, reduced_grade_efficiency)


def plot_hydrocyclone_analysis_Q(
    feed_volumetric_flow_rate: float,
    feed_volumetric_concentration: float,
    hydrocyclone_diameter: float,
    properties: PhysicalProperties,
    k: float,
    n: float,
    particle_diameters: NDArray,
) -> None:
    """Построение графиков анализа гидроциклонов при заданном расходе."""
    _plot_hydrocyclone_analysis(
        'Q', hydrocyclone_diameter, feed_volumetric_concentration,
        properties, k, n, particle_diameters,
        feed_volumetric_flow_rate=feed_volumetric_flow_rate)


def plot_hydrocyclone_analysis_delta_p(
    pressure_drop: float,
    feed_volumetric_concentration: float,
    hydrocyclone_diameter: float,
    properties: PhysicalProperties,
    k: float,
    n: float,
    particle_diameters: NDArray,
) -> None:
    """Построение графиков анализа гидроциклонов при заданном перепаде давления."""
    _plot_hydrocyclone_analysis(
        'delta_p', hydrocyclone_diameter, feed_volumetric_concentration,
        properties, k, n, particle_diameters,
        pressure_drop=pressure_drop)


def _plot_hydrocyclone_analysis(
    operation_mode: Literal['Q', 'delta_p'],
    hydrocyclone_diameter: float,
    feed_volumetric_concentration: float,
    properties: PhysicalProperties,
    k: float,
    n: float,
    particle_diameters: NDArray,
    feed_volumetric_flow_rate: float = 0.0,
    pressure_drop: float = 0.0,
) -> None:
    """Общая логика построения графиков."""
    hydrocyclones = build_standard_configs(hydrocyclone_diameter)

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))

    for row, hydrocyclone in enumerate(hydrocyclones):
        if operation_mode == 'Q':
            results = hydrocyclone.calculate_from_flow_rate(
                properties, feed_volumetric_flow_rate, feed_volumetric_concentration)
        else:
            results = hydrocyclone.calculate_from_pressure_drop(
                properties, pressure_drop, feed_volumetric_concentration)
        _plot_row(axes[row], hydrocyclone, results,
                  particle_diameters, k, n, operation_mode)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("МОДЕЛЬ РАСЧЁТА ГИДРОЦИКЛОНА")
    print("="*60)

    properties = PhysicalProperties(solid_density=2650)
    k = 50e-6   # характерный размер частиц Розин-Раммлера
    n = 1.5     # параметр Розин-Раммлера
    particle_diameters = np.linspace(0, 1e-3, 500)

    print("\n[Режим 1] Фиксированный перепад давления")
    plot_hydrocyclone_analysis_delta_p(
        pressure_drop=100000,
        feed_volumetric_concentration=0.05,
        hydrocyclone_diameter=45e-3,
        properties=properties,
        k=k, n=n, particle_diameters=particle_diameters,
    )

    # print("\n[Режим 2] Фиксированный расход")
    # plot_hydrocyclone_analysis_Q(
    #     feed_volumetric_flow_rate=0.0002,
    #     feed_volumetric_concentration=0.05,
    #     hydrocyclone_diameter=45e-3,
    #     properties=properties,
    #     k=k, n=n, particle_diameters=particle_diameters,
    # )
