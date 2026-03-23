"""
Visualization functions for hydrocyclone characteristics:
particle size distribution, probability density, and reduced separation efficiency.
"""

from matplotlib.axes import Axes
from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt

from pytroleum.plant.solid_cyclone.models import BaseHydrocyclone
from pytroleum.plant.solid_cyclone.inputs import (
    PhysicalProperties,
    OperationConditions,
    SizeDistribution,
)
from pytroleum.plant.solid_cyclone.geometry import build_standard_configs
from pytroleum.plant.solid_cyclone.efficiency import (
    cumulative_size_distribution,
    calculate_reduced_grade_efficiency,
    calculate_reduced_total_efficiency,
    calculate_total_efficiency,
    probability_density,
)

CUMULATIVE_DISTRIBUTION_COL = 0  # column 0 — cumulative distribution
PROBABILITY_DENSITY_COL = 1      # column 1 — probability density
GRADE_EFFICIENCY_COL = 2         # column 2 — grade efficiency


def _plot_cumulative_distribution(
    ax: Axes,
    hydrocyclone: BaseHydrocyclone,
    size_dist: SizeDistribution,
    reduced_cut_size: float,
) -> None:
    """Plot cumulative particle size distribution (Rosin-Rammler)."""
    normalized_diameter = size_dist.particle_diameters / reduced_cut_size
    y = cumulative_size_distribution(
        size_dist.particle_diameters, size_dist.k, size_dist.n)
    ax.plot(normalized_diameter, y, 'b-', linewidth=2)
    ax.set_xlabel("$d/d_{50}'$", fontsize=10)
    ax.set_ylabel('y(d)', fontsize=10)
    ax.set_xlim((0, 10))
    ax.set_ylim((0, 1.2))
    ax.grid(True, alpha=0.3)
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    ax.text(
        0.95, 0.05,
        f'{hydrocyclone.name}\n'
        f'k={size_dist.k*1e6:.0f} µm\n'
        f'n={size_dist.n:.1f}',
        transform=ax.transAxes, fontsize=9,
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5),
    )


def _plot_probability_density(
    ax: Axes,
    hydrocyclone: BaseHydrocyclone,
    size_dist: SizeDistribution,
    reduced_cut_size: float,
) -> None:
    """Plot particle size probability density."""
    normalized_diameter = size_dist.particle_diameters / reduced_cut_size
    dy_dd = probability_density(
        size_dist.particle_diameters, size_dist.k, size_dist.n)
    dy_dd_scaled = dy_dd * reduced_cut_size
    ax.plot(normalized_diameter, dy_dd_scaled, 'b-', linewidth=2)
    ax.set_xlabel("$d/d_{50}'$", fontsize=10)
    ax.set_ylabel("$dy/d(d/d_{50}')$", fontsize=10)
    ax.set_xlim((0, 10))
    ax.set_ylim((0, 0.6))
    ax.grid(True, alpha=0.3)
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    ax.text(
        0.95, 0.95,
        f'{hydrocyclone.name}\n'
        f'k={size_dist.k*1e6:.0f} µm\n'
        f'n={size_dist.n:.1f}',
        transform=ax.transAxes, fontsize=9,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5),
    )


def _plot_grade_efficiency(
    ax: Axes,
    hydrocyclone: BaseHydrocyclone,
    size_dist: SizeDistribution,
    conditions: OperationConditions,
) -> NDArray:
    """Plot reduced and total grade efficiency."""
    normalized_diameter = size_dist.particle_diameters / hydrocyclone.reduced_cut_size
    reduced_grade_efficiency = calculate_reduced_grade_efficiency(
        size_dist.particle_diameters, hydrocyclone.reduced_cut_size,
        'plitt', hydrocyclone.m, hydrocyclone.alpha)
    water_flow_ratio = hydrocyclone.water_flow_ratio
    G = reduced_grade_efficiency * (1 - water_flow_ratio) + water_flow_ratio

    ax.plot(normalized_diameter, reduced_grade_efficiency, 'r-',
            linewidth=2, label="$G'(d/d_{50}')$")
    ax.plot(normalized_diameter, G, 'b--', linewidth=2, label="$G(d/d_{50}')$")
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    ax.set_xlabel("$d/d_{50}'$", fontsize=10)
    ax.set_ylabel("$G(d)$, $G'(d)$", fontsize=10)
    ax.set_xlim((0, 3))
    ax.set_ylim((0, 1.2))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.text(
        0.95, 0.05,
        _grade_efficiency_info_text(hydrocyclone, conditions),
        transform=ax.transAxes, fontsize=9,
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5),
    )

    return reduced_grade_efficiency


def _grade_efficiency_info_text(
    hydrocyclone: BaseHydrocyclone,
    conditions: OperationConditions,
) -> str:
    """Build annotation text for the G(d) plot."""
    if conditions.mode == 'Q':
        operating_line = f'Q={hydrocyclone.feed_volumetric_flow_rate*1000*60:.1f} L/min'
    else:
        operating_line = f'ΔP={hydrocyclone.pressure_drop/1000:.2f} kPa'

    return (
        f'{hydrocyclone.name}\n'
        f"$d_{{50}}'$={hydrocyclone.reduced_cut_size*1e6:.1f} µm\n"
        f'm={hydrocyclone.m:.2f}\n'
        f'$R_w$={hydrocyclone.water_flow_ratio:.3f}\n'
        f'{operating_line}'
    )


def _compute_total_efficiencies(
    hydrocyclone: BaseHydrocyclone,
    size_dist: SizeDistribution,
    reduced_grade_efficiency: NDArray,
) -> tuple[NDArray | np.floating, NDArray | np.floating]:
    """Calculate E_T' and E_T."""
    reduced_total_efficiency = calculate_reduced_total_efficiency(
        size_dist.particle_diameters, reduced_grade_efficiency,
        size_dist.k, size_dist.n)
    total_efficiency = calculate_total_efficiency(
        reduced_total_efficiency, hydrocyclone.water_flow_ratio)
    return reduced_total_efficiency, total_efficiency


def _print_efficiency(
    hydrocyclone: BaseHydrocyclone,
    reduced_total_efficiency: float,
    total_efficiency: float,
) -> None:
    """Print separation efficiency to console."""
    hydrocyclone.design.summary()
    print(f"=== {hydrocyclone.name} ===")
    print(f"Q = {hydrocyclone.feed_volumetric_flow_rate*1000*60:.2f} L/min, "
          f"ΔP = {hydrocyclone.pressure_drop/1000:.2f} kPa")
    print(f"E_T' = {reduced_total_efficiency*100:.1f}%")
    print(f"E_T  = {total_efficiency*100:.1f}%")
    print()


def _plot_row(
    axes_row,
    hydrocyclone: BaseHydrocyclone,
    size_dist: SizeDistribution,
    conditions: OperationConditions,
) -> None:
    """Draw a row of three plots for one hydrocyclone type."""
    _plot_cumulative_distribution(axes_row[CUMULATIVE_DISTRIBUTION_COL],
                                  hydrocyclone,
                                  size_dist,
                                  hydrocyclone.reduced_cut_size)
    _plot_probability_density(axes_row[PROBABILITY_DENSITY_COL],
                              hydrocyclone,
                              size_dist,
                              hydrocyclone.reduced_cut_size)
    reduced_grade_efficiency = _plot_grade_efficiency(axes_row[GRADE_EFFICIENCY_COL],
                                                      hydrocyclone,
                                                      size_dist,
                                                      conditions)

    reduced_total_efficiency, total_efficiency = _compute_total_efficiencies(
        hydrocyclone, size_dist, reduced_grade_efficiency)
    _print_efficiency(
        hydrocyclone, float(reduced_total_efficiency), float(total_efficiency))


def _calculate_results(
    hydrocyclone: BaseHydrocyclone,
    properties: PhysicalProperties,
    conditions: OperationConditions,
) -> None:
    """Calculate hydrocyclone output parameters for the given operating conditions."""
    if conditions.mode == 'Q':
        hydrocyclone.calculate_from_flow_rate(
            properties,
            conditions.feed_volumetric_flow_rate,
            conditions.feed_volumetric_concentration,
        )
    else:
        hydrocyclone.calculate_from_pressure_drop(
            properties,
            conditions.pressure_drop,
            conditions.feed_volumetric_concentration,
        )


def plot_hydrocyclone_analysis(
    conditions: OperationConditions,
    hydrocyclone_diameter: float,
    properties: PhysicalProperties,
    size_dist: SizeDistribution,
) -> None:
    """Plot hydrocyclone analysis charts."""
    hydrocyclones = build_standard_configs(hydrocyclone_diameter)

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))

    for row, hydrocyclone in enumerate(hydrocyclones):
        _calculate_results(hydrocyclone, properties, conditions)
        _plot_row(axes[row], hydrocyclone, size_dist, conditions)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("HYDROCYCLONE CALCULATION MODEL")
    print("="*60)

    properties = PhysicalProperties(solid_density=1500)

    size_dist = SizeDistribution(
        particle_diameters=np.linspace(1e-6, 200e-6, 500),  # type: ignore[call-overload]
        k=10.9918e-6,
        n=0.9187,
    )

    # print("\n[Mode 1] Fixed pressure drop")
    # plot_hydrocyclone_analysis(
    #     conditions=OperatingConditions(
    #         feed_volumetric_concentration=0.05,
    #         mode='delta_p',
    #         pressure_drop=100000,
    #     ),
    #     hydrocyclone_diameter=45e-3,
    #     properties=properties,
    #     size_dist=size_dist,
    # )

    print("\n[Mode 2] Fixed flow rate")
    plot_hydrocyclone_analysis(
        conditions=OperationConditions(
            feed_volumetric_concentration=0.00033,
            mode='Q',
            feed_volumetric_flow_rate=0.0002,
        ),
        hydrocyclone_diameter=17.61e-3,
        properties=properties,
        size_dist=size_dist,
    )
