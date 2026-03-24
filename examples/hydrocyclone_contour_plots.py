"""
Contour plots of hydrocyclone characteristics.
"""

import numpy as np
import matplotlib.pyplot as plt

from pytroleum.plant.solid_cyclone.geometry import (
    CycloneDesign,
    RIETEMA_DIAMETER_PROPORTIONS, RIETEMA_LENGTH_PROPORTIONS, RIETEMA_CONE_ANGLE,
    BRADLEY_DIAMETER_PROPORTIONS, BRADLEY_LENGTH_PROPORTIONS, BRADLEY_CONE_ANGLE,
    DEMCO_DIAMETER_PROPORTIONS, DEMCO_LENGTH_PROPORTIONS, DEMCO_CONE_ANGLE,
)
from pytroleum.plant.solid_cyclone.inputs import (
    PhysicalProperties,
    SizeDistribution,
)
from pytroleum.plant.solid_cyclone.models import (
    BaseHydrocyclone,
    RietemaHydrocyclone,
    BradleyHydrocyclone,
    DemcoHydrocyclone,
)
from pytroleum.plant.solid_cyclone.efficiency import (
    calculate_reduced_grade_efficiency,
    calculate_reduced_total_efficiency,
    calculate_total_efficiency,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


GRID_ROWS = 50
GRID_COLS = 50

Q_MIN = 5.0    # L/min
Q_MAX = 25.0   # L/min

DC_MIN = 10e-3  # m
DC_MAX = 30e-3  # m

MODELS = [
    ('Rietema',
     CycloneDesign(0.01, RIETEMA_DIAMETER_PROPORTIONS,
                   RIETEMA_LENGTH_PROPORTIONS, RIETEMA_CONE_ANGLE),
     RietemaHydrocyclone),
    ('Bradley',
     CycloneDesign(0.01, BRADLEY_DIAMETER_PROPORTIONS,
                   BRADLEY_LENGTH_PROPORTIONS, BRADLEY_CONE_ANGLE),
     BradleyHydrocyclone),
    ('Demco',
     CycloneDesign(0.01, DEMCO_DIAMETER_PROPORTIONS,
                   DEMCO_LENGTH_PROPORTIONS, DEMCO_CONE_ANGLE),
     DemcoHydrocyclone),
]

# ---------------------------------------------------------------------------
# Grid computation
# ---------------------------------------------------------------------------


def _compute_grid(
    Dc_grid: np.ndarray,
    Q_grid: np.ndarray,
    design: CycloneDesign,
    hydrocyclone_cls: type[BaseHydrocyclone],
    properties: PhysicalProperties,
    feed_volumetric_concentration: float,
    size_dist: SizeDistribution,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    n_Dc, n_Q = Dc_grid.shape
    cut_size = np.empty((n_Dc, n_Q))
    water_ratio = np.empty((n_Dc, n_Q))
    reduced_eff = np.empty((n_Dc, n_Q))
    total_eff = np.empty((n_Dc, n_Q))

    for i in range(n_Dc):
        hydrocyclone = hydrocyclone_cls(
            '', CycloneDesign(Dc_grid[i, 0],
                              design.diameter_proportions,
                              design.length_proportions,
                              design.cone_angle))
        for j in range(n_Q):
            hydrocyclone.calculate_from_flow_rate(
                properties, Q_grid[i, j], feed_volumetric_concentration)

            grade_eff = calculate_reduced_grade_efficiency(
                size_dist.particle_diameters, hydrocyclone.reduced_cut_size,
                'plitt', hydrocyclone.m, hydrocyclone.alpha)
            reduced_total = calculate_reduced_total_efficiency(
                size_dist.particle_diameters, grade_eff, size_dist.k, size_dist.n)
            total = calculate_total_efficiency(
                reduced_total, hydrocyclone.water_flow_ratio)

            cut_size[i, j] = hydrocyclone.reduced_cut_size * 1e6
            water_ratio[i, j] = hydrocyclone.water_flow_ratio
            reduced_eff[i, j] = reduced_total * 100
            total_eff[i, j] = total * 100

    return cut_size, water_ratio, reduced_eff, total_eff

# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------


if __name__ == '__main__':
    properties = PhysicalProperties(solid_density=1500)
    feed_volumetric_concentration = 0.00033

    size_dist = SizeDistribution(
        # type: ignore[call-overload]
        particle_diameters=np.linspace(1e-6, 200e-6, 500),
        k=10.9918e-6,
        n=0.9187,
    )

    Dc_range = np.linspace(DC_MIN, DC_MAX, GRID_ROWS)
    Q_range = np.linspace(Q_MIN / (1000 * 60), Q_MAX / (1000 * 60), GRID_COLS)
    Q_grid, Dc_grid = np.meshgrid(Q_range, Dc_range)

    Q_plot = Q_grid * 1000 * 60   # m³/s → L/min
    Dc_plot = Dc_grid * 1000        # m → mm

    for name, design, cls in MODELS:
        cut_size, water_ratio, reduced_eff, total_eff = _compute_grid(
            Dc_grid, Q_grid, design, cls,
            properties, feed_volumetric_concentration, size_dist
        )

        _, ax = plt.subplots(figsize=(6, 5))
        contour = ax.contour(Q_plot, Dc_plot, cut_size)
        ax.clabel(contour, inline=True, fontsize=8)
        ax.set_xlabel('$Q$, L/min')
        ax.set_ylabel('$D_c$, mm')
        ax.set_title(f"{name} — reduced cut size $d_{{50}}'$, µm")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        _, ax = plt.subplots(figsize=(6, 5))
        contour = ax.contour(Q_plot, Dc_plot, water_ratio, levels=10)
        ax.clabel(contour, inline=True, fontsize=8)
        ax.set_xlabel('$Q$, L/min')
        ax.set_ylabel('$D_c$, mm')
        ax.set_title(f'{name} — liquid fraction in underflow $R_w$')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        _, ax = plt.subplots(figsize=(6, 5))
        contour = ax.contour(
            Q_plot, Dc_plot, reduced_eff, levels=np.arange(50, 105, 5))
        ax.clabel(contour, inline=True, fontsize=8)
        ax.set_xlabel('$Q$, L/min')
        ax.set_ylabel('$D_c$, mm')
        ax.set_title(f"{name} — reduced efficiency $E_T'$, %")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        _, ax = plt.subplots(figsize=(6, 5))
        contour = ax.contour(
            Q_plot, Dc_plot, total_eff, levels=np.arange(50, 105, 5))
        ax.clabel(contour, inline=True, fontsize=8)
        ax.set_xlabel('$Q$, L/min')
        ax.set_ylabel('$D_c$, mm')
        ax.set_title(f'{name} — total efficiency $E_T$, %')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
