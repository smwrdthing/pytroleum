"""
Contour plots of hydrocyclone characteristics.
"""

import numpy as np
import matplotlib.pyplot as plt

from pytroleum.plant.solid_cyclone.geometry import (
    CycloneDesign,
    RIETEMA_DEFAULT_PROPORTIONS,
    BRADLEY_DEFAULT_PROPORTIONS,
    DEMCO_DEFAULT_PROPORTIONS,
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

CUT_SIZE_IDX = 0
WATER_RATIO_IDX = 1
REDUCED_EFF_IDX = 2
TOTAL_EFF_IDX = 3

GRID_ROWS = 50
GRID_COLS = 50

Q_MIN = 5.0    # L/min
Q_MAX = 25.0   # L/min

DC_MIN = 10e-3  # m
DC_MAX = 30e-3  # m

MODELS: list[tuple[str, list, type[BaseHydrocyclone]]] = [
    ('Rietema', RIETEMA_DEFAULT_PROPORTIONS, RietemaHydrocyclone),
    ('Bradley', BRADLEY_DEFAULT_PROPORTIONS, BradleyHydrocyclone),
    ('Demco', DEMCO_DEFAULT_PROPORTIONS, DemcoHydrocyclone),
]

# ---------------------------------------------------------------------------
# Grid computation
# ---------------------------------------------------------------------------


def _compute_grid(
    Dc_grid: np.ndarray,
    Q_grid: np.ndarray,
    proportions: list[float],
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
            '', CycloneDesign(Dc_grid[i, 0], proportions))
        for j in range(n_Q):
            res = hydrocyclone.calculate_from_flow_rate(
                properties, Q_grid[i, j], feed_volumetric_concentration)

            grade_eff = calculate_reduced_grade_efficiency(
                size_dist.particle_diameters, res['reduced_cut_size'],
                'plitt', res['m'], res['alpha'])
            reduced_total = calculate_reduced_total_efficiency(
                size_dist.particle_diameters, size_dist.k, size_dist.n, grade_eff)
            total = calculate_total_efficiency(
                reduced_total, res['water_flow_ratio'])

            cut_size[i, j] = res['reduced_cut_size'] * 1e6
            water_ratio[i, j] = res['water_flow_ratio']
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
        particle_diameters=np.linspace(1e-6, 200e-6, 500),
        k=10.9918e-6,
        n=0.9187,
    )

    Dc_range = np.linspace(DC_MIN, DC_MAX, GRID_ROWS)
    Q_range = np.linspace(Q_MIN / (1000 * 60), Q_MAX / (1000 * 60), GRID_COLS)
    Q_grid, Dc_grid = np.meshgrid(Q_range, Dc_range)

    Q_plot = Q_grid * 1000 * 60   # m³/s → L/min
    Dc_plot = Dc_grid * 1000        # m → mm

    # Compute grids for each model
    grids = {}
    for name, proportions, cls in MODELS:
        grids[name] = _compute_grid(
            Dc_grid, Q_grid, proportions, cls,
            properties, feed_volumetric_concentration, size_dist,
        )

    # ---------------------------------------------------------------------------
    # Rietema
    # ---------------------------------------------------------------------------

    _, ax = plt.subplots(figsize=(6, 5))
    contour = ax.contour(Q_plot, Dc_plot, grids['Rietema'][CUT_SIZE_IDX])
    ax.clabel(contour, inline=True, fontsize=8)
    ax.set_xlabel('$Q$, L/min')
    ax.set_ylabel('$D_c$, mm')
    ax.set_title("Rietema — reduced cut size $d_{50}'$, µm")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    _, ax = plt.subplots(figsize=(6, 5))
    contour = ax.contour(
        Q_plot, Dc_plot, grids['Rietema'][WATER_RATIO_IDX], levels=10)
    ax.clabel(contour, inline=True, fontsize=8)
    ax.set_xlabel('$Q$, L/min')
    ax.set_ylabel('$D_c$, mm')
    ax.set_title('Rietema — liquid fraction in underflow $R_w$')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    _, ax = plt.subplots(figsize=(6, 5))
    contour = ax.contour(
        Q_plot, Dc_plot, grids['Rietema'][REDUCED_EFF_IDX], levels=np.arange(50, 105, 5))
    ax.clabel(contour, inline=True, fontsize=8)
    ax.set_xlabel('$Q$, L/min')
    ax.set_ylabel('$D_c$, mm')
    ax.set_title("Rietema — reduced efficiency $E_T'$, %")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    _, ax = plt.subplots(figsize=(6, 5))
    contour = ax.contour(
        Q_plot, Dc_plot, grids['Rietema'][TOTAL_EFF_IDX], levels=np.arange(50, 105, 5))
    ax.clabel(contour, inline=True, fontsize=8)
    ax.set_xlabel('$Q$, L/min')
    ax.set_ylabel('$D_c$, mm')
    ax.set_title('Rietema — total efficiency $E_T$, %')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------------------------
    # Bradley
    # ---------------------------------------------------------------------------

    _, ax = plt.subplots(figsize=(6, 5))
    contour = ax.contour(Q_plot, Dc_plot, grids['Bradley'][CUT_SIZE_IDX])
    ax.clabel(contour, inline=True, fontsize=8)
    ax.set_xlabel('$Q$, L/min')
    ax.set_ylabel('$D_c$, mm')
    ax.set_title("Bradley — reduced cut size $d_{50}'$, µm")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    _, ax = plt.subplots(figsize=(6, 5))
    contour = ax.contour(
        Q_plot, Dc_plot, grids['Bradley'][WATER_RATIO_IDX], levels=10)
    ax.clabel(contour, inline=True, fontsize=8)
    ax.set_xlabel('$Q$, L/min')
    ax.set_ylabel('$D_c$, mm')
    ax.set_title('Bradley — liquid fraction in underflow $R_w$')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    _, ax = plt.subplots(figsize=(6, 5))
    contour = ax.contour(
        Q_plot, Dc_plot, grids['Bradley'][REDUCED_EFF_IDX], levels=np.arange(50, 105, 5))
    ax.clabel(contour, inline=True, fontsize=8)
    ax.set_xlabel('$Q$, L/min')
    ax.set_ylabel('$D_c$, mm')
    ax.set_title("Bradley — reduced efficiency $E_T'$, %")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    _, ax = plt.subplots(figsize=(6, 5))
    contour = ax.contour(
        Q_plot, Dc_plot, grids['Bradley'][TOTAL_EFF_IDX], levels=np.arange(50, 105, 5))
    ax.clabel(contour, inline=True, fontsize=8)
    ax.set_xlabel('$Q$, L/min')
    ax.set_ylabel('$D_c$, mm')
    ax.set_title('Bradley — total efficiency $E_T$, %')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------------------------
    # Demco
    # ---------------------------------------------------------------------------

    _, ax = plt.subplots(figsize=(6, 5))
    contour = ax.contour(Q_plot, Dc_plot, grids['Demco'][CUT_SIZE_IDX])
    ax.clabel(contour, inline=True, fontsize=8)
    ax.set_xlabel('$Q$, L/min')
    ax.set_ylabel('$D_c$, mm')
    ax.set_title("Demco — reduced cut size $d_{50}'$, µm")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    _, ax = plt.subplots(figsize=(6, 5))
    contour = ax.contour(
        Q_plot, Dc_plot, grids['Demco'][WATER_RATIO_IDX], levels=10)
    ax.clabel(contour, inline=True, fontsize=8)
    ax.set_xlabel('$Q$, L/min')
    ax.set_ylabel('$D_c$, mm')
    ax.set_title('Demco — liquid fraction in underflow $R_w$')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    _, ax = plt.subplots(figsize=(6, 5))
    contour = ax.contour(
        Q_plot, Dc_plot, grids['Demco'][REDUCED_EFF_IDX], levels=np.arange(50, 105, 5))
    ax.clabel(contour, inline=True, fontsize=8)
    ax.set_xlabel('$Q$, L/min')
    ax.set_ylabel('$D_c$, mm')
    ax.set_title("Demco — reduced efficiency $E_T'$, %")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    _, ax = plt.subplots(figsize=(6, 5))
    contour = ax.contour(
        Q_plot, Dc_plot, grids['Demco'][TOTAL_EFF_IDX], levels=np.arange(50, 105, 5))
    ax.clabel(contour, inline=True, fontsize=8)
    ax.set_xlabel('$Q$, L/min')
    ax.set_ylabel('$D_c$, mm')
    ax.set_title('Demco — total efficiency $E_T$, %')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
