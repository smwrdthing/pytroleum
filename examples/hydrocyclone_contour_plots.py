"""
Контурные графики характеристик гидроциклонов.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

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

CUT_SIZE_COL = 0          # столбец 0 — контурный график отсечного размера
WATER_RATIO_COL = 1       # столбец 1 — контурный график Rw
EFFICIENCY_COL = 2        # столбец 2 — контурный график приведённой эффективности E'_T
TOTAL_EFFICIENCY_COL = 3  # столбец 3 — контурный график полной эффективности E_T

TITLE_ROW = 0  # заголовки столбцов ставятся только в строке 0

GRID_ROWS = 50  # количество строк сетки
GRID_COLS = 50  # количество столбцов сетки

Q_MIN = 5.0   # минимальный расход, л/мин
Q_MAX = 25.0  # максимальный расход, л/мин


def _compute_grid(
    Dc_grid: np.ndarray,
    Q_grid: np.ndarray,
    proportions: list[float],
    hydrocyclone_cls: type[BaseHydrocyclone],
    properties: PhysicalProperties,
    feed_volumetric_concentration: float,
    size_dist: SizeDistribution,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Расчёт характеристик на всей сетке (Dc × Q)."""
    n_Dc, n_Q = Dc_grid.shape
    cut_size = np.empty((n_Dc, n_Q))
    water_ratio = np.empty((n_Dc, n_Q))
    reduced_eff = np.empty((n_Dc, n_Q))
    total_eff = np.empty((n_Dc, n_Q))

    for i in range(n_Dc):
        hydrocyclone = hydrocyclone_cls(
            '', CycloneDesign(Dc_grid[i, 0], proportions))
        for j in range(n_Q):
            results = hydrocyclone.calculate_from_flow_rate(
                properties, Q_grid[i, j], feed_volumetric_concentration)

            grade_eff = calculate_reduced_grade_efficiency(
                size_dist.particle_diameters, results['reduced_cut_size'],
                'plitt', results['m'], results['alpha'])
            reduced_total = calculate_reduced_total_efficiency(
                size_dist.particle_diameters, size_dist.k, size_dist.n,
                grade_eff)
            total = calculate_total_efficiency(
                reduced_total, results['water_flow_ratio'])

            cut_size[i, j] = results['reduced_cut_size'] * 1e6
            water_ratio[i, j] = results['water_flow_ratio']
            reduced_eff[i, j] = reduced_total * 100
            total_eff[i, j] = total * 100

    return cut_size, water_ratio, reduced_eff, total_eff


def _plot_one_contour_ax(
    ax: Axes,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z: np.ndarray,
    title: str | None,
    row: int,
    row_label: str | None = None,
    levels=None,
) -> None:
    """Рисует один контурный подграфик."""
    contour_kwargs = {} if levels is None else {'levels': levels}
    contour = ax.contour(x_grid, y_grid, z, **contour_kwargs)
    ax.clabel(contour, inline=True, fontsize=8)
    ax.set_xlabel('$Q$, л/мин')
    ax.set_ylabel('$D_c$, мм')
    if row == TITLE_ROW and title is not None:
        ax.set_title(title)
    if row_label is not None:
        ax.text(-0.3, 0.5, row_label, transform=ax.transAxes,
                fontsize=12, fontweight='bold', rotation=90, va='center')
    ax.grid(True, alpha=0.3)


def _plot_contour_row(
    axes_row,
    row: int,
    name: str,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    reduced_cut_size: np.ndarray,
    water_flow_ratio: np.ndarray,
    reduced_total_efficiency: np.ndarray,
    total_efficiency: np.ndarray,
) -> None:
    """Рисует строку из 4 контурных подграфиков для одного типа гидроциклонов."""
    _plot_one_contour_ax(
        axes_row[CUT_SIZE_COL], x_grid, y_grid, reduced_cut_size,
        title="Отсечной размер $d_{50}'$, мкм",
        row=row, row_label=name)
    _plot_one_contour_ax(
        axes_row[WATER_RATIO_COL], x_grid, y_grid, water_flow_ratio,
        title='Соотношение потоков воды $R_w$',
        row=row, levels=10)
    _plot_one_contour_ax(
        axes_row[EFFICIENCY_COL], x_grid, y_grid, reduced_total_efficiency,
        title="Приведённая эффективность $E_T'$, %",
        row=row, levels=np.arange(50, 100 + 5, 5))
    _plot_one_contour_ax(
        axes_row[TOTAL_EFFICIENCY_COL], x_grid, y_grid, total_efficiency,
        title='Полная эффективность $E_T$, %',
        row=row, levels=np.arange(50, 100 + 5, 5))


ALL_CONFIGS = [
    ('Rietema', RIETEMA_DEFAULT_PROPORTIONS, RietemaHydrocyclone),
    ('Bradley', BRADLEY_DEFAULT_PROPORTIONS, BradleyHydrocyclone),
    ('Demco', DEMCO_DEFAULT_PROPORTIONS, DemcoHydrocyclone),
]


def plot_contour_graphs(
    configs: list = ALL_CONFIGS,
) -> None:
    """Контурные графики для заданных типов гидроциклонов."""
    Dc_range = np.linspace(10e-3, 30e-3, GRID_ROWS)
    Q_range = np.linspace(Q_MIN / (1000 * 60), Q_MAX / (1000 * 60), GRID_COLS)
    Q_grid, Dc_grid = np.meshgrid(Q_range, Dc_range)

    properties = PhysicalProperties(solid_density=1500)
    feed_volumetric_concentration = 0.00033

    size_dist = SizeDistribution(
        particle_diameters=np.linspace(1e-6, 200e-6, 500),
        k=10.9918e-6,
        n=0.9187,
    )

    _, axes = plt.subplots(len(configs), 4, figsize=(20, 4 * len(configs)),
                           squeeze=False)

    for row, (name, proportions, hydrocyclone_cls) in enumerate(configs):
        cut_size, water_ratio, reduced_eff, total_eff = _compute_grid(
            Dc_grid, Q_grid, proportions, hydrocyclone_cls,
            properties, feed_volumetric_concentration, size_dist,
        )

        _plot_contour_row(
            axes[row], row, name,
            x_grid=Q_grid * 1000 * 60,
            y_grid=Dc_grid * 1000,
            reduced_cut_size=cut_size,
            water_flow_ratio=water_ratio,
            reduced_total_efficiency=reduced_eff,
            total_efficiency=total_eff,
        )

    plt.suptitle(
        f'Характеристики гидроциклонов '
        f'($C_v = {feed_volumetric_concentration}$)',
        fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Все три типа:
    plot_contour_graphs()

    # Только один тип:
    # plot_contour_graphs([ALL_CONFIGS[0]])  # Rietema
