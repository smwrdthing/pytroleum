"""
Контурные графики характеристик гидроциклонов.
"""

import numpy as np
import matplotlib.pyplot as plt

from pytroleum.plant.solid_cyclone.geometry import GeometryParameters
from pytroleum.plant.solid_cyclone.properties import PhysicalProperties
from pytroleum.plant.solid_cyclone.models import (
    BaseHydrocyclone,
    RietemaHydrocyclone,
    BradleyHydrocyclone,
    DemcoHydrocyclone,
)
from pytroleum.plant.solid_cyclone.configs import build_standard_configs
from pytroleum.plant.solid_cyclone.efficiency import (
    calculate_reduced_grade_efficiency,
    calculate_reduced_total_efficiency,
)

CUT_SIZE_COL = 0     # столбец 0 — контурный график отсечного размера
WATER_RATIO_COL = 1  # столбец 1 — контурный график Rw
EFFICIENCY_COL = 2   # столбец 2 — контурный график приведённой эффективности E'_T

TITLE_ROW = 0  # заголовки столбцов ставятся только в строке 0

GRID_ROWS = 50  # количество строк сетки
GRID_COLS = 50  # количество столбцов сетки

V_IN_MIN = 1.0  # минимальная скорость во входном патрубке, м/с
V_IN_MAX = 3.0  # максимальная скорость во входном патрубке, м/с


def _compute_for_point(
    hydrocyclone_diameter: float,      # диаметр корпуса для данной точки сетки, м
    feed_volumetric_flow_rate: float,  # объёмный расход для данной точки сетки, м³/с
    geometry_ratios: dict,             # словарь пропорций {'Di/Dc':...}
    properties: PhysicalProperties,    # физические свойства жидкости и твёрдой фазы
    feed_volumetric_concentration: float,  # объёмная концентрация твёрдых частиц
    hydrocyclone_cls: type,            # NOTE см. вторую заметку в функции
    particle_diameters: np.ndarray,    # сетка диаметров частиц
    k: float,                          # параметр Розин-Раммлера k
    n: float,                          # параметр Розин-Раммлера n
) -> tuple[float, float, float]:
    """Расчёт характеристик гидроциклона для одной точки (Dc, Q)."""

    # NOTE у функции слишком много параметров

    # NOTE расчётная функциональность точно должна находиться внутри класса гидроциклона,
    # NOTE передавать функции целый класс конкретного гидроциклона для расчёта громоздко
    # NOTE и избыточно

    geometry = GeometryParameters.from_named(
        hydrocyclone_diameter=hydrocyclone_diameter,
        feed_inlet_diameter=hydrocyclone_diameter * geometry_ratios['Di/Dc'],
        overflow_diameter=hydrocyclone_diameter * geometry_ratios['Do/Dc'],
        underflow_diameter=hydrocyclone_diameter * geometry_ratios['Du/Dc'],
        hydrocyclone_length=hydrocyclone_diameter * geometry_ratios['L/Dc'],
        vortex_finder_length=hydrocyclone_diameter * geometry_ratios['l/Dc'],
        angle=geometry_ratios['angle'],
    )
    # создаёт экземпляр нужного подкласса; имя '' — не нужно для расчёта
    hydrocyclone: BaseHydrocyclone = hydrocyclone_cls('', geometry)
    # рассчитывает все параметры при заданном расходе Q
    results = hydrocyclone.calculate_from_flow_rate(properties,
                                                    feed_volumetric_flow_rate,
                                                    feed_volumetric_concentration)

    reduced_grade_efficiency = calculate_reduced_grade_efficiency(
        particle_diameters,
        results['reduced_cut_size'],
        'plitt',
        results['m'],
        results['alpha'],
    )
    efficiency = calculate_reduced_total_efficiency(
        particle_diameters, k, n, reduced_grade_efficiency) * 100

    return results['reduced_cut_size'] * 1e6, results['water_flow_ratio'], efficiency


compute_vectorized = np.vectorize(
    _compute_for_point,
    excluded=[
        'geometry_ratios',   # словарь пропорций одинаков для всей строки сетки
        'properties',        # физические свойства одинаковы для всей сетки
        'feed_volumetric_concentration',  # концентрация одинакова для всей сетки
        'hydrocyclone_cls',  # класс модели одинаков для всей строки сетки
        'particle_diameters',  # сетка частиц одинакова для всей сетки
        'k',                 # параметр Розин-Раммлера одинаков для всей сетки
        'n',                 # параметр Розин-Раммлера одинаков для всей сетки
    ],
)

# NOTE np.vectorize под капотом делает те же циклы на чистом Python и не избавляет от
# NOTE потенциальных проблем с производительностью, расчётные функции сами по себе должны
# NOTE быть нацелены на работу с массивами сопоставимых размеров, чтобы сказывалось
# NOTE преимущество от использования numpy


def plot_contour_graphs() -> None:
    """Контурные графики для трёх типов гидроциклонов."""
    hydrocyclone_diameter_range = np.linspace(30e-3, 80e-3, GRID_ROWS)

    properties = PhysicalProperties(solid_density=2650)
    feed_volumetric_concentration = 0.05  # объёмная концентрация твёрдых частиц 5%
    k = 50e-6                             # параметр Розин-Раммлера: 50 мкм
    n = 1.5                               # параметр Розин-Раммлера
    particle_diameters = np.linspace(0, 1e-3, 500)

    # создаёт три конфигурации при Dc=0.01 м; нас интересуют только их пропорции
    reference_configs = build_standard_configs(hydrocyclone_diameter_range[0])
    # список классов в том же порядке, что reference_configs
    hydrocyclone_classes = [RietemaHydrocyclone,
                            BradleyHydrocyclone, DemcoHydrocyclone]

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))

    for row, (hydrocyclone, hydrocyclone_cls) in enumerate(
            zip(reference_configs, hydrocyclone_classes)):

        # словарь безразмерных пропорций {'Di/Dc': ..., ...}
        ratios = hydrocyclone.geometry.get_geometry_ratios()
        # добавляет угол θ в словарь; get_geometry_ratios() не включает его
        ratios['angle'] = hydrocyclone.geometry.angle

        # Диапазон расходов из ограничения скорости: Q = v * π·Di²/4, Di = Dc·(Di/Dc)
        Di_Dc = ratios['Di/Dc']
        feed_volumetric_flow_rate_min = V_IN_MIN * np.pi * \
            (hydrocyclone_diameter_range[0] * Di_Dc)**2 / 4
        feed_volumetric_flow_rate_max = V_IN_MAX * np.pi * \
            (hydrocyclone_diameter_range[-1] * Di_Dc)**2 / 4
        feed_volumetric_flow_rate_range = np.linspace(feed_volumetric_flow_rate_min,
                                                      feed_volumetric_flow_rate_max,
                                                      GRID_COLS)

        # прямоугольная сетка в пространстве (Q, Dc)
        feed_volumetric_flow_rate_grid, hydrocyclone_diameter_grid = np.meshgrid(
            feed_volumetric_flow_rate_range, hydrocyclone_diameter_range)

        reduced_cut_size, water_flow_ratio, reduced_total_efficiency = compute_vectorized(
            hydrocyclone_diameter_grid, feed_volumetric_flow_rate_grid,
            geometry_ratios=ratios,
            properties=properties,
            feed_volumetric_concentration=feed_volumetric_concentration,
            hydrocyclone_cls=hydrocyclone_cls,
            particle_diameters=particle_diameters,
            k=k,
            n=n,
        )

        feed_volumetric_flow_rate_lpm = feed_volumetric_flow_rate_grid * 1000 * 60
        hydrocyclone_diameter_mm = hydrocyclone_diameter_grid * 1000

        # Отсечной размер
        ax = axes[row, CUT_SIZE_COL]
        contour = ax.contour(feed_volumetric_flow_rate_lpm,
                             hydrocyclone_diameter_mm,
                             reduced_cut_size,
                             levels=np.arange(10, 40+5, 5))
        ax.clabel(contour, inline=True, fontsize=8)
        ax.set_xlabel('$Q$, л/мин')
        ax.set_ylabel('$D_c$, мм')
        if row == TITLE_ROW:
            ax.set_title("Отсечной размер $d_{50}'$, мкм")
        ax.text(-0.3, 0.5, hydrocyclone.name, transform=ax.transAxes,
                fontsize=12, fontweight='bold', rotation=90, va='center')
        ax.grid(True, alpha=0.3)

        # Соотношение потоков воды
        ax = axes[row, WATER_RATIO_COL]
        contour = ax.contour(feed_volumetric_flow_rate_lpm,
                             hydrocyclone_diameter_mm, water_flow_ratio,
                             levels=10)
        ax.clabel(contour, inline=True, fontsize=8)
        ax.set_xlabel('$Q$, л/мин')
        ax.set_ylabel('$D_c$, мм')
        if row == TITLE_ROW:
            ax.set_title('Соотношение потоков воды $R_w$')
        ax.grid(True, alpha=0.3)

        # Приведённая эффективность
        ax = axes[row, EFFICIENCY_COL]
        contour = ax.contour(feed_volumetric_flow_rate_lpm,
                             hydrocyclone_diameter_mm,
                             reduced_total_efficiency,
                             levels=np.arange(50, 100+5, 5))
        ax.clabel(contour, inline=True, fontsize=8)
        ax.set_xlabel('$Q$, л/мин')
        ax.set_ylabel('$D_c$, мм')
        if row == TITLE_ROW:
            ax.set_title("Приведённая эффективность $E_T'$, %")
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f'Характеристики гидроциклонов '
        f'($C_v = {feed_volumetric_concentration}$)',
        fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_contour_graphs()
