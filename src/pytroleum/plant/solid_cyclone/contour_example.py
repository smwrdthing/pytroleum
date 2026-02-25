"""
Контурные графики характеристик гидроциклонов.

Пример использования модуля hydrocyclone для построения контурных
графиков зависимости ключевых характеристик от диаметра циклона и
расхода жидкости.
"""

import numpy as np
import matplotlib.pyplot as plt

from hydrocyclone_new import (  # импорт из основного модуля расчёта гидроциклона
    GeometryParameters,                  # dataclass с диаметрами и длинами циклона
    PhysicalProperties,                  # dataclass с mu, rho, rhos
    BaseHydrocyclone,                    # абстрактный базовый класс
    RietemaHydrocyclone,                 # конкретная модель Rietema
    BradleyHydrocyclone,                 # конкретная модель Bradley
    DemcoHydrocyclone,                   # конкретная модель Demco
    build_standard_configs,              # строит три стандартные конфигурации
    calculate_reduced_grade_efficiency,  # вычисляет G'(d)
    calculate_reduced_total_efficiency,  # вычисляет E'_T
)

# NOTE перенести в examples репозитория

# Индексы столбцов на сетке графиков
CUT_SIZE_COL = 0     # столбец 0 — контурный график отсечного размера
WATER_RATIO_COL = 1  # столбец 1 — контурный график Rw
EFFICIENCY_COL = 2   # столбец 2 — контурный график приведённой эффективности E'_T

# заголовки столбцов ставятся только в строке 0 (Rietema), остальные строки без заголовков
TITLE_ROW = 0

GRID_SIZE = 50  # разрешение расчётной сетки: 50×50=2500 точек
# NOTE для сеток в целом можно заводить две переменные, одну для количества строк,
# NOTE другую для количества столбцов, это может быть полезно, если в каком-то направлении
# NOTE потребуется сделать сетку гуще (разместить больше значений)

V_IN_MIN = 1.0  # минимальная скорость во входном патрубке, м/с
V_IN_MAX = 3.0  # максимальная скорость во входном патрубке, м/с


def _compute_for_point(
    Dc: float,                         # диаметр корпуса для данной точки сетки, м
    Q: float,                          # объёмный расход для данной точки сетки, м³/с
    geometry_ratios: dict,  # словарь пропорций {'Di/Dc':..., 'Do/Dc':...,}
    properties: PhysicalProperties,    # физические свойства жидкости и твёрдой фазы
    Cv: float,                         # объёмная концентрация твёрдых частиц
    hydrocyclone_cls: type,  # NOTE см. вторую заметку в функции
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
        Dc=Dc,
        Di=Dc * geometry_ratios['Di/Dc'],
        Do=Dc * geometry_ratios['Do/Dc'],
        Du=Dc * geometry_ratios['Du/Dc'],
        L=Dc * geometry_ratios['L/Dc'],
        l_vortex=Dc * geometry_ratios['l/Dc'],
        theta=geometry_ratios['theta'],
    )
    # создаёт экземпляр нужного подкласса; имя '' — не нужно для расчёта
    hydrocyclone: BaseHydrocyclone = hydrocyclone_cls('', geometry)
    # рассчитывает все параметры при заданном расходе Q
    results = hydrocyclone.calculate_from_flow_rate(properties, Q, Cv)

    reduced_grade = calculate_reduced_grade_efficiency(
        particle_diameters,
        results['reduced_cutoff_diameter'],
        'plitt',
        results['m'],
        results['alpha'],
    )
    efficiency = calculate_reduced_total_efficiency(
        particle_diameters, k, n, reduced_grade) * 100

    return results['reduced_cutoff_diameter'] * 1e6, results['Rw'], efficiency


_compute_vectorized = np.vectorize(
    _compute_for_point,
    excluded=[               # параметры из этого списка НЕ итерируются
        # — передаются как константы в каждый вызов
        'geometry_ratios',   # словарь пропорций одинаков для всей строки сетки
        'properties',        # физические свойства одинаковы для всей сетки
        'Cv',                # концентрация одинакова для всей сетки
        'hydrocyclone_cls',  # класс модели одинаков для всей строки сетки
        'particle_diameters',  # сетка частиц одинакова для всей сетки
        'k',                 # параметр Розин-Раммлера одинаков для всей сетки
        'n',                 # параметр Розин-Раммлера одинаков для всей сетки
    ],
)  # Dc и Q итерируются поэлементно по переданным массивам (Dc_grid, Q_grid)

# NOTE np.vectorize под капотом делает те же циклы на чистом Python и не избавляет от
# NOTE потенциальных проблем с производительностью, расчётные функции сами по себе должны
# NOTE быть нацелены на работу с массивами сопоставимых размеров, чтобы сказывалось
# NOTE преимущество от использования numpy


def plot_contour_graphs() -> None:
    # NOTE содержание этой функции можно закинуть в examples репозитория
    """Контурные графики для трёх типов гидроциклонов."""
    Dc_range = np.linspace(0.01, 0.08, GRID_SIZE)

    properties = PhysicalProperties(mu=0.001, rho=1000, rhos=2650)
    Cv = 0.05          # объёмная концентрация твёрдых частиц 5%
    k = 50e-6          # параметр Розин-Раммлера: 50 мкм
    n = 1.5            # параметр Розин-Раммлера
    particle_diameters = np.linspace(0, 1e-3, 500)

    # Берём пропорции из эталонных конфигураций
    # создаёт три конфигурации при Dc=0.01 м; нас интересуют только их пропорции
    reference_configs = build_standard_configs(Dc_range[0])
    # список классов в том же порядке, что reference_configs
    hydrocyclone_classes = [RietemaHydrocyclone,
                            BradleyHydrocyclone, DemcoHydrocyclone]

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))

    for row, (hydrocyclone, hydrocyclone_cls) in enumerate(
            zip(reference_configs, hydrocyclone_classes)):

        # словарь безразмерных пропорций {'Di/Dc': ..., ...}
        ratios = hydrocyclone.geometry.get_geometry_ratios()
        # добавляет угол θ в словарь; get_geometry_ratios() не включает его
        ratios['theta'] = hydrocyclone.geometry.theta

        # Диапазон расходов из ограничения скорости: Q = v * π·Di²/4, Di = Dc·(Di/Dc)
        Di_Dc = ratios['Di/Dc']
        Q_min = V_IN_MIN * np.pi * (Dc_range[0] * Di_Dc)**2 / 4
        Q_max = V_IN_MAX * np.pi * (Dc_range[-1] * Di_Dc)**2 / 4
        Q_range = np.linspace(Q_min, Q_max, GRID_SIZE)

        # прямоугольная сетка в пространстве (Q, Dc)
        Q_grid, Dc_grid = np.meshgrid(Q_range, Dc_range)

        cut_sizes, Rw_vals, efficiencies = _compute_vectorized(
            Dc_grid, Q_grid,
            geometry_ratios=ratios,
            properties=properties,
            Cv=Cv,
            hydrocyclone_cls=hydrocyclone_cls,
            particle_diameters=particle_diameters,
            k=k,
            n=n,
        )

        Q_lpm = Q_grid * 1000 * 60  # перевод м³/с → л/мин
        Dc_mm = Dc_grid * 1000

        # Отсечной размер
        ax = axes[row, CUT_SIZE_COL]
        contour = ax.contour(Q_lpm, Dc_mm, cut_sizes,
                             levels=10)
        ax.clabel(contour, inline=True, fontsize=8)
        ax.set_xlabel('$Q$, л/мин')
        ax.set_ylabel('$D_c$, мм')
        if row == TITLE_ROW:
            ax.set_title("Отсечной размер $d_{50}'$, мкм")
        ax.text(-0.3, 0.5, hydrocyclone.name, transform=ax.transAxes,
                fontsize=12, fontweight='bold', rotation=90, va='center')
        ax.grid(True, alpha=0.3)

        # Water ratio
        ax = axes[row, WATER_RATIO_COL]
        contour = ax.contour(Q_lpm, Dc_mm, Rw_vals, levels=10)
        ax.clabel(contour, inline=True, fontsize=8)
        ax.set_xlabel('$Q$, л/мин')
        ax.set_ylabel('$D_c$, мм')
        if row == TITLE_ROW:
            ax.set_title('Водное отношение $R_w$')
        ax.grid(True, alpha=0.3)

        # Приведённая эффективность
        ax = axes[row, EFFICIENCY_COL]
        contour = ax.contour(Q_lpm, Dc_mm, efficiencies,
                             levels=10)
        ax.clabel(contour, inline=True, fontsize=8)
        ax.set_xlabel('$Q$, л/мин')
        ax.set_ylabel('$D_c$, мм')
        if row == TITLE_ROW:
            ax.set_title("Приведённая эффективность $E_T'$, %")
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Характеристики гидроциклонов ($C_v = {Cv}$)', fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_contour_graphs()
