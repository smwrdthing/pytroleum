import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List

from geometry import GeometryParameters
from physics import PhysicalParameters, OperatingParameters
from model import MODEL_PARAMS, calculate_hydrocyclone_parameters
from model import calculate_grade_efficiency
from distributions import feed_cdf_rosin_rammler, feed_cdf_rosin_rammler_derivative
from efficiency import calculate_reduced_total_efficiency
from efficiency import calculate_total_efficiency_from_reduced


def plot_hydrocyclone_analysis():
    """Построение графиков анализа гидроциклонов"""

    # Конфигурации гидроциклонов
    @dataclass
    class HydrocycloneConfig:
        name: str
        geometry: GeometryParameters
        htype: str

    hydrocyclone_configs: List[HydrocycloneConfig] = [
        HydrocycloneConfig(
            name='Rietema',
            geometry=GeometryParameters(
                Dc=0.075,      # 75 мм
                Di=0.021,      # 21 мм
                Do=0.025,      # 25 мм
                Du=0.0125,     # 12.5 мм
                L=0.375,       # 375 мм
                l_vortex=0.075  # 75 мм
            ),
            htype='rietema'
        ),
        HydrocycloneConfig(
            name='Bradley',
            geometry=GeometryParameters(
                Dc=0.075,      # 75 мм
                Di=0.015,      # 15 мм
                Do=0.020,      # 20 мм
                Du=0.010,      # 10 мм
                L=0.180,       # 180 мм
                l_vortex=0.060  # 60 мм
            ),
            htype='bradley'
        ),
        HydrocycloneConfig(
            name='Demco',
            geometry=GeometryParameters(
                Dc=0.075,      # 75 мм
                Di=0.024,      # 24 мм
                Do=0.028,      # 28 мм
                Du=0.014,      # 14 мм
                L=0.240,       # 240 мм
                l_vortex=0.080  # 80 мм
            ),
            htype='demco'
        )
    ]

    # Физические параметры
    physical = PhysicalParameters(
        mu=0.001,    # Па·с, вязкость воды
        rho=1000,    # кг/м³, плотность воды
        rhos=2650    # кг/м³, плотность песка
    )

    # Рабочие параметры
    operating = OperatingParameters(
        Q=0.001,     # м³/с, расход (60 л/мин)
        Cv=0.05      # 5% объёмная концентрация
    )

    # Параметры распределения Розин-Раммлер
    k = 50e-6        # 50 мкм, характерный размер
    n = 1.5          # параметр формы

    # Диапазон размеров частиц для всех графиков
    d = np.linspace(0, 1e-3, 500)

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))

    # Для каждого типа циклона
    for row, config in enumerate(hydrocyclone_configs):
        # Расчёт параметров гидроциклона
        results = calculate_hydrocyclone_parameters(
            geometry=config.geometry,
            physical=physical,
            operating=operating,
            model_params=MODEL_PARAMS[config.htype]
        )

        d50_prime = results['d50_prime']
        Rw = results['Rw']
        m = results['m']
        alpha = results['alpha']

        # Нормированный размер
        d_norm = d / d50_prime

        # 1. График y(d) - кумулятивное распределение от d/d50'
        ax = axes[row, 0]
        y = feed_cdf_rosin_rammler(d, k, n)
        ax.plot(d_norm, y, 'b-', linewidth=2, label='y(d)')
        ax.set_xlabel("$d/d_{50}'$", fontsize=10)
        ax.set_ylabel('y(d)', fontsize=10)
        ax.set_xlim([0, 6])
        ax.set_ylim([0, 1.2])
        ax.grid(True, alpha=0.3)
        ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5, linewidth=2)
        # Текст в правом нижнем углу
        ax.text(0.95, 0.05, f'{config.name}\nk={k*1e6:.0f} мкм\nn={n:.1f}',
                transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        # 2. График dy/dd - функция плотности распределения от d/d50'
        ax = axes[row, 1]
        dy_dd = feed_cdf_rosin_rammler_derivative(d, k, n)
        # Масштабируем производную для отображения
        dy_dd_scaled = dy_dd * d50_prime  # чтобы размерность была 1/(d/d50')
        ax.plot(d_norm, dy_dd_scaled, 'b-', linewidth=2, label='dy/d(d/d50\')')
        ax.set_xlabel("$d/d_{50}'$", fontsize=10)
        ax.set_ylabel("$dy/d(d/d_{50}')$", fontsize=10)
        ax.set_xlim([0, 6])
        ax.set_ylim([0, 0.55])
        ax.grid(True, alpha=0.3)
        ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5, linewidth=2)
        # Текст в правом верхнем углу
        ax.text(0.95, 0.95, f'{config.name}\nk={k*1e6:.0f} мкм\nn={n:.1f}',
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        # 3. График G'(d) и G(d) - приведенная и обычная вероятность уноса
        ax = axes[row, 2]
        G_prime = calculate_grade_efficiency(d, d50_prime, 'plitt', m, alpha)
        G = G_prime * (1 - Rw) + Rw

        ax.plot(d_norm, G_prime, 'r-', linewidth=2, label="$G'(d/d_{50}')$")
        ax.plot(d_norm, G, 'b--', linewidth=2, label="$G(d/d_{50}')$")
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=2)
        ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5, linewidth=2)
        ax.set_xlabel("$d/d_{50}'$", fontsize=10)
        ax.set_ylabel("$G(d)$, $G'(d)$", fontsize=10)
        ax.set_xlim([0, 6])
        ax.set_ylim([0, 1.2])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        # Текст в правом нижнем углу
        ax.text(0.95, 0.05,
                f'{config.name}\n$d_{{50}}\'$={d50_prime*1e6:.1f} мкм\n'
                f'm={m:.2f}\n$R_w$={Rw:.3f}',
                transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        # Расчёт приведённой полной эффективности
        E_T_prime = calculate_reduced_total_efficiency(d, k, n, G_prime)

        # Расчёт полной эффективности
        E_T = calculate_total_efficiency_from_reduced(E_T_prime, Rw)
        print(f"=== {config.name} ===")
        print(f"  E_T' = {E_T_prime:.4f} ({E_T_prime*100:.1f}%)")
        print(f"  E_T  = {E_T:.4f} ({E_T*100:.1f}%)")
        print()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_hydrocyclone_analysis()
