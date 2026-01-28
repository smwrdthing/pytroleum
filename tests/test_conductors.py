import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from numpy import float64
from scipy.constants import g
from pytroleum.sdyna.conductors import _compute_pressure_for


def visualize_pressure_profile():

    h0 = 2.0
    h1 = 1.4
    h2 = 0.6
    rho_water = 1000.0
    rho_oil = 850.0

    # Давление наверху
    p0 = 0.1

    # Расчет давлений
    p1 = p0 + rho_oil * g * (h1 - h2) / 1e6
    p2 = p0 + rho_oil * g * (h1 - h2) / 1e6 + rho_water * g * (h2 - 0) / 1e6

    levels = np.array([h0, h1, h2])
    pressures = np.array([p0, p1, p2])

    # Создаем массив высот для плавного графика
    heights = np.linspace(0, h0, 500)
    pressure_values = [_compute_pressure_for(
        h, levels, pressures) for h in heights]

    # Точки интерполяции
    interp_heights = [0, h2, h1, h0]
    interp_pressures = [p2, p1, p0, p0]

    # Дополнительные контрольные точки
    test_heights = [0.3, 1.0, 1.7]
    test_pressures = [_compute_pressure_for(
        h, levels, pressures) for h in test_heights]

    # Создаем график
    fig, ax = plt.subplots(figsize=(10, 8))

    # Области фаз
    ax.axhspan(h1, h0, alpha=0.2, color='lightblue', label='Газовая зона')
    ax.axhspan(h2, h1, alpha=0.25, color='gold', label='Нефтяная зона')
    ax.axhspan(0, h2, alpha=0.2, color='lightgreen', label='Водная зона')

    # Основной профиль давления
    ax.plot(pressure_values, heights, 'k-', linewidth=2,
            label='Профиль давления', alpha=0.8, zorder=3)

    # Точки интерполяции (граничные)
    ax.plot(interp_pressures, interp_heights, 'ko', markersize=6,
            label='Граничные точки', zorder=5, markerfacecolor='black')

    # Дополнительные контрольные точки
    ax.plot(test_pressures, test_heights, 'rs', markersize=6,
            label='Контрольные точки', zorder=5)

    # Настройки графика
    ax.set_xlabel('Давление (МПа)', fontsize=12)
    ax.set_ylabel('Высота (м)', fontsize=12)

    # Сетка
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Легенда
    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_pressure_profile()
