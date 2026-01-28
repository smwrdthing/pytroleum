import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from numpy import float64
from scipy.constants import g
from pytroleum.sdyna.conductors import _compute_pressure_for


def visualize_pressure_profile():
    h0 = 2.0  # м
    h1 = 1.4  # м
    h2 = 0.6  # м
    rho_water = 1000.0  # кг/м³
    rho_oil = 650.0  # кг/м³

    # Давление наверху
    p0 = 101325.0

    # Расчет давлений
    p1 = p0 + rho_oil * g * (h1 - h2)
    p2 = p0 + rho_oil * g * (h1 - h2) + rho_water * g * (h2 - 0)

    levels = np.array([h0, h1, h2])
    pressures = np.array([p0, p1, p2])

    # Создаем массив высот для плавного графика
    heights = np.linspace(0, h0, 500)

    pressure_values = np.array([_compute_pressure_for(
        h, levels, pressures) for h in heights])

    # Точки интерполяции
    interp_heights = np.array([0, h2, h1, h0])
    interp_pressures = np.array([p2, p1, p0, p0])

    # Дополнительные контрольные точки
    test_heights = np.array([0.3, 1.0, 1.7])
    test_pressures = np.array([_compute_pressure_for(
        h, levels, pressures) for h in test_heights])

    # NOTE в тестовых точках следует вычислять значения давления без использования
    # NOTE функции
    # NOTE
    # NOTE _compute_pressure_for должна работать с массивами тоже, так что по идее
    # NOTE можно обойтись без цикла

    # Создаем график
    fig, ax = plt.subplots(figsize=(10, 8))

    # Области фаз
    ax.axhspan(h1*1000, h0*1000, alpha=0.2,
               color='lightblue', label='Газовая зона')
    ax.axhspan(h2*1000, h1*1000, alpha=0.25,
               color='gold', label='Нефтяная зона')
    ax.axhspan(0, h2*1000, alpha=0.2, color='lightgreen', label='Водная зона')

    # Основной профиль давления (высота в мм, давление в бар)
    ax.plot(pressure_values / 1e5, heights * 1000, 'k-', linewidth=2,
            label='Профиль давления', alpha=0.8, zorder=3)

    # Точки интерполяции (граничные)
    ax.plot(interp_pressures / 1e5, interp_heights * 1000, 'ko', markersize=6,
            label='Граничные точки', zorder=5, markerfacecolor='black')

    # Дополнительные контрольные точки
    ax.plot(test_pressures / 1e5, test_heights * 1000, 'rs', markersize=6,
            label='Контрольные точки', zorder=5)

    # Настройки графика
    ax.set_xlabel('Давление, бар', fontsize=12)
    ax.set_ylabel('Высота, мм', fontsize=12)

    # Сетка
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Легенда
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, h0 * 1000)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_pressure_profile()
