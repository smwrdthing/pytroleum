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

    # Дополнительные контрольные точки
    h01 = 1.7
    h12 = 1.0
    h20 = 0.3

    # Давления в дополнительных точках
    p00 = p0
    p01 = p0+rho_oil * g * (h1 - h12)
    p20 = p0+rho_oil * g * (h1 - h2) + rho_water * g * (h2 - h20)

    levels = np.array([h0, h1, h2])
    pressures = np.array([p0, p1, p2])

    # Создаем массив высот для плавного графика
    heights = np.linspace(0, h0, 500)

    # pressure_values = np.array([_compute_pressure_for(
    # h, levels, pressures) for h in heights])

    # NOTE уходим от цикла :
    # NOTE внутри _compute_pressure_for вызывается np.interp, эта функция работает с numpy
    # NOTE массивами "из коробки", поэтому heights можно сразу передавать так, pylance
    # NOTE будет ругаться, потому что у _compute_pressure_for в сигнатуре не прописана
    # NOTE возможность передачи массивов, но код запустится
    # NOTE
    # NOTE Чтобы убрать сообщние об ошибке можно на линии после кода написать
    # NOTE "# type: ignore", но по-хорошему нужно добавить сигнатуру вызова с массивами
    # NOTE через overload, если функция умеет с ними работать (пока можем оставить так)
    pressure_values = _compute_pressure_for(
        heights, levels, pressures)  # type: ignore

    # NOTE _compute_pressure_for должна работать с массивами тоже, так что по идее
    # NOTE можно обойтись без цикла

    # Точки интерполяции (граничные)
    interp_heights = np.array([0, h2, h1, h0])
    interp_pressures = np.array([p2, p1, p0, p0])

    # Дополнительные контрольные точки
    control_heights = np.array([h20, h12, h01])
    control_pressures = np.array([p20, p01, p00])

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
    ax.plot(control_pressures / 1e5, control_heights * 1000, 'ro',
            markersize=8, markerfacecolor='none', markeredgewidth=2,
            label='Контрольные точки', zorder=6)

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
