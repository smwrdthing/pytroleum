import matplotlib.pyplot as plt
import numpy as np
import CoolProp.constants as CoolConst
from pytroleum.tdyna.eos import factory_eos
from pytroleum.sdyna.opdata import factory_state, factory_flow
from pytroleum.sdyna.network import DynamicNetwork
from pytroleum.sdyna.convolumes import SectionVertical
from pytroleum.sdyna.conductors import Fixed
import pytest

vessel = SectionVertical(1.0, 1.2, lambda h: 0)

thermodynamic_state = (CoolConst.PT_INPUTS, 1e5, 300)

vessel.state = factory_state(
    [factory_eos({"air": 1}, with_state=thermodynamic_state),
        factory_eos({"water": 1}, with_state=thermodynamic_state)],
    vessel.compute_volume_with_level,
    np.array([1e5, 1e5]),
    np.array([300, 300]),
    np.array([1.2, 0.3]))

inlet = Fixed([0, 1], sink=vessel)

water_density = 1000
pipe_diameter = 0.05
pipe_area = np.pi * (pipe_diameter / 2) ** 2
flow_velocity = 0.8
mass_flow_rate = water_density * flow_velocity * pipe_area

inlet.flow = factory_flow(
    [factory_eos({"air": 1}, with_state=thermodynamic_state),
        factory_eos({"water": 1}, with_state=thermodynamic_state)],
    np.array([1.5e5, 1.5e5]),
    np.array([300, 300]),
    pipe_area,
    0.9,
    np.array([0.0, mass_flow_rate], dtype=np.float64))


class DynamicVerticalvessel(DynamicNetwork):
    def __init__(self) -> None:
        super().__init__()

    def ode_system(self, t, y):
        return super().ode_system(t, y)

    def advance(self):
        return super().advance()


net = DynamicVerticalvessel()

net.add_control_volume(vessel)
net.add_conductor(inlet)
net.evaluate_size()

vessel.advance()

net.prepare_solver(0.1)

t = [0.0]
h = [vessel.state.level[1]]

for n in range(4500):
    net.advance()
    t.append(t[-1] + net.solver.time_step)
    h.append(vessel.state.level[1])

t = np.array(t)
h = np.array(h)

D = 1.0
H = 1.2
h0 = 0.3
A_vessel = np.pi * (D / 2) ** 2
t_fill = (H - h0) * A_vessel * water_density / mass_flow_rate


def h_analytical(t):
    return np.minimum(h0 + (mass_flow_rate / (water_density * A_vessel)) * t, H)


t_an = np.linspace(0, t_fill, 100)
h_an = h_analytical(t_an)


def test_error():
    """Тест на соответствие численного решения аналитическому."""
    h_analytical_t = h_analytical(t)
    error = np.mean(np.abs((h - h_analytical_t) / h_analytical_t) * 100)
    print(f"Средняя относительная ошибка: {error:.4f}%")

    threshold = 1.0  # 1%
    assert error < threshold


# ВЫЗЫВАЕМ ФУНКЦИЮ ДЛЯ ВЫВОДА СООБЩЕНИЯ
test_error()

if __name__ == "__main__":
    plt.figure(figsize=(10, 6))

    plt.plot(t, h, 'b-', linewidth=2, label='Численное решение')
    plt.plot(t_an, h_an, 'r--', linewidth=2, label='Аналитическое решение')

    plt.xlabel("Время, с", fontsize=12)
    plt.ylabel("Уровень воды, м", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=vessel.height, color='black', linestyle='--',
                alpha=0.7, label=f'Макс. уровень: {vessel.height} м')
    plt.legend(loc='lower right')
    plt.ylim(0, 1.3)
    plt.xlim(0, 450)
    plt.show()
