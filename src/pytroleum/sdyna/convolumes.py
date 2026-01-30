from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import CoolProp.constants as CoolConst
from scipy.constants import g
from pytroleum import meter
from pytroleum.tdyna.eos import factory_eos

from typing import Callable, overload
from numpy.typing import NDArray
from numpy import float64
from pytroleum.sdyna.interfaces import Conductor
from pytroleum.sdyna.opdata import StateData
from pytroleum.sdyna.conductors import Fixed

_NUMBER_OF_GRADUATION_POINTS = 500


class ControlVolume(ABC):

    def __init__(self) -> None:
        self.outlets: list[Conductor] = []
        self.inlets: list[Conductor] = []
        self.volume: float | float64 = np.inf
        self._net_mass_flow = np.zeros(1)
        self._net_energy_flow = np.zeros(1)
        self.state: StateData

    # TODO Introduce custom decorator for iterable inputs
    def connect_inlet(self, conductor: Conductor) -> None:
        """Adds new conductor in the inlets list of control volume."""
        if conductor not in self.inlets:
            conductor.sink = self
            self.inlets.append(conductor)

    def connect_outlet(self, conductor: Conductor) -> None:
        """Adds new conductor in the outlets list of control volume."""
        if conductor not in self.outlets:
            conductor.source = self
            self.outlets.append(conductor)

    # NOTE methods are listed in the inteded execution order for advancement

    def compute_fluid_energy_specific(self) -> None:
        """Computes specific energy from mass and energy."""
        self.state.energy_specific = self.state.energy/self.state.mass

    # NOTE :
    # All secondary parameters are computed with finite-difference approximation of
    # respective differentials with additional reasonable assumptions for preformance and
    # robustness reasons.
    #
    # In this case eos contains parameters from previous time step and aids computations
    # for custom state object that should hold new values. We should update eos only at
    # the very end of time step processing.

    def compute_fluid_temperature(self) -> None:
        """Computes temperature from specific energy."""
        temperature = []
        for eos, energy_specific in zip(self.state.equation,
                                        self.state.energy_specific):
            partial_derivative_energy = eos.first_partial_deriv(
                CoolConst.iUmass, CoolConst.iT, CoolConst.iP)
            fluid_new_temperature = (
                (energy_specific-eos.umass())/partial_derivative_energy + eos.T())
            temperature.append(fluid_new_temperature)
        self.state.temperature[:] = temperature

    def compute_liquid_density(self) -> None:
        """Computes density of liquid from temperature."""
        liquid_density = []
        for eos, new_T in zip(
                self.state.equation[1:], self.state.temperature[1:]):
            density_partial_derivative = eos.first_partial_deriv(
                CoolConst.iDmass, CoolConst.iT, CoolConst.iP)
            fluid_new_density = eos.rhomass()+density_partial_derivative*(new_T-eos.T())
            liquid_density.append(fluid_new_density)
        self.state.density[1:] = liquid_density

    def compute_fluid_volume(self) -> None:
        """Computes volume of liquid from mass and density, vapor volume is computed as
        from vacant space of control volume"""
        self.state.volume[1:] = self.state.mass[1:]/self.state.density[1:]
        self.state.volume[0] = self.volume - np.sum(self.state.volume[1:])

    def compute_vapor_density(self) -> None:
        """Computes density of vapor from mass and volume"""
        self.state.density[0] = self.state.mass[0]/self.state.volume[0]

    def compute_vapor_pressure(self) -> None:
        """Computes vapor pressure from density and temperature"""
        vapor_eos = self.state.equation[0]

        pressure_partial_derivative_wrt_temperatrue = (
            vapor_eos.first_partial_deriv(CoolConst.iP, CoolConst.iT, CoolConst.iDmass))
        pressure_partial_derivative_wrt_density = (
            vapor_eos.first_partial_deriv(CoolConst.iP, CoolConst.iDmass, CoolConst.iT))

        change_in_density = self.state.density[0]-vapor_eos.rhomass()
        change_in_temperature = self.state.temperature[0] - vapor_eos.T()

        new_vapor_pressure = vapor_eos.p() + (
            pressure_partial_derivative_wrt_temperatrue*change_in_temperature +
            pressure_partial_derivative_wrt_density*change_in_density)

        self.state.pressure[0] = new_vapor_pressure

    def update_equations_of_state(self):
        """Updates equations of state with pressure and temperature values"""
        for eos, new_T in zip(self.state.equation, self.state.temperature):
            eos.update(CoolConst.PT_INPUTS, self.state.pressure[0], new_T)

    def reset_flow_rates(self):
        """Assigns zero for net flow in control volume"""
        self._net_mass_flow = np.zeros_like(self._net_mass_flow)
        self._net_energy_flow = np.zeros_like(self._net_energy_flow)

    @abstractmethod
    def advance(self) -> None:
        """Resolves control volume state at given time step"""
        return


class Atmosphere(ControlVolume):

    _STANDARD_TEMPERATURE = 20+273.15
    _STANDARD_PRESSURE = 101_330

    def __init__(self) -> None:
        super().__init__()

        # Possible TODO :
        # Implement selector-method to set standard temperature and pressure for
        # specified STP refernce
        _eos_air = factory_eos({"air": 1}, with_state=(
            CoolConst.PT_INPUTS, self._STANDARD_PRESSURE, self._STANDARD_TEMPERATURE))

        self.state = StateData(
            equation=[_eos_air],
            pressure=np.array([self._STANDARD_PRESSURE]),
            temperature=np.array([self._STANDARD_TEMPERATURE]),
            density=np.array([_eos_air.rhomass()]),
            energy_specific=np.array([_eos_air.umass()]),
            dynamic_viscosity=np.array([_eos_air.viscosity()]),
            thermal_conductivity=np.array([_eos_air.conductivity()]),
            mass=np.array([np.inf]),
            energy=np.array([np.inf]),
            level=np.array([np.inf]),
            volume=np.array([np.inf]))

    def advance(self) -> None:
        return


class SectionHorizontal(ControlVolume):

    @overload
    def __init__(
        self,
        length_left_semiaxis: float | float64,
        length_cylinder: float | float64,
        length_right_semiaxis: float | float64,
        diameter: float | float64,
        volume_modificator: Callable[[float | float64], float | float64]
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        length_left_semiaxis: float | float64,
        length_cylinder: float | float64,
        length_right_semiaxis: float | float64,
        diameter: float | float64,
        volume_modificator: Callable[[NDArray[float64]], NDArray[float64]]
    ) -> None:
        ...

    def __init__(
            self,
            length_left_semiaxis,
            length_cylinder,
            length_right_semiaxis,
            diameter,
            volume_modificator
    ):
        super().__init__()

        self.length_left_semiaxis = length_left_semiaxis
        self.length_cylinder = length_cylinder
        self.length_right_semiaxis = length_right_semiaxis
        self.diameter = diameter
        self.volume_modificator = volume_modificator

        self._volume_pure = self.compute_volume_with_level(diameter)
        self.volume = self._volume_pure + self.volume_modificator(diameter)

        self._level_graduated, self._volume_graduated = meter.graduate(
            0, diameter, self.compute_volume_with_level, _NUMBER_OF_GRADUATION_POINTS)
        self._volume_graduated = (
            self._volume_graduated + self.volume_modificator(self._level_graduated))

    @overload
    def compute_volume_with_level(self, level: float | float64) -> float | float64:
        ...

    @overload
    def compute_volume_with_level(self, level: NDArray[float64]) -> NDArray[float64]:
        ...

    def compute_volume_with_level(self, level):
        """Returns volume corresponding to provided level in horizontal section"""
        volume_pure = meter.volume_section_horiz_ellipses(
            self.length_left_semiaxis,
            self.length_cylinder,
            self.length_right_semiaxis,
            self.diameter, level)
        volume_modificator = self.volume_modificator(level)
        volume = volume_pure+volume_modificator
        return volume

    @overload
    def compute_level_with_volume(self, volume: float | float64) -> float | float64:
        ...

    @overload
    def compute_level_with_volume(self, volume: NDArray[float64]) -> NDArray[float64]:
        ...

    def compute_level_with_volume(self, volume):
        """Returns level corresponding to provided volume in horizontal section"""
        return meter.inverse_graduate(
            volume, self._level_graduated, self._volume_graduated)

    def compute_fluid_level(self) -> None:
        """Computes fluid level from fluid volume"""
        self.state.level = self.compute_level_with_volume(
            np.flip(np.cumsum(np.flip(self.state.volume))))

    def compute_liquid_pressure(self) -> None:
        """Computes liquid's pressure profile for layered multiphase situation"""
        layers_thickness = -np.diff(self.state.level[1:], append=0)
        individual_hydrostatic_head = self.state.density[1:]*g*layers_thickness
        cumulative_hydrostatic_head = np.cumsum(individual_hydrostatic_head)
        self.state.pressure[1:] = (
            self.state.pressure[0] + cumulative_hydrostatic_head)

    def compute_secondary_parameters(self) -> None:
        """Determines values of state parameters from masses and energies"""
        self.compute_fluid_energy_specific()
        self.compute_fluid_temperature()
        self.compute_liquid_density()
        self.compute_fluid_volume()
        self.compute_vapor_density()
        self.compute_vapor_pressure()
        self.compute_fluid_level()
        self.compute_liquid_pressure()

    def advance(self) -> None:
        self.reset_flow_rates()
        self.compute_secondary_parameters()
        self.update_equations_of_state()


class SectionVertical(ControlVolume):

    @overload
    def __init__(
            self,
            diameter: float | float64,
            height: float | float64,
            volume_modificator: Callable[[float | float64], float | float64]
    ) -> None:
        ...

    @overload
    def __init__(
            self,
            diameter: float | float64,
            height: float | float64,
            volume_modificator: Callable[[NDArray[float64]], NDArray[float64]]
    ) -> None:
        ...

    def __init__(
            self,
            diameter,
            height,
            volume_modificator
    ):
        super().__init__()

        self.diameter = diameter
        self.height = height
        self.volume_modificator = volume_modificator

        self._volume_pure = self.compute_volume_with_level(height)
        self.volume = self._volume_pure + self.volume_modificator(height)

        self._level_graduated, self._volume_graduated = meter.graduate(
            0, height, self.compute_volume_with_level, _NUMBER_OF_GRADUATION_POINTS)
        self._volume_graduated = (
            self._volume_graduated + self.volume_modificator(self._level_graduated))

    @overload
    def compute_volume_with_level(self, level: float | float64) -> float | float64:
        ...

    @overload
    def compute_volume_with_level(self, level: NDArray[float64]) -> NDArray[float64]:
        ...

    def compute_volume_with_level(self, level):
        """Returns volume corresponding to provided level in vertical cylinder"""
        # NOTE для заданного диаметра радиус и площадь не меняются от итерации к итерации,
        # NOTE можно сделать их атрибутами и посчитать один раз в __init__, чтобы не
        # NOTE считать их каждый раз
        radius = self.diameter / 2
        area = np.pi * radius ** 2

        # NOTE этот валидатор лучше убрать, потому что он "нечестен" с пользователем,
        # NOTE ситуация, когда нам передаётся уровень меньше нуля или больше заданной
        # NOTE высоты — это ошибка пользователя, и он должен о ней узнать через
        # NOTE сообщение об ошибке, а не тихо получить какое-то значение
        # NOTE
        # NOTE Здесь скорее подошёл бы exception с корректным сообщением, но делать его
        # NOTE тут тоже не следует, обработка исключений процесс медленный, в этом случае
        # NOTE передача корректного значения уровня — забота пользователя
        if isinstance(level, np.ndarray):
            level_clipped = np.clip(level, 0, self.height)
        else:
            level_clipped = min(max(level, 0), self.height)

        volume_pure = area * level_clipped
        volume_modificator = self.volume_modificator(level_clipped)
        volume = volume_pure + volume_modificator
        return volume

    @overload
    def compute_level_with_volume(self, volume: float | float64) -> float | float64:
        ...

    @overload
    def compute_level_with_volume(self, volume: NDArray[float64]) -> NDArray[float64]:
        ...

    def compute_level_with_volume(self, volume):
        """Returns level corresponding to provided volume in vertical cylinder"""

        # NOTE общий алгоритм inverse_graduate справляется, но он использует интерполяцию
        # NOTE по массиву заранее вычисленных значений V(h), это нужно было для
        # NOTE горизонтального сосуда, потому что для него из формулы для объёма нельзя
        # NOTE выразить уровень, у вертикального сосуда так сделать можно, в таком случае
        # NOTE лучше использовать формулу

        return meter.inverse_graduate(
            volume, self._level_graduated, self._volume_graduated)

    def compute_fluid_level(self) -> None:
        """Computes fluid level from fluid volume"""
        self.state.level = self.compute_level_with_volume(
            np.flip(np.cumsum(np.flip(self.state.volume))))

    def compute_liquid_pressure(self) -> None:
        """Computes liquid's pressure profile for layered multiphase situation"""
        layers_thickness = -np.diff(self.state.level[1:], append=0)
        individual_hydrostatic_head = self.state.density[1:]*g*layers_thickness
        cumulative_hydrostatic_head = np.cumsum(individual_hydrostatic_head)
        self.state.pressure[1:] = (
            self.state.pressure[0] + cumulative_hydrostatic_head)

    def compute_secondary_parameters(self) -> None:
        """Determines values of state parameters from masses and energies"""
        self.compute_fluid_energy_specific()
        self.compute_fluid_temperature()
        self.compute_liquid_density()
        self.compute_fluid_volume()
        self.compute_vapor_density()
        self.compute_vapor_pressure()
        self.compute_fluid_level()
        self.compute_liquid_pressure()

    def advance(self) -> None:
        self.reset_flow_rates()
        self.compute_secondary_parameters()
        self.update_equations_of_state()


if __name__ == "__main__":
    from pytroleum.sdyna.opdata import factory_state, factory_flow
    import matplotlib.pyplot as plt

    # 1. СОЗДАЕМ ВЕРТИКАЛЬНЫЙ РЕЗЕРВУАР
    vessel = SectionVertical(1.0, 1.2, lambda h: 0)  # диаметр 1 м, высота 2 м

    # 2. ТЕРМОДИНАМИЧЕСКОЕ СОСТОЯНИЕ
    thermodynamic_state = (CoolConst.PT_INPUTS, 1e5, 300)  # 100 кПа, 300 К

    # 3. ИНИЦИАЛИЗИРУЕМ СОСТОЯНИЕ РЕЗЕРВУАРА (две фазы: воздух + вода)
    vessel.state = factory_state(
        [factory_eos({"air": 1}, with_state=thermodynamic_state),      # воздух
         factory_eos({"water": 1}, with_state=thermodynamic_state)],   # вода
        vessel.compute_volume_with_level,  # функция объема
        # давления: воздух=100 кПа, вода=100 кПа
        np.array([1e5, 1e5]),
        np.array([300, 300]),            # температуры: 300 К
        np.array([2.0, 0.3]))            # уровни: воздух=2 м, вода=0.5 м

    # 4. СОЗДАЕМ FIXED
    inlet = Fixed([0, 1], sink=vessel)

    # 5. НАСТРАИВАЕМ ПАРАМЕТРЫ ПОТОКА ЧЕРЕЗ FIXED
    inlet.flow = factory_flow(
        [factory_eos({"air": 1}, with_state=thermodynamic_state),
         factory_eos({"water": 1}, with_state=thermodynamic_state)],
        np.array([1.5e5, 1.5e5]),          # давления на входе: 150 кПа
        np.array([300, 300]),              # температуры: 300 К
        np.pi*(0.05)**2/4,
        0.9,
        np.array([0, 1], dtype=np.float64))

    # 6. ВЫЧИСЛЯЕМ РАСХОД ИЗ ПАРАМЕТРОВ FIXED
    water_density = 1000  # кг/м³ (плотность воды при 300К)
    pipe_area = np.pi*(0.05)**2/4  # м²
    flow_velocity = 0.8  # м/с
    mass_flow_rate = water_density * flow_velocity * pipe_area  # кг/с

    # Объемный расход (м³/с)
    volume_flow_rate = mass_flow_rate / water_density

    # 7.ЗАПОЛНЕНИЕ РЕЗЕРВУАРА

    # NOTE здесь нужно использовать код из network, в которому уже организован решатель
    # NOTE
    # NOTE надо через наследование создать класс, который будет описывать конкретно эту
    # NOTE систему, всё подключить и использовать advance пример использования должен был
    # NOTE остаться в network.py

    dt = 1.0  # шаг времени, с
    t = [0.0]  # массив времени
    h_water = [vessel.state.level[1]]  # массив уровней воды

    for step in range(450):

        # NOTE с учётом заметки выше цикл решателя будет выглядеть очень просто
        # NOTE >> for step in steps:
        # NOTE >>     network.advance()
        # NOTE где network — это объект класса, в котором описана вся система

        # Обновляем объем воды
        current_volume_water = vessel.state.volume[1]
        new_volume_water = current_volume_water + volume_flow_rate * dt

        # Ограничиваем максимальным объемом резервуара
        total_volume = vessel.volume
        if new_volume_water > total_volume:
            new_volume_water = total_volume
            print(
                f"Резервуар заполнился на шаге {step} (t = {t[-1]+dt:.1f} с)")

        # Обновляем состояние резервуара
        vessel.state.volume[1] = new_volume_water  # объем воды
        vessel.state.volume[0] = total_volume - \
            new_volume_water  # объем воздуха

        # Пересчитываем уровни жидкостей
        vessel.compute_fluid_level()

        # Сохраняем данные для графика
        t.append(t[-1] + dt)
        h_water.append(vessel.state.level[1])

        # Если резервуар полный, останавливаем
        if new_volume_water >= total_volume:
            break

    # 8.Аналитическое решение
    D = 1.0  # диаметр резервуара, м
    H = 1.2  # высота резервуара, м
    h0 = 0.3
    A_vessel = np.pi * (D/2)**2  # площадь резервуара, м²
    t_fill = (H - h0) * A_vessel * water_density / \
        mass_flow_rate  # время заполнения, с

    def h_analytical(t):
        return np.minimum(h0 + (mass_flow_rate / (water_density * A_vessel)) * t, H)

    # Создаем массив времени для аналитического решения
    t_an = np.linspace(0, t_fill * 1.1, 100)
    h_an = h_analytical(t_an)

    # 9. ПОСТРОЕНИЕ ГРАФИКА h от t
    plt.figure(figsize=(10, 6))

    plt.plot(t, h_water, 'b-', linewidth=2, label='Численное решение')
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
