# Control volumes here
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import CoolProp.constants as CoolConst
from scipy.constants import g
from pytroleum import meter
from pytroleum.tdyna.eos import factory_eos

from typing import TYPE_CHECKING, Callable, overload
from numpy.typing import NDArray
from numpy import float64
from pytroleum.sdyna.interfaces import Conductor
from pytroleum.sdyna.opdata import StateData


class ControlVolume(ABC):

    def __init__(self) -> None:
        self.outlets: list[Conductor] = []
        self.inlets: list[Conductor] = []
        self.volume: float | float64 = np.inf
        self.net_mass_flow = np.zeros(1)
        self.net_energy_flow = np.zeros(1)
        self.state: StateData

    # TODO Introduce custom decorator for iterable inputs
    def connect_inlet(self, conductor: Conductor) -> None:
        if conductor not in self.inlets:
            conductor.sink = self
            self.inlets.append(conductor)

    def connect_outlet(self, conductor: Conductor) -> None:
        if conductor not in self.outlets:
            conductor.source = self
            self.outlets.append(conductor)

    # NOTE methods are listed in the inteded execution order for advancement

    def compute_fluid_energy_specific(self) -> None:
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
        temperature = []
        for eos, energy_specific in zip(self.state.equation_of_state,
                                        self.state.energy_specific):
            partial_derivative_energy = eos.first_partial_deriv(
                CoolConst.iUmass, CoolConst.iT, CoolConst.iP)
            fluid_new_temperature = (
                (energy_specific-eos.umass())/partial_derivative_energy + eos.T())
            temperature.append(fluid_new_temperature)
        self.state.temperature[:] = temperature

    def compute_liquid_density(self) -> None:
        liquid_density = []
        for eos, new_T in zip(
                self.state.equation_of_state[1:], self.state.temperature[1:]):
            density_partial_derivative = eos.first_partial_deriv(
                CoolConst.iDmass, CoolConst.iT, CoolConst.iP)
            fluid_new_density = eos.rhomass()+density_partial_derivative*(new_T-eos.T())
            liquid_density.append(fluid_new_density)
        self.state.density[1:] = liquid_density

    def compute_fluid_volume(self) -> None:
        self.state.volume[1:] = self.state.mass[1:]/self.state.density[1:]
        self.state.volume[0] = self.volume - np.sum(self.state.volume[1:])

    def compute_vapor_density(self) -> None:
        self.state.density[0] = self.state.mass[0]/self.state.volume[0]

    def compute_vapor_pressure(self) -> None:
        # Vapor pressure should be computed from finite difference approximation too, as
        # DmassT pair is not supported
        vapor_eos = self.state.equation_of_state[0]

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
        for eos, new_T in zip(self.state.equation_of_state, self.state.temperature):
            # Also check if internal energy is consistent for this approach
            eos.update(CoolConst.PT_INPUTS, self.state.pressure[0], new_T)

    def reset_flow_rates(self):
        self.net_mass_flow = np.zeros_like(self.net_mass_flow)
        self.net_energy_flow = np.zeros_like(self.net_energy_flow)

    # This is as far as common routines go, to get other parameters details about liquid
    # spatial distribution should be known

    @abstractmethod
    def advance(self) -> None:
        return


class Atmosphere(ControlVolume):

    # Class for atmosphere representation. Should impose nominal infinite
    # volume and constant values for thermodynamic paramters

    _STANDARD_TEMPERATURE = 20+273.15
    _STANDARD_PRESSURE = 101_330

    def __init__(self) -> None:
        super().__init__()

        # Possible TODO :
        # Implement selector-method to set standard temperature and pressure for
        # specified STP refernce
        eos_air = factory_eos({"air": 1}, with_state=(
            CoolConst.PT_INPUTS, self._STANDARD_PRESSURE, self._STANDARD_TEMPERATURE))

        self.state = StateData(
            equation_of_state=[eos_air],
            pressure=np.array([self._STANDARD_PRESSURE]),
            temperature=np.array([self._STANDARD_TEMPERATURE]),
            density=np.array([eos_air.rhomass()]),
            energy_specific=np.array([eos_air.umass()]),
            dynamic_viscosity=np.array([eos_air.viscosity()]),
            thermal_conductivity=np.array([eos_air.conductivity()]),
            mass=np.array([np.inf]),
            energy=np.array([np.inf]),
            level=np.array([np.inf]),
            volume=np.array([np.inf]))

    def advance(self) -> None:
        return


class Reservoir(ControlVolume):

    def __init__(self) -> None:
        super().__init__()

    def advance(self) -> None:
        pass


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

        self.volume_pure = self.compute_volume_with_level(diameter)
        self.volume = self.volume_pure + self.volume_modificator(diameter)

        self.level_graduated, self.volume_graduated = meter.graduate(
            0, diameter, self.compute_volume_with_level)
        self.volume_graduated = (
            self.volume_graduated + self.volume_modificator(self.level_graduated))

    @overload
    def compute_volume_with_level(self, level: float | float64) -> float | float64:
        ...

    @overload
    def compute_volume_with_level(self, level: NDArray[float64]) -> NDArray[float64]:
        ...

    def compute_volume_with_level(self, level):
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
        return meter.inverse_graduate(volume, self.level_graduated, self.volume_graduated)

    def compute_fluid_level(self) -> None:
        self.state.level = self.compute_level_with_volume(
            np.flip(np.cumsum(np.flip(self.state.volume))))

    def compute_liquid_pressure(self):
        layers_thickness = -np.diff(self.state.level[1:], append=0)
        individual_hydrostatic_head = self.state.density[1:]*g*layers_thickness
        cumulative_hydrostatic_head = np.cumsum(individual_hydrostatic_head)
        self.state.pressure[1:] = (
            self.state.pressure[0] + cumulative_hydrostatic_head)

    def compute_secondary_parameters(self) -> None:
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

    def advance(self):
        pass
