# Control volumes here
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from scipy.constants import g
import CoolProp.constants as CoolConst
from pytroleum import meter

from typing import Callable, overload
from numpy.typing import NDArray
from numpy import float64
from pytroleum.sdyna.interfaces import Conductor
from pytroleum.sdyna.opdata import StateData


class ControlVolume(ABC):

    # Abstract base class for control volume

    def __init__(self) -> None:
        self.outlets: list[Conductor] = []
        self.inlets: list[Conductor] = []
        self.volume: float | float64 = np.inf

    # Introduce custom decorator for iterable inputs
    def connect_inlet(self, conductor: Conductor) -> None:
        if conductor not in self.inlets:
            conductor.sink = self
            self.inlets.append(conductor)

    # Introduce custom decorator for iterable inputs
    def connect_outlet(self, conductor: Conductor) -> None:
        if conductor not in self.outlets:
            conductor.source = self
            self.outlets.append(conductor)

    def specify_state(self, state: StateData) -> None:
        self.state = state

    # NOTE
    # methods for advancment are listed in the order of intended sequential execution

    def compute_fluid_energy_specific(self) -> None:
        self.state.energy_specific = self.state.energy/self.state.mass

    def compute_fluid_temperature(self) -> None:
        # density does not depend much on pressure, so vapor
        # pressure might be used as value for system's pressure
        pressure_system = self.state.pressure[0]
        T = []

        for eos, u in zip(self.state.equation_of_state, self.state.energy_specific):
            # update algortihm, this is not going to work (PUmass is not supported)
            eos.update(CoolConst.PUmass_INPUTS, pressure_system, u)
            T.append(eos.T())
        T = np.array(T)
        self.state.temperature = T

    def compute_liquid_density(self) -> None:
        # eos should be valid for current state from pressure-energy
        # computations, so we can read values without an update
        for phase_index, eos in enumerate(self.state.equation_of_state[1:], start=1):
            self.state.density[phase_index] = eos.rhomass()

    def compute_fluid_volume(self) -> None:
        self.state.volume[1:] = self.state.mass[1:]/self.state.density[1:]
        self.state.volume[0] = self.volume - np.sum(self.state.volume[1:])

    def compute_vapor_density(self) -> None:
        self.state.density[0] = self.state.mass[0]/self.state.volume[0]

    def compute_vapor_pressure(self) -> None:
        self.state.equation_of_state[0].update(
            CoolConst.DmassT_INPUTS, self.state.density[0], self.state.temperature[0])
        self.state.pressure[0] = self.state.equation_of_state[0].p()

    # This is as far as common routines go, to get other parameters details about liquid
    # spatial distribution should be known

    @abstractmethod
    def advance(self) -> None:
        return


class Atmosphere(ControlVolume):

    # Class for atmosphere representation. Should impose nominal infinite
    # volume and constant values for thermodynamic paramters

    def __init__(self) -> None:
        super().__init__()

        # Possible TODO :
        # Implement selector-method to set standard temperature and pressure for
        # specified STP refernce
        self._standard_temperature = 20+273.15
        self._standard_pressure = 101_330
        eos_air = factory_eos({"air": 1}, with_state=(
            CoolConst.PT_INPUTS, self._standard_pressure, self._standard_temperature))

        self.specify_state(
            StateData(
                equation_of_state=[eos_air],
                pressure=np.array([self._standard_pressure]),
                temperature=np.array([self._standard_temperature]),
                density=np.array([eos_air.rhomass()]),
                energy_specific=np.array([eos_air.umass()]),
                dynamic_viscosity=np.array([eos_air.viscosity()]),
                thermal_conductivity=np.array([eos_air.conductivity()]),
                mass=np.array([np.inf]),
                energy=np.array([np.inf]),
                level=np.array([np.inf]),
                volume=np.array([np.inf]))
        )

    def advance(self) -> None:
        return


class Reservoir(ControlVolume):

    # Class to represent petroleum fluids reservoir. In context of dynamical system
    # modelling imposes infinite volume and constant params (for now?)

    def __init__(self) -> None:
        super().__init__()

    def advance(self) -> None:
        pass


class SectionHorizontal(ControlVolume):

    # Class for horizontal section
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
        # This might look nasty, but I think doing it in place
        # more elegant than trying to break it down
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
        self.compute_vapor_pressure()
        self.compute_fluid_level()
        self.compute_liquid_pressure()

    def advance(self) -> None:
        self.compute_secondary_parameters()


class SectionVertical(ControlVolume):

    # Class for vertical section, not really needed right now, might be useful later for
    # tests and other equipment?

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
