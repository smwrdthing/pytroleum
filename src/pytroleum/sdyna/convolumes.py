# Control volumes here
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from scipy.constants import g
import CoolProp.constants as CoolConst
from abc import ABC, abstractmethod
from typing import Callable
from pytroleum.sdyna.interfaces import Conductor
from pytroleum.sdyna import opdata as opd
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

    def matter_spec_energy(self):
        self.state.energy_specific = self.state.energy/self.state.mass

    def matter_temperature(self):
        T = []

        for eos, p, u in zip(
                self.state.equation_of_state,
                self.state.pressure,
                self.state.energy_specific):
            eos.update(CoolConst.PUmass_INPUTS, p, u)
            T.append(eos.T())
        T = np.array(T)
        self.state.temperature = T

    @abstractmethod
    def advance(self) -> None:
        return


class Atmosphere(ControlVolume):

    # Class for atmosphere representation. Should impose nominal infinite
    # volume and constant values for thermodynamic paramters

    def __init__(self) -> None:
        super().__init__()

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
        self.diameter = diameter
        self.length_left_semiaxis = length_left_semiaxis
        self.length_cylinder = length_cylinder
        self.length_right_semiaxis = length_right_semiaxis
        self.volume_modificator = volume_modificator
        self.volume_pure = self.volume_with_level(diameter)
        self.volume = self.volume_pure+self.volume_modificator(diameter)
        # Graduation is needed for level
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

    def matter_level(self) -> Numeric:
        volume_of_matter_cumulative = np.cumsum(self.state.volume[::-1])
        level_of_matter = meter.inverse_graduate(
            volume_of_matter_cumulative, self.level_graduated, self.volume_graduated)
        return level_of_matter

    def advance(self):
        pass


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
