# Control volumes here
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
import CoolProp.constants as CoolConst
from abc import ABC, abstractmethod
from typing import Callable
from pytroleum.sdyna.interfaces import Conductor
from pytroleum.sdyna import opdata as opd
from pytroleum import meter

type Numeric = float | NDArray


class ControlVolume(ABC):

    # Abstract base class for control volume

    def __init__(self) -> None:
        self.outlets: list[Conductor] = []
        self.inlets: list[Conductor] = []
        self.volume: Numeric = np.inf

    # Introduce custom decorator for iterable inputs
    def connect_inlet(self, conductor: Conductor) -> None:
        if conductor not in self.inlets:
            conductor.connect_sink(self)
            self.inlets.append(conductor)

    # Introduce custom decorator for iterable inputs
    def connect_outlet(self, conductor: Conductor) -> None:
        if conductor not in self.outlets:
            conductor.connect_source(self)
            self.outlets.append(conductor)

    def specify_state(self, state: opd.StateData):
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

    def advance(self):
        return


class Reservoir(ControlVolume):

    # Class to represent petroleum fluids reservoir. In context of dynamical system
    # modelling imposes infinite volume and constant params (for now?)

    def __init__(self) -> None:
        super().__init__()

    def advance(self):
        pass


class SectionHorizontal(ControlVolume):

    # Class for horizontal section

    def __init__(self,
                 diameter: Numeric,
                 length_cylinder: Numeric,
                 length_left_semiaxis: Numeric,
                 length_right_semiaxis: Numeric,
                 volume_modificator: Callable[[Numeric | Numeric], Numeric]) -> None:
        super().__init__()
        self.diameter = diameter
        self.length_left_semiaxis = length_left_semiaxis
        self.length_cylinder = length_cylinder
        self.length_right_semiaxis = length_right_semiaxis
        self.volume_modificator = volume_modificator
        self.volume_pure = self.volume_with_level(diameter)
        self.volume = self.volume_pure-self.volume_modificator(diameter)
        # Graduation is needed for level
        self.level_graduated, self.volume_graduated = meter.graduate(
            0, diameter, self.volume_with_level)

    def volume_with_level(self, level: Numeric) -> Numeric:
        V_pure = meter.volume_section_horiz_ellipses(self.length_left_semiaxis,
                                                     self.length_cylinder,
                                                     self.length_right_semiaxis,
                                                     self.diameter, level)
        V_modif = self.volume_modificator(level)
        V = V_pure+V_modif
        return V

    def level_with_volume(self, volume: Numeric) -> Numeric:
        return meter.inverse_graduate(volume, self.level_graduated, self.volume_graduated)

    def matter_level(self) -> Numeric:
        V_cumulative = np.cumsum(self.state.volume[::-1])
        h = meter.inverse_graduate(
            V_cumulative, self.level_graduated, self.volume_graduated)
        return h

    def advance(self):
        pass


class SectionVertical(ControlVolume):

    # Class for vertical section, not really needed right now, might be useful later for
    # tests and other equipment?

    def __init__(self,
                 diameter: Numeric,
                 height: Numeric,
                 volume_modificator: Numeric) -> None:
        super().__init__()
        self.diameter = diameter
        self.height = height
        self.volume_modificator = volume_modificator

    def advance(self):
        pass
