# Control volumes here
from __future__ import annotations
import numpy as np
import CoolProp.constants as CoolConst
from abc import ABC, abstractmethod
from interfaces import Conductor
from typing import Callable
from opdata import StateData
import pytroleum.meter as meter
from pytroleum.meter import Numeric


class ControlVolume(ABC):

    # Abstract base class for control volume

    def __init__(self) -> None:
        self.outlets: list[Conductor] = []
        self.inlets: list[Conductor] = []
        self.volume: Numeric = np.inf
        self.conditions = StateData()

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

    def matter_spec_energy(self):
        self.conditions.u = self.conditions.E/self.conditions.m

    def matter_temperature(self):
        T = []
        for eos, p, u in zip(self.conditions.eos, self.conditions.p, self.conditions.u):
            T.append(eos.update(CoolConst.PUmass_INPUTS, p, u))
        T = np.array(T)
        self.conditions.T = T

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

    def __init__(self, D: Numeric, L: Numeric, H_left: Numeric, H_right: Numeric,
                 volume_modificator: Callable[[Numeric | Numeric], Numeric]) -> None:
        super().__init__()
        self.D = D
        self.H_left, self.L, self.H_right = H_left, L, H_right

        self.volume_modificator = volume_modificator
        self.volume_pure = self.volume_with_level(D)
        self.volume = self.volume_pure-self.volume_modificator(D)
        # Graduation is needed for level
        self.grad_levels, self.grad_volumes = meter.graduate(
            0, D, self.volume_with_level)

    def volume_with_level(self, h: Numeric) -> Numeric:
        V_pure = meter.volume_section_horiz_ellipses(
            self.H_left, self.L, self.H_right, self.D, h)
        V_modif = self.volume_modificator(h)
        V = V_pure+V_modif
        return V

    def level_with_volume(self, volume: Numeric) -> Numeric:
        return meter.inverse_graduate(volume, self.grad_levels, self.grad_volumes)

    def matter_level(self, h: Numeric) -> Numeric:
        V_cumulative = np.cumsum(self.conditions.V[::-1])
        h = meter.inverse_graduate(
            V_cumulative, self.grad_levels, self.grad_volumes)
        return h

    def advance(self):
        pass


class SectionVertical(ControlVolume):

    # Class for vertical section, not really needed right now, might be useful later for
    # tests and other equipment?

    def __init__(self, D: Numeric, H: Numeric, mod: Numeric) -> None:
        super().__init__()
        self.D = D
        self.H = H
        self.mod = mod

    def advance(self):
        pass
