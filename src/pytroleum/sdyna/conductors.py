# Conductors here

import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod
from typing import Callable, Iterable
import pytroleum.sdyna.opdata as opd
import pytroleum.tport.efflux as efflux
from pytroleum.sdyna.interfaces import ControlVolume
from pytroleum.sdyna.controllers import PropIntDiff, StartStop

type Numeric = float | NDArray


class Conductor(ABC):

    # Abstract base class for conductor

    @abstractmethod
    def __init__(self) -> None:
        return

    def specify_flow(self, flow: opd.FlowData):
        self.flow = flow

    def connect_source(self, convolume: ControlVolume) -> None:
        if self not in convolume.outlets:
            convolume.outlets.append(self)
            self.source = convolume

    def connect_sink(self, convolume: ControlVolume) -> None:
        if self not in convolume.inlets:
            convolume.inlets.append(self)
            self.sink = convolume

    @abstractmethod
    def advance(self) -> None:
        return


class Valve(Conductor):

    # Subclass to represent Valve

    def __init__(self,
                 diameter_pipe: float,
                 diameter_valve: float,
                 elevation: float,
                 discharge_coefficient: float,
                 opening=0) -> None:
        super().__init__()
        self.diameter_pipe = diameter_pipe
        self.diameter_valve = diameter_valve

        self.discharge_coefficient = discharge_coefficient
        self.elevation = elevation

        self.opening = opening

        self.controller: PropIntDiff | StartStop | None = None

    # getter/setter for pipe diameter  ---------------------------------------------------
    @property
    def diameter_pipe(self):
        return self._diameter_pipe

    @diameter_pipe.setter
    def diameter_pipe(self, new_diameter_pipe):
        self._diameter_pipe = new_diameter_pipe
        self._area_pipe = np.pi*new_diameter_pipe**2/4
    # ------------------------------------------------------------------------------------

    # getter/setter for valve diameter ---------------------------------------------------
    @property
    def diameter_valve(self):
        return self._diameter_valve

    @diameter_valve.setter
    def diameter_valve(self, new_diameter_valve):
        self._diameter_valve = new_diameter_valve
        self._area_valve = np.pi*new_diameter_valve**2/4
    # ------------------------------------------------------------------------------------

    # getter/setter for pipe area  -------------------------------------------------------
    @property
    def area_pipe(self):
        return self._area_pipe

    @area_pipe.setter
    def area_pipe(self, new_area_pipe):
        self._area_pipe = new_area_pipe
        self._diameter_pipe = np.sqrt(4*new_area_pipe/np.pi)
    # ------------------------------------------------------------------------------------

    # getter/setter for valve area -------------------------------------------------------
    @property
    def area_valve(self):
        return self._area_valve

    @area_valve.setter
    def area_valve(self, new_area_vale):
        self._area_valve = new_area_vale
        self._diameter_valve = np.sqrt(4*new_area_vale/np.pi)
    # ------------------------------------------------------------------------------------

    def compute_mass_flow_rate(self):
        pass

    def advance(self):
        pass


class CentrifugalPump(Conductor):

    # Subclass ro representcentrifugal pump

    def __init__(self) -> None:
        super().__init__()

    def advance(self) -> None:
        pass


class UnderPass(Conductor):

    # Subclass to represent passage at the bottom of section formed by crest of weir and
    # internal wall of vessel

    def __init__(self) -> None:
        super().__init__()

    def advance(self):
        pass


class OverPass(Conductor):

    # Subclass to represent passage at the top of section formed by crest of weir and
    # internal wall of vessel

    def __init__(self) -> None:
        super().__init__()

    def advance(self) -> None:
        pass


class FurnaceHeatConduti(Conductor):

    # Subcalss to represent heat flux from furnace

    def __init__(self) -> None:
        super().__init__()

    def advance(self):
        pass


class PhaseInterface(Conductor):

    # Subclass to represent interfacial interactinos

    def __init__(self) -> None:
        super().__init__()

    def advance(self):
        pass
