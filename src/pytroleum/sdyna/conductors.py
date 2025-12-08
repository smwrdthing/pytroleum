# Conductors here

import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Iterable
import pytroleum.sdyna.opdata as opd
from pytroleum.sdyna.interfaces import ControlVolume


class Conductor(ABC):

    # Abstract base class for conductor

    @abstractmethod
    def __init__(self) -> None:
        self.sink: ControlVolume | None = None
        self.source: ControlVolume | None = None

    def specify_flow(self, flow: opd.FlowData):
        self.flow = flow

    def connect_source(self, convolume: ControlVolume) -> None:
        if self not in convolume.outlets:
            self.source = convolume

    def connect_sink(self, convolume: ControlVolume) -> None:
        if self not in convolume.inlets:
            self.sink = convolume

    @abstractmethod
    def advance(self) -> None:
        return


class Valve(Conductor):

    # Subclass to represent Valve

    def __init__(self) -> None:
        super().__init__()

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
