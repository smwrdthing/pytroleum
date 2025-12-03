# Conductors here

import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Iterable
from interfaces import ControlVolume


class Conductor(ABC):

    # Abstract base class for conductor

    @abstractmethod
    def __init__(self) -> None:
        self.sink: ControlVolume | None = None
        self.source: ControlVolume | None = None

    def connect_as_sink(self, convolume: ControlVolume):
        self.sink = convolume

    def connect_as_source(self, convolume: ControlVolume):
        self.source = convolume

    @abstractmethod
    def advance(self):
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

    def advance(self):
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

    def advance(self):
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
