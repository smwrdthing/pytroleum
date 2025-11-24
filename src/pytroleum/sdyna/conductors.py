# Conductors here

import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Iterable
from interfaces import GenericCV


class AbstractCDR(ABC):

    # Abstract base class for conductor

    @abstractmethod
    def __init__(self) -> None:
        self.sink: GenericCV | None = None
        self.source: GenericCV | None = None

    def connect_as_sink(self, convolume: GenericCV):
        self.sink = convolume

    def connect_as_source(self, convolume: GenericCV):
        self.source = convolume

    @abstractmethod
    def advance(self):
        return


class Valve(AbstractCDR):

    # Subclass to represent Valve

    def __init__(self) -> None:
        super().__init__()

    def advance(self):
        pass


class CentrifugalPump(AbstractCDR):

    # Subclass ro representcentrifugal pump

    def __init__(self) -> None:
        super().__init__()

    def advance(self):
        pass


class UnderPass(AbstractCDR):

    # Subclass to represent passage at the bottom of section formed by crest of weir and
    # internal wall of vessel

    def __init__(self) -> None:
        super().__init__()

    def advance(self):
        pass


class OverPass(AbstractCDR):

    # Subclass to represent passage at the top of section formed by crest of weir and
    # internal wall of vessel

    def __init__(self) -> None:
        super().__init__()

    def advance(self):
        pass


class LumpedFurnaceWall(AbstractCDR):

    # Subcalss to represent heat flux from furnace

    def __init__(self) -> None:
        super().__init__()

    def advance(self):
        pass


class PhaseInterface(AbstractCDR):

    # Subclass to represent interfacial interactinos

    def __init__(self) -> None:
        super().__init__()

    def advance(self):
        pass
