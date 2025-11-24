# Control volumes here
from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from interfaces import GenericCDR
from typing import Callable


class AbstractCV(ABC):

    # Abstract base class for control volume

    @abstractmethod
    def __init__(self) -> None:
        # Matter state manager here?
        self.outlets: list[GenericCDR] = []
        self.inlets: list[GenericCDR] = []

    # Introduce custom decorator for iterable inputs
    def connet_as_inlet(self, conductor: GenericCDR) -> None:
        if conductor not in self.inlets:
            self.inlets.append(conductor)

    # Introduce custom decorator for iterable inputs
    def connect_as_outlet(self, conductor: GenericCDR) -> None:
        if conductor not in self.outlets:
            self.outlets.append(conductor)

    @abstractmethod
    def advance(self) -> None:
        return


class Atmosphere(AbstractCV):

    # Class for atmosphere representation. Should imposs nominal infinite
    # volume and constant values for thermodynamic paramters

    def __init__(self) -> None:
        super().__init__()

    def advance(self):
        pass


class Reservoir(AbstractCV):

    # Class to represent petroleum fluids reservoir. In context of dynamical system
    # modelling imposes infinite volume and constant params (for now?)

    def __init__(self) -> None:
        super().__init__()

    def advance(self):
        pass


class SectionH(AbstractCV):

    # Class for horizontal section

    def __init__(self, D: float, L: float, H_left: float, H_right: float,
                 mod: Callable) -> None:
        super().__init__()
        self.D = D
        self.H_left, self.L, self.H_right = H_left, L, H_right
        self.mod = mod

    def advance(self):
        pass


class SectionV(AbstractCV):

    # Class for vertical section, not really needed right now, might be useful later for
    # tests and other equipment?

    def __init__(self, D: float, H: float, mod: Callable) -> None:
        super().__init__()
        self.D = D
        self.H = H
        self.mod = mod

    def advance(self):
        pass
