import abc
import numpy as np


class AbstractCDR:

    # Should be abstract base class for conductor

    def __init__(self) -> None:
        self.sink = None
        self.source = None

    def connect_as_sink(self, convolume):
        self.sink = convolume

    def connect_as_source(self, convolume):
        self.source = convolume

    def advance(self):
        pass


class Valve(AbstractCDR):

    # Subclass to represent Valve

    def __init__(self) -> None:
        super().__init__()


class CentrifugalPump(AbstractCDR):

    # Subclass ro representcentrifugal pump

    def __init__(self) -> None:
        super().__init__()


class UnderPass(AbstractCDR):

    # Subclass to represent passage at the bottom of section formed by crest of weir and
    # internal wall of vessel

    def __init__(self) -> None:
        super().__init__()


class OverPass(AbstractCDR):

    # Subclass to represent passage at the top of section formed by crest of weir and
    # internal wall of vessel

    def __init__(self) -> None:
        super().__init__()


class LumpedFurnaceWall(AbstractCDR):

    # Subcalss to represent heat flux from furnace

    def __init__(self) -> None:
        super().__init__()


class PhaseInterface(AbstractCDR):

    # Subclass to represent interfacial interactinos

    def __init__(self) -> None:
        super().__init__()
