# System-level functionality goes here
from abc import ABC, abstractmethod
from pytroleum.sdyna.interfaces import ControlVolume, Conductor


class DynamicNetwork(ABC):

    # Must figure out how to connect objects quick

    def __init__(self) -> None:
        self.control_volumes: list[ControlVolume] = []
        self.conductors: list[Conductor] = []

    @abstractmethod
    def advance(self):
        pass


class EmulsionTreater(DynamicNetwork):

    def __init__(self) -> None:
        super().__init__()
