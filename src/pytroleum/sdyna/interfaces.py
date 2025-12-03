from __future__ import annotations
from typing import Protocol, runtime_checkable


@runtime_checkable
class ControlVolume(Protocol):

    # Interface for control volume

    outlets: list[Conductor]
    inlets: list[Conductor]

    def __init__(self) -> None:
        ...

    def connect_as_inlet(self, conductor: Conductor):
        ...

    def connect_as_outlet(self, conductor: Conductor):
        ...

    def advance(self) -> None:
        ...


@runtime_checkable
class Conductor(Protocol):

    # Interface for conductor

    source: ControlVolume
    sink: ControlVolume

    def __init__(self) -> None:
        ...

    def connect_as_source(self, convolume: ControlVolume) -> None:
        ...

    def connect_as_sink(self, convolume: ControlVolume) -> None:
        ...

    def advance(self) -> None:
        ...
