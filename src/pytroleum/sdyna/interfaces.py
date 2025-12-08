from __future__ import annotations
from typing import Protocol, runtime_checkable
from pytroleum.sdyna.opdata import StateData, FlowData


@runtime_checkable
class ControlVolume(Protocol):

    # Interface for control volume

    outlets: list[Conductor]
    inlets: list[Conductor]
    state: StateData

    def __init__(self) -> None:
        ...

    def connect_inlet(self, conductor: Conductor):
        ...

    def connect_outlet(self, conductor: Conductor):
        ...

    def advance(self) -> None:
        ...


@runtime_checkable
class Conductor(Protocol):

    # Interface for conductor

    source: ControlVolume
    sink: ControlVolume
    flow: FlowData

    def __init__(self) -> None:
        ...

    def connect_source(self, convolume: ControlVolume) -> None:
        ...

    def connect_sink(self, convolume: ControlVolume) -> None:
        ...

    def advance(self) -> None:
        ...
