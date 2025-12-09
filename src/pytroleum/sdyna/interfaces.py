from __future__ import annotations
from typing import Protocol, runtime_checkable
from pytroleum.sdyna.opdata import StateData, FlowData
from pytroleum.sdyna.controllers import PropIntDiff, StartStop


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

    phase_index: float
    source: ControlVolume
    sink: ControlVolume
    flow: FlowData
    controller: PropIntDiff | StartStop | None

    def __init__(self, phase_index: float,
                 source: ControlVolume | None,
                 sink: ControlVolume | None) -> None:
        ...

    def connect_source(self, convolume: ControlVolume) -> None:
        ...

    def connect_sink(self, convolume: ControlVolume) -> None:
        ...

    def advance(self) -> None:
        ...
