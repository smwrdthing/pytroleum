from __future__ import annotations
from typing import Protocol


class GenericCV(Protocol):

    # Interface for control volume

    outlets: list[GenericCDR]
    inlets: list[GenericCDR]

    def __init__(self) -> None:
        ...

    def connect_as_inlet(self, conductor: GenericCDR):
        ...

    def connect_as_outlet(self, conductor: GenericCDR):
        ...

    def advance(self) -> None:
        ...


class GenericCDR(Protocol):

    # Interface for conductor

    source: GenericCV
    sink: GenericCV

    def __init__(self) -> None:
        ...

    def connect_as_source(self, convolume: GenericCV) -> None:
        ...

    def connect_as_sink(self, convolume: GenericCV) -> None:
        ...

    def advance(self) -> None:
        ...
