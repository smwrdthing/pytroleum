from __future__ import annotations
from dataclasses import dataclass, field


import numpy as np


from pytroleum.plant.ejector.locator import Place, Stream, ANY, ALL
import pytroleum.plant.ejector.interface as ifc


DIMENSION: ifc.MappedDimension = np.full(Place.SIZE, np.nan)


@dataclass
class Design:
    diameter: ifc.MappedDimension = field(init=False)
    area: ifc.MappedDimension = field(init=False)
    length: ifc.MappedDimension = field(init=False)
    angle: ifc.MappedDimension = field(init=False)

    def __post_init__(self):
        self.diameter = DIMENSION.copy()
        self.area = DIMENSION.copy()
        self.length = DIMENSION.copy()
        self.angle = DIMENSION.copy()

    def throat(self, diameter: float):
        self.diameter[Place.THROAT] = diameter
        self.area[Place.THROAT] = np.pi*diameter**2/4

    def nozzle(self, diameter: float):
        self.diameter[Place.NOZZLE] = diameter
        self.area[Place.NOZZLE] = np.pi*diameter**2/4

    def area_ratio(self, of: ifc.DimensionIndex, to: ifc.DimensionIndex) -> float:
        return self.area[of]/self.area[to]

    # we need interface if we want to compute areas from here
    def mixing_area_at(self, conditions: ifc.Conditions, whose: ifc.FluidIndex):

        if whose == Stream.PRIMPARY:
            pass

        if whose == Stream.SECONDARY:
            pass

    def shock_area_at(self, conditions: ifc.Conditions):
        pass
