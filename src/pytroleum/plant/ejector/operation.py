from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


import numpy as np
from typing import TYPE_CHECKING, Iterable
if TYPE_CHECKING:
    from pytroleum.tdyna.CoolStub import AbstractState
else:
    from CoolProp import AbstractState
import CoolProp.constants as CoolConst


from pytroleum.plant.ejector.locator import Stream, Place, ANY, ALL
from pytroleum.plant.ejector.design import Design
from pytroleum.plant.ejector import laws
import pytroleum.plant.ejector.interface as ifc


# NaN to indicate invalid / not applicable entries
FIELD: ifc.MappedField = np.full((Stream.SIZE, Place.SIZE), np.nan)
MIX_EXCLUDED = (Stream.PRIMPARY, Stream.SECONDARY)


@dataclass
class Constraints(ABC):

    """Base class for operation conditions and requirements storage and management"""

    fluid: ifc.MappedFluidList = field(init=False)

    flow_rate: ifc.MappedField = field(init=False)
    pressure: ifc.MappedField = field(init=False)
    temperature: ifc.MappedField = field(init=False)
    density: ifc.MappedField = field(init=False)
    velocity: ifc.MappedField = field(init=False)
    Mach: ifc.MappedField = field(init=False)

    def __post_init__(self) -> None:
        self.flow_rate = FIELD.copy()
        self.pressure = FIELD.copy()
        self.temperature = FIELD.copy()
        self.density = FIELD.copy()
        self.velocity = FIELD.copy()
        self.Mach = FIELD.copy()

        self.Mach[ALL, Place.SHOCK] = 1.0  # naturally

    def adopt_state_from(self, place: int, whose: ifc.FluidIndices = MIX_EXCLUDED
                         ) -> None:

        for fluid_idx in whose:
            p = self.pressure[fluid_idx, place]
            T = self.temperature[fluid_idx, place,]

            self.fluid[fluid_idx].update(CoolConst.PT_INPUTS, p, T)

    def record_state_to(self, place: int, whose: ifc.FluidIndices = MIX_EXCLUDED) -> None:

        for fluid_idx in whose:
            self.pressure[fluid_idx, place] = self.fluid[fluid_idx].p()
            self.temperature[fluid_idx, place] = self.fluid[fluid_idx].T()


@dataclass
class Requirements(Constraints):

    """Dataclass to hold and manage field requirements for the design procedure
    of the ejector"""

    fluid: list[ifc.EquationOfState]
    primary_inflow_state: ifc.PressureAndTemperature
    secondary_inflow_state: ifc.PressureAndTemperature
    backpressure: float  # starred P_c in the paper

    def __post_init__(self) -> None:
        super().__post_init__()

        PRESSURE, TEMPERATURE = 0, 1

        # tuple indexing should work for np.array

        primary_inlet = Stream.PRIMPARY, Place.INLET
        self.pressure[primary_inlet] = self.primary_inflow_state[PRESSURE]
        self.temperature[primary_inlet] = self.primary_inflow_state[TEMPERATURE]

        secondary_inlet = Stream.SECONDARY, Place.INLET
        self.pressure[secondary_inlet] = self.secondary_inflow_state[PRESSURE]
        self.temperature[secondary_inlet] = self.secondary_inflow_state[TEMPERATURE]

        self.pressure[ALL, Place.OUTLET] = self.backpressure


@dataclass
class Conditions(Constraints):

    """Dataclass to hold and manage fields describing specific operation condition
    in the ejector"""

    def __post_init__(self) -> None:
        super().__post_init__()

    def adopt(self, requirements: ifc.Requirements) -> None:

        for f in requirements.fluid:
            f_copy = AbstractState(f.backend_name(), "&".join(f.fluid_names()))
            f_copy.update(CoolConst.PT_INPUTS, f.p(), f.T())

        # Containers are already in-place, we don't need to copy them, just the content
        self.pressure[:] = requirements.pressure[:]
        self.temperature[:] = requirements.temperature[:]

    def flow_through(self, area, whose: ifc.FluidIndex,
                     efficiency: float = laws.ISENTROPIC_EFFICIECNY) -> None:

        # We jump to inlet conditinos first to adopt parameters needed for mass flow rate
        # computation
        self.adopt_state_from(Place.INLET, [whose])

        # Flow rate has same value everywhere due to continuity
        self.flow_rate[whose, ALL] = laws.mass_flow_rate(
            area, self.fluid[whose], efficiency)

    def heat_capactiy_ratio(self, whose: ifc.FluidIndex) -> float:
        # Consider caching here
        hcr = (self.fluid[whose].cpmass() /
               self.fluid[whose].cvmass())

        return hcr

    def nozzle_mach_for(self, design: ifc.Design) -> None:

        self.Mach[Stream.PRIMPARY, Place.NOZZLE] = laws._mach_for(
            design.area_ratio(Place.NOZZLE, Place.THROAT),
            self.heat_capactiy_ratio(Stream.PRIMPARY)
        )

    def nozzle_pressure(self) -> None:

        hcr = self.heat_capactiy_ratio(Stream.PRIMPARY)

        pressure_ratio = laws.isentropic_pressure_ratio(
            hcr, self.Mach[Stream.PRIMPARY, Place.NOZZLE])

        self.pressure[Stream.PRIMPARY, Place.NOZZLE] = (
            self.pressure[Stream.PRIMPARY, Place.INLET] / pressure_ratio)

    def shock_pressure(self) -> None:

        hcr = self.heat_capactiy_ratio(Stream.SECONDARY)
        pressure_ratio = laws.isentropic_pressure_ratio(
            hcr, self.Mach[Stream.SECONDARY, Place.SHOCK])

        self.pressure[Stream.SECONDARY, Place.SHOCK] = (
            self.pressure[Stream.SECONDARY, Place.INLET] / pressure_ratio)

    def mixing_mach_for(self, deisng: ifc.Design) -> None:
        pass


# NOTE (refactoring idea)
# Different analysis methods might expect differen set of predefined
# conditions/requirements, so it probably would be benefitial to separate them??
