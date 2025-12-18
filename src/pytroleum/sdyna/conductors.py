# Conductors here
from abc import ABC, abstractmethod
import numpy as np
from scipy.constants import g, R
from scipy.optimize import newton
import CoolProp.constants as CoolConst
from pytroleum import meter
from pytroleum.tport import efflux
from pytroleum.sdyna.opdata import FlowData
from pytroleum.sdyna.interfaces import ControlVolume, Section
from pytroleum.sdyna.controllers import PropIntDiff, StartStop

from typing import Callable, Iterable, overload
from numpy.typing import NDArray
from numpy import float64


class Conductor(ABC):

    # Abstract base class for conductor
    @overload
    def __init__(self, phase_index: int,
                 source: ControlVolume | None = None,
                 sink: ControlVolume | None = None) -> None:
        ...

    @overload
    def __init__(self, phase_index: list[int],
                 source: ControlVolume | None = None,
                 sink: ControlVolume | None = None) -> None:
        ...

    @abstractmethod
    def __init__(self, phase_index,
                 source=None,
                 sink=None) -> None:

        # Possible TODO
        # default flow attribute in FlowData

        if source is None:
            from pytroleum.sdyna.convolumes import Atmosphere
            self.source = Atmosphere()
        if sink is None:
            from pytroleum.sdyna.convolumes import Atmosphere
            self.sink = Atmosphere()
        self.phase_index = phase_index
        self.controller: PropIntDiff | StartStop | None = None

    def specify_flow(self, flow: FlowData) -> None:
        self.flow = flow

    def connect_source(self, convolume: ControlVolume) -> None:
        if self not in convolume.outlets:
            convolume.outlets.append(self)
            self.source = convolume

    def connect_sink(self, convolume: ControlVolume) -> None:
        if self not in convolume.inlets:
            convolume.inlets.append(self)
            self.sink = convolume

    @abstractmethod
    def advance(self) -> None:
        return


class Valve(Conductor):

    # Subclass to represent Valve

    def __init__(
            self, phase_index: int,
            diameter_pipe: float | float64,
            diameter_valve: float | float64,
            discharge_coefficient: float | float64,
            opening: float | float64 = 0,
            elevation: float | float64 = 0,
            source: ControlVolume | None = None,
            sink: ControlVolume | None = None) -> None:

        super().__init__(phase_index, source, sink)

        self.diameter_pipe = diameter_pipe
        self.diameter_valve = diameter_valve
        self.discharge_coefficient = discharge_coefficient
        self.elevation = elevation
        self.opening = opening

        self.controller: PropIntDiff | StartStop | None = None
        self.phase_index: int

    # getter/setter for pipe diameter  ---------------------------------------------------
    @property
    def diameter_pipe(self) -> float | float64:
        return self._diameter_pipe

    @diameter_pipe.setter
    def diameter_pipe(self, new_diameter_pipe) -> None:
        self._diameter_pipe = new_diameter_pipe
        self._area_pipe = np.pi*new_diameter_pipe**2/4
    # ------------------------------------------------------------------------------------

    # getter/setter for valve diameter ---------------------------------------------------
    @property
    def diameter_valve(self) -> float | float64:
        return self._diameter_valve

    @diameter_valve.setter
    def diameter_valve(self, new_diameter_valve) -> None:
        self._diameter_valve = new_diameter_valve
        self._area_valve = np.pi*new_diameter_valve**2/4
    # ------------------------------------------------------------------------------------

    # getter/setter for pipe area  -------------------------------------------------------
    @property
    def area_pipe(self) -> float | float64:
        return self._area_pipe

    @area_pipe.setter
    def area_pipe(self, new_area_pipe) -> None:
        self._area_pipe = new_area_pipe
        self._diameter_pipe = np.sqrt(4*new_area_pipe/np.pi)
    # ------------------------------------------------------------------------------------

    # getter/setter for valve area -------------------------------------------------------
    @property
    def area_valve(self) -> float | float64:
        return self._area_valve

    @area_valve.setter
    def area_valve(self, new_area_vale) -> None:
        self._area_valve = new_area_vale
        self._diameter_valve = np.sqrt(4*new_area_vale/np.pi)
    # ------------------------------------------------------------------------------------

    def compute_flow(self):

        upstream_state = self.source.state
        downstream_state = self.sink.state

        if self.controller is not None:
            # direct assignment corresponds to controller signal interpretation
            self.opening = self.controller.signal

        if self.phase_index > 0:  # conductor deals with liquid phase

            upstream_pressure = _compute_pressure_for_elevation(
                self.elevation, upstream_state.level, upstream_state.pressure)
            downstream_pressure = _compute_pressure_for_elevation(
                self.elevation, downstream_state.level, downstream_state.pressure)

            mass_flow_rate = efflux.incompressible(
                self.area_valve*self.opening,
                self.area_pipe,
                self.discharge_coefficient,
                upstream_state.density[self.phase_index],
                upstream_pressure,
                downstream_pressure
            )
        else:  # conductor deals with vapor phase

            upstream_pressure = upstream_state.pressure[self.phase_index]
            downstream_pressure = downstream_state.pressure[self.phase_index]

            mass_flow_rate = efflux.compressible(
                self.area_valve*self.opening,
                self.discharge_coefficient,

                # eos makes storing k and R excessive
                upstream_state.equation_of_state[self.phase_index].cvmass() /
                upstream_state.equation_of_state[self.phase_index].cpmass(),
                R/upstream_state.equation_of_state[self.phase_index].molar_mass(),

                upstream_state.density[self.phase_index],
                upstream_state.temperature[self.phase_index],
                upstream_pressure,

                downstream_state.density[self.phase_index],
                downstream_state.temperature[self.phase_index],
                downstream_pressure
            )

        if mass_flow_rate >= 0:
            donor = self.source.state
        else:
            donor = self.sink.state

        self.flow.mass_flow_rate[self.phase_index] = mass_flow_rate

        (
            self.flow.energy_specific[self.phase_index],
            self.flow.temperature[self.phase_index],
            self.flow.density[self.phase_index],
        ) = (
            donor.energy_specific[self.phase_index],
            donor.temperature[self.phase_index],
            donor.density[self.phase_index]
        )

        self.flow.pressure[self.phase_index] = max(
            upstream_pressure, downstream_pressure)
        self.flow.velocity[self.phase_index] = (
            self.flow.mass_flow_rate[self.phase_index] /
            self.flow.density[self.phase_index] /
            self.area_pipe)
        self.flow.energy_specific_flow[self.phase_index] = (
            self.flow.energy_specific[self.phase_index] +
            self.flow.pressure[self.phase_index] / self.flow.density[self.phase_index] +
            self.flow.velocity[self.phase_index]**2/2)
        self.flow.energy_flow = self.flow.energy_specific_flow*self.flow.mass_flow_rate

    def advance(self):
        self.compute_flow()


class CentrifugalPump(Conductor):

    # Subclass ro representcentrifugal pump

    def __init__(self, phase_index: int,
                 source: ControlVolume | None = None,
                 sink: ControlVolume | None = None) -> None:
        super().__init__(phase_index, source, sink)

    def advance(self) -> None:
        pass


class UnderPass(Conductor):

    # Subclass to represent passage at the bottom of section formed by crest of weir and
    # internal wall of vessel

    def __init__(self,
                 edge_level: float | float64,
                 discharge_coefficient: float | float64,
                 locking_offset: float | float64 = 0,
                 source: Section | None = None,
                 sink: Section | None = None) -> None:

        super().__init__(0, source, sink)

        self.edge_level = edge_level
        # NOTE implement as property later?
        self.locking_offset = locking_offset
        self.locking_level = self.edge_level+self.locking_offset
        self.discharge_coefficient = discharge_coefficient
        self.is_locked: bool = False

        # For type checker only
        self.source: Section
        self.sink: Section

    def check_if_locked(self):
        self.is_locked = False
        criterila_level = self.equal_level_distribution()[0]
        if criterila_level > self.locking_level:
            self.is_locked = True

    def equal_level_distribution(self) -> tuple[float | float64, float | float64]:

        liquid_common_volume = (
            self.source.state.volume[1:]+self.sink.state.volume[1:])
        liquid_total_volume = np.sum(liquid_common_volume)

        # To make this stuff work graduated levels should correspond in neghbouring
        # sections, for more complicated cases there are workarounds, which can be
        # implemented too, but for now we do this
        common_level_graduated = self.source.level_graduated
        common_volume_graduated = self.source.volume_graduated+self.sink.volume_graduated

        self._liquid_volume_fractions = liquid_common_volume/liquid_total_volume

        # We get desired level via interpolation then
        liquid_common_level = meter.inverse_graduate(
            liquid_total_volume, common_level_graduated, common_volume_graduated)

        # return shape is like that for consistency with balance algorithm
        return liquid_common_level, liquid_common_level

    def _hydrostatic_balance_objective(
            self, levels: tuple[float, float], disbalance: float | float64 = 0) -> tuple:

        # Reading inputs
        source_level, sink_level = levels

        # Processing
        source_volume_with_level = (
            self.source.compute_volume_with_level(source_level))
        sink_volume_with_level = (
            self.sink.compute_volume_with_level(sink_level))
        total_volume_with_level = source_volume_with_level+sink_volume_with_level
        liquid_common_volume = (
            self.source.state.volume[1:]+self.sink.state.volume[1:])
        liquid_total_volume = np.sum(liquid_common_volume)
        liquid_reference_density = 0.5*(
            self.source.state.density[1:]+self.sink.state.density[1:])

        # Should be entirely internal stuff
        self._liquid_volume_fractions = liquid_common_volume/liquid_total_volume
        self._liquid_pseudo_density = np.sum(
            liquid_reference_density*self._liquid_volume_fractions)

        # Pressure exerted on the bottom of source section (pseudo-pure liquid - based)
        source_vapor_volume = (
            self.source.volume - source_volume_with_level)
        source_vapor_density = self.source.state.mass[0]/source_vapor_volume

        # This stuff is a good approximation at best, so no reason to sweat CoolProp's
        # EoS here to get pressure, ideal gas should do, especially considering pressure
        # difference is of interest, errors should cancel out anyways
        source_vapor_pressure = (
            source_vapor_density*R*self.source.state.temperature[0] /
            self.source.state.equation_of_state[0].molar_mass())
        source_liquid_pressure = self._liquid_pseudo_density*g*source_level
        source_total_pressure = source_vapor_pressure+source_liquid_pressure

        # Pressure exerted on the bottom of sink section (pseudo-pure liquid - based)
        sink_vapor_volume = (
            self.sink.volume-sink_volume_with_level)
        sink_vapor_density = self.sink.state.mass[0]/sink_vapor_volume
        sink_vapor_pressure = (
            sink_vapor_density*R*self.sink.state.temperature[0] /
            self.sink.state.equation_of_state[0].molar_mass())
        sink_liquid_pressure = self._liquid_pseudo_density*g*sink_level
        sink_total_pressure = sink_vapor_pressure+sink_liquid_pressure

        # Balance-based residual
        pressure_difference = source_total_pressure-sink_total_pressure
        residual_balance = pressure_difference + disbalance

        # Conservation-based residual
        residual_conservation = liquid_total_volume-total_volume_with_level

        return residual_balance, residual_conservation

    def hydrostatic_balance_distribution(
            self, level_guesse: tuple[float, float] | None = None,
            disbalance: float | float64 = 0) -> tuple:

        if level_guesse is None:
            level_guesse = (
                self.source.state.level[1], self.sink.state.level[1])

        solution = newton(
            lambda levels: self._hydrostatic_balance_objective(
                levels, disbalance),
            level_guesse)

        liquid_level_source = solution[0]
        liquid_level_sink = solution[1]

        return liquid_level_source, liquid_level_sink

    def perform_distribution(self):
        if self.is_locked:
            new_levels = self.hydrostatic_balance_distribution()
        else:
            new_levels = self.equal_level_distribution()
        liquid_total_level_source, liquid_total_level_sink = new_levels

        liquid_total_volume_source = self.source.compute_volume_with_level(
            liquid_total_level_source)
        liquid_total_volume_sink = self.sink.compute_volume_with_level(
            liquid_total_level_sink)

        liquid_volume_source = liquid_total_volume_source*self._liquid_volume_fractions
        liquid_volume_sink = liquid_total_volume_sink*self._liquid_volume_fractions

        liquid_mass_source = self.source.state.density[1:]*liquid_volume_source
        liquid_mass_sink = self.sink.state.density[1:]*liquid_volume_sink

        # Changes in energy should be handled according to flow direction and
        # new values of mass
        liquid_mass_difference_source = (
            liquid_mass_source-self.source.state.mass[1:])
        liquid_mass_difference_sink = (
            liquid_mass_source-self.sink.state.mass[1:])

        if liquid_mass_difference_source < 0:
            # liquid leaves source
            flow_specific_energy = self.source.state.energy_specific[1:]
        else:
            # liquid comes to source
            flow_specific_energy = self.sink.state.energy_specific[1:]

        liquid_energy_source = (
            self.source.state.energy[1:] +
            flow_specific_energy*liquid_mass_difference_source)
        liquid_energy_sink = (
            self.sink.state.energy[1:] +
            flow_specific_energy*liquid_mass_difference_sink)

        # Updating state variables accordingly
        self.source.state.mass[1:] = liquid_mass_source
        self.source.state.energy[1:] = liquid_energy_source
        self.sink.state.mass[1:] = liquid_mass_sink
        self.sink.state.energy[1:] = liquid_energy_sink

        # NOTE :
        # We should check if this violates mass and/or energy balance. It should not,
        # because algorithm was built with this objective in mind, but still.

    def compute_vapor_flow_rate(self):
        if self.is_locked:
            # All other stuff must be zero
            self.flow.mass_flow_rate[self.phase_index] = 0
            self.flow.energy_flow[self.phase_index] = 0
            self.flow.velocity[self.phase_index] = 0
            self.flow.energy_flow[self.phase_index] = 0
        else:
            # Actually compute flow
            flow_area = meter.area_cs_circle_trunc(
                self.source.diameter, self.edge_level)-meter.area_cs_circle_trunc(
                    self.source.diameter, self.source.state.level[0])
            flow_elevation = 0.5*(self.edge_level+self.source.state.level[0])
            vapor_mass_flow_rate = efflux.compressible(
                flow_area, self.discharge_coefficient,
                self.source.state.equation_of_state[0].cpmass() /
                self.source.state.equation_of_state[0].cvmass(),
                R/self.source.state.equation_of_state[0].molar_mass(),
                self.source.state.density[0],
                self.source.state.temperature[0],
                self.source.state.pressure[0],
                self.sink.state.density[0],
                self.sink.state.temperature[0],
                self.sink.state.pressure[0],
            )
            if vapor_mass_flow_rate >= 0:
                donor = self.source.state
            else:
                donor = self.sink.state

            energy_specific = donor.energy_specific[self.phase_index]
            pressure = donor.pressure[self.phase_index]
            density = donor.density[self.phase_index]
            velocity = vapor_mass_flow_rate / density / flow_area

            energy_specific_flow = (
                g*flow_elevation + energy_specific + pressure/density + velocity**2/2)
            energy_flow = energy_specific_flow*vapor_mass_flow_rate

            self.flow.mass_flow_rate[self.phase_index] = vapor_mass_flow_rate
            self.flow.energy_flow[self.phase_index] = energy_specific
            self.flow.pressure[self.phase_index] = pressure
            self.flow.density[self.phase_index] = density
            self.flow.velocity[self.phase_index] = velocity
            self.flow.energy_specific_flow[self.phase_index] = energy_specific_flow
            self.flow.energy_flow[self.phase_index] = energy_flow

    def advance(self):
        self.check_if_locked()
        self.compute_vapor_flow_rate()
        self.perform_distribution()  # NOTE : this disrupted solver in legacy, be careful


class OverPass(Conductor):

    # TODO : finish OverPass

    # Subclass to represent passage at the top of section formed by crest of weir and
    # internal wall of vessel

    def __init__(self,
                 edge_level: float | float64,
                 discharge_coefficient: float | float64,
                 source: Section | None = None,
                 sink: Section | None = None) -> None:

        self.phase_index: list[int]

        # Only vapor and lightest fluid passes
        super().__init__([0, 1], source, sink)

        self.source: Section
        self.sink: Section

        self.edge_level = edge_level
        self.discharge_coefficinet = discharge_coefficient
        self.is_reached: bool = False

        # Possible TODO : recast into property, simplify total volume computations
        self._vapor_flow_area = (
            meter.area_cs_circle_trunc(self.source.diameter, self.source.diameter) -
            meter.area_cs_circle_trunc(self.source.diameter, self.edge_level)
        )

    def check_if_reached(self):
        self.is_reached = False
        criterial_level = max(max(self.source.state.level),
                              max(self.sink.state.level))
        if criterial_level >= self.edge_level:
            self.is_reached = True

    def compute_vapor_flow(self):

        phase_index = self.phase_index[0]

        vapor_mass_flow_rate = efflux.compressible(
            self._vapor_flow_area,
            self.discharge_coefficinet,
            self.source.state.equation_of_state[phase_index].cpmass() /
            self.source.state.equation_of_state[phase_index].cvmass(),
            R/self.source.state.equation_of_state[phase_index].molar_mass(),
            self.source.state.density[phase_index],
            self.source.state.temperature[phase_index],
            self.source.state.pressure[phase_index],
            self.sink.state.density[phase_index],
            self.sink.state.temperature[phase_index],
            self.sink.state.pressure[phase_index]
        )

        self.flow.mass_flow_rate[phase_index] = vapor_mass_flow_rate
        self.flow

    def compute_liquid_overflow(self):
        # NOTE : for this to work other conductors must be resolved before, so
        # order of execution must be enforced
        phase_index = self.phase_index[1]  # lightest liquid
        self.flow.mass_flow_rate[phase_index] = 0
        if self.is_reached:
            # collect net flow rates for source formed by other conductors
            other_liquid_flow_rates_inlet = 0
            other_liquid_flow_rates_outlet = 0
            for inlet in self.source.inlets:
                if inlet is not self:
                    other_liquid_flow_rates_inlet += inlet.flow.mass_flow_rate[1:]
            for outlet in self.source.outlets:
                if outlet is not self:
                    other_liquid_flow_rates_outlet += outlet.flow.mass_flow_rate[1:]

            overflow_rate = self.source.state.density[1]*np.sum(
                (other_liquid_flow_rates_inlet-other_liquid_flow_rates_outlet) /
                self.source.state.density)

            if overflow_rate < 0:
                # NOTE : assigining integer 0 potentially can lead to troubles if numpy
                # will not resolve array elements' types correctly
                overflow_rate = 0

            self.flow.mass_flow_rate[phase_index] = overflow_rate

    def advance(self) -> None:
        self.check_if_reached()
        self.compute_vapor_flow()
        self.compute_liquid_overflow()


class FurnacePolynomial(Conductor):

    # Subclass to represent heat flux from furnace with polynomial
    # approximation of furnace heat_flux(fuel_flow_rate)-like characteristic

    def __init__(self, phase_index: int,
                 source: ControlVolume | None = None,
                 sink: ControlVolume | None = None) -> None:
        super().__init__(phase_index, source, sink)

    def advance(self):
        pass


class PhaseInterface(Conductor):

    # Subclass to represent interfacial interactions

    def __init__(self, phase_index: int,
                 source: ControlVolume | None = None,
                 sink: ControlVolume | None = None) -> None:
        super().__init__(phase_index, source, sink)

    def advance(self):
        pass


def _compute_pressure_for_elevation(
        elevation: float | float64,
        levels_profile: NDArray[float64],
        pressures_profile: NDArray[float64]) -> float | float64:

    # Processing algorithm complies with introduced data storage convention for
    # multiphase situation

    pressure_on_elevation = np.interp(
        elevation,
        [0, *np.flip(levels_profile)],
        [*np.flip(pressures_profile), pressures_profile[0]]
    )

    return pressure_on_elevation


if __name__ == "__main__":
    vlv = Valve(0, 100e-3, 80e-3, 0.61, 1)
    print(vlv.diameter_pipe*1e3)
