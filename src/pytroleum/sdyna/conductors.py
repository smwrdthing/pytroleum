from abc import ABC, abstractmethod
import numpy as np
from scipy.constants import g, R
from scipy.optimize import fsolve
from pytroleum import meter
from pytroleum.tport import efflux
from pytroleum.sdyna.opdata import FlowData, StateData
from pytroleum.sdyna.interfaces import ControlVolume, Section
from pytroleum.sdyna.controllers import PropIntDiff, StartStop

from typing import Iterable
from numpy.typing import NDArray
from numpy import float64

# TODO : sort out type specifications and actual assignments in ints


class Conductor(ABC):

    @abstractmethod
    def __init__(self, of_phase: int | Iterable[int],
                 source: ControlVolume | None = None,
                 sink: ControlVolume | None = None) -> None:

        if source is None:
            from pytroleum.sdyna.convolumes import Atmosphere
            source = Atmosphere()
        if sink is None:
            from pytroleum.sdyna.convolumes import Atmosphere
            sink = Atmosphere()

        self.connect_source(source)
        self.connect_sink(sink)

        self.of_phase = of_phase
        self.controller: PropIntDiff | StartStop | None = None

        self.flow: FlowData

    def connect_source(self, convolume: ControlVolume) -> None:
        """Assigns control volume as source for conductor and adds conductor to control
        volume's outlets list if conductor is not already here, does nothing otherwise."""
        if self not in convolume.outlets:
            convolume.outlets.append(self)
            self.source = convolume

    def connect_sink(self, convolume: ControlVolume) -> None:
        """Assigns control volume as sink for conductor and adds conductor to control
        volume's inlets list if conductor is not already here, does nothing otherwise."""
        if self not in convolume.inlets:
            convolume.inlets.append(self)
            self.sink = convolume

    def propagate_flow(self):
        """Contributes flow rate values to net flow attributes of source and sink with
        appropriate signs."""
        self.source.net_mass_flow = self.source.net_mass_flow-self.flow.mass_flow_rate
        self.source.net_energy_flow = self.source.net_energy_flow-self.flow.energy_flow

        self.sink.net_mass_flow = self.sink.net_mass_flow + self.flow.mass_flow_rate
        self.sink.net_energy_flow = self.sink.net_energy_flow + self.flow.energy_flow

    @abstractmethod
    def advance(self) -> None:
        """Resolves conductor state at given time step."""
        return


class Fixed(Conductor):

    def __init__(self, of_phase: int | list[int],
                 source: ControlVolume | None = None,
                 sink: ControlVolume | None = None) -> None:
        super().__init__(of_phase, source, sink)

    def advance(self) -> None:
        self.propagate_flow()
        return


class Valve(Conductor):

    def __init__(
            self, of_phase: int,
            diameter_pipe: float | float64,
            diameter_valve: float | float64,
            discharge_coefficient: float | float64,
            opening: float | float64 = 0,
            elevation: float | float64 = 0,
            source: ControlVolume | None = None,
            sink: ControlVolume | None = None) -> None:

        super().__init__(of_phase, source, sink)

        self.diameter_pipe = diameter_pipe
        self.diameter_valve = diameter_valve
        self.discharge_coefficient = discharge_coefficient
        self.elevation = elevation
        self.opening = opening
        self.controller: PropIntDiff | StartStop | None = None

        self.of_phase: int

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
        """Computes flow rate with respect to upstream and downstream states."""

        upstream_state = self.source.state
        downstream_state = self.sink.state

        if self.controller is not None:
            # direct assignment corresponds to controller signal interpretation
            self.opening = self.controller.signal

        if self.of_phase > 0:  # conductor deals with liquid phase

            upstream_pressure = _compute_pressure_for(
                self.elevation, upstream_state.level, upstream_state.pressure)
            downstream_pressure = _compute_pressure_for(
                self.elevation, downstream_state.level, downstream_state.pressure)

            mass_flow_rate = efflux.incompressible(
                self.area_valve*self.opening,
                self.area_pipe,
                self.discharge_coefficient,
                upstream_state.density[self.of_phase],
                upstream_pressure,
                downstream_pressure
            )
        else:  # conductor deals with vapor phase

            upstream_pressure = upstream_state.pressure[self.of_phase]
            downstream_pressure = downstream_state.pressure[self.of_phase]

            mass_flow_rate = efflux.compressible(
                self.area_valve*self.opening,
                self.discharge_coefficient,

                # eos makes storing k and R excessive
                upstream_state.equation[self.of_phase].cpmass() /
                upstream_state.equation[self.of_phase].cvmass(),
                R/upstream_state.equation[self.of_phase].molar_mass(),

                upstream_state.density[self.of_phase],
                upstream_state.temperature[self.of_phase],
                upstream_pressure,

                downstream_state.density[self.of_phase],
                downstream_state.temperature[self.of_phase],
                downstream_pressure
            )

        if mass_flow_rate >= 0:
            donor = self.source.state
        else:
            donor = self.sink.state

        self.flow.mass_flow_rate[self.of_phase] = mass_flow_rate

        (
            self.flow.energy_specific[self.of_phase],
            self.flow.temperature[self.of_phase],
            self.flow.density[self.of_phase],
        ) = (
            donor.energy_specific[self.of_phase],
            donor.temperature[self.of_phase],
            donor.density[self.of_phase]
        )

        self.flow.pressure[self.of_phase] = max(
            upstream_pressure, downstream_pressure)
        self.flow.velocity[self.of_phase] = (
            self.flow.mass_flow_rate[self.of_phase] /
            self.flow.density[self.of_phase] /
            self.area_pipe)
        self.flow.energy_specific_flow[self.of_phase] = (
            self.flow.energy_specific[self.of_phase] +
            self.flow.pressure[self.of_phase] / self.flow.density[self.of_phase] +
            self.flow.velocity[self.of_phase]**2/2)
        self.flow.energy_flow = self.flow.energy_specific_flow*self.flow.mass_flow_rate

    def advance(self):
        self.compute_flow()
        self.propagate_flow()


class CentrifugalPump(Conductor):

    def __init__(self, of_phase: int, elevation: float = 0,
                 source: ControlVolume | None = None,
                 sink: ControlVolume | None = None) -> None:

        self.of_phase = of_phase
        self.elevation = elevation

        self.coefficients: tuple[float, float, float] | NDArray[float64]
        self.resistance_coeff: float
        self.angular_velocity: float
        self.flow_area: float  # maybe better to do this in base class

        super().__init__(of_phase, source, sink)

    def characteristic_reference(
            self, angular_velocity: float,
            volume_flow_rates: tuple[float, float, float],
            heads: tuple[float, float, float]) -> None:
        """Computes coefficients for pump's quadratic characteristic model with given
        angular velocity and three reference values of volumetric flow rate and head."""

        coeff_matrix = []
        free_vector = []
        for flow_rate, head in zip(volume_flow_rates, heads):
            coeff_matrix.append([
                angular_velocity**2,
                -2*angular_velocity*flow_rate,
                -flow_rate**2])
            free_vector.append(head)
        self.coefficients = np.linalg.solve(coeff_matrix, free_vector)

    def compute_flow(self) -> None:
        """Computes mass and energy flow produced by pump"""
        upstream_pressure = _compute_pressure_for(
            self.elevation, self.source.state.level, self.source.state.pressure)
        downstream_pressure = _compute_pressure_for(
            self.elevation, self.sink.state.level, self.sink.state.pressure)

        # Pump should not allow backflow due to the inverse rotation issues,
        # so we always take from source
        density = self.source.state.density[self.of_phase]
        pressure = self.source.state.pressure[self.of_phase]
        temperature = self.source.state.temperature[self.of_phase]
        energy_specific = self.source.state.energy_specific[self.of_phase]
        pressure_difference = downstream_pressure-upstream_pressure

        static_head_difference = pressure_difference/density/g
        k1, k2, k3 = self.coefficients
        A = k3 + self.resistance_coeff/(2*g*self.flow_area**2)
        B = 2*k2*self.angular_velocity
        C = static_head_difference - k1*self.angular_velocity**2
        D = np.sqrt(B**2-4*A*C)
        volumetric_flow_rate = (np.sqrt(D)-B)/(2*A)

        mass_flow_rate = 0
        if volumetric_flow_rate > 0:
            mass_flow_rate = volumetric_flow_rate*density

        velocity = mass_flow_rate/density/self.flow_area
        energy_specific_flow = (
            energy_specific + pressure/density + g*self.elevation + velocity**2/2)
        flow_energy = energy_specific_flow*mass_flow_rate

        self.flow.mass_flow_rate[self.of_phase] = mass_flow_rate
        self.flow.velocity[self.of_phase] = velocity
        self.flow.temperature[self.of_phase] = temperature
        self.flow.density[self.of_phase] = density
        self.flow.energy_specific[self.of_phase] = energy_specific
        self.flow.energy_specific_flow[self.of_phase] = energy_specific_flow
        self.flow.energy_flow[self.of_phase] = flow_energy

    def advance(self) -> None:
        self.compute_flow()
        self.propagate_flow()


class UnderPass(Conductor):

    # Subclass to represent passage at the bottom of section formed by crest of weir and
    # internal wall of vessel

    def __init__(self,
                 edge_level: float | float64,
                 discharge_coefficient: float | float64,
                 locking_offset: float | float64 = 0,
                 source: Section | None = None,
                 sink: Section | None = None) -> None:

        self.of_phase: int
        super().__init__(0, source, sink)

        self.edge_level = edge_level
        # NOTE implement as property later?
        self.locking_offset = locking_offset
        self.locking_level = self.edge_level+self.locking_offset
        self.discharge_coefficient = discharge_coefficient
        self.is_locked: bool = False

        self.source: Section
        self.sink: Section

    def check_if_locked(self):
        """Switches corresponding flag if locking conditions are met/not met."""
        self.is_locked = False
        criterila_level = self.equal_level_distribution()[0]
        if criterila_level > self.locking_level:
            self.is_locked = True

    def equal_level_distribution(self) -> tuple[float | float64, float | float64]:
        """Performs liquid distribution among neighboring section so that final values
        of total level are same on both sides."""
        liquid_common_volume = (
            self.source.state.volume[1:]+self.sink.state.volume[1:])
        liquid_total_volume = np.sum(liquid_common_volume)

        # To make this stuff work graduated levels should correspond in neighbouring
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
        """Represents system of equations describing hydrostatic balance conditions to
        minimize with rootfinding.

        Parameters
        ----------
        levels
            tuple of level values on both sides.
        disbalance, optional
            disbalance introduced in equations for tuning, by default 0.

        Returns
        -------
            Residuals of hydrostatic balance equations.
        """
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
            self.source.state.equation[0].molar_mass())
        source_liquid_pressure = self._liquid_pseudo_density*g*source_level
        source_total_pressure = source_vapor_pressure+source_liquid_pressure

        # Pressure exerted on the bottom of sink section (pseudo-pure liquid - based)
        sink_vapor_volume = (
            self.sink.volume-sink_volume_with_level)
        sink_vapor_density = self.sink.state.mass[0]/sink_vapor_volume
        sink_vapor_pressure = (
            sink_vapor_density*R*self.sink.state.temperature[0] /
            self.sink.state.equation[0].molar_mass())
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
        """Performs distribution of liquids among neighboring control volumes so that
        values of pressure exerted on the bottom of control volumes are same.

        Parameters
        ----------
        level_guesse, optional
            initial guesse for levels to pass to rootfinding algorithm, if None - values
            of levels from previous time step are used, by default None.
        disbalance, optional
            disbalance in hydrostatic balance equations for tuning, by default 0.

        Returns
        -------
            Values of total level that comply to hydrostatic balance.
        """

        if level_guesse is None:
            level_guesse = (
                self.source.state.level[1], self.sink.state.level[1])

        solution = fsolve(
            lambda levels: self._hydrostatic_balance_objective(
                levels, disbalance),
            level_guesse)

        liquid_level_source = solution[0]
        liquid_level_sink = solution[1]

        return liquid_level_source, liquid_level_sink

    def distribute(self):
        """Checks if hydrostatic lock is formed, does hydrostatic balance distribution
        if yes and equal level distribution otherwise. For multiphase situation recombines
        phase composition from initial volume fractions."""
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
            liquid_mass_sink-self.sink.state.mass[1:])

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

    def compute_vapor_flow(self):
        """Determines vapor flow rate value if there is a gap between liquid surface and
        weir's crest, sets 0 for vapor flow rate otherwise"""
        if self.is_locked:
            # All other stuff must be zero
            self.flow.mass_flow_rate[self.of_phase] = 0
            self.flow.energy_flow[self.of_phase] = 0
            self.flow.velocity[self.of_phase] = 0
            self.flow.energy_flow[self.of_phase] = 0
        else:
            # Actually compute flow
            flow_area = meter.area_cs_circle_trunc(
                self.source.diameter, self.edge_level)-meter.area_cs_circle_trunc(
                    self.source.diameter, self.source.state.level[0])
            flow_elevation = 0.5*(self.edge_level+self.source.state.level[0])
            vapor_mass_flow_rate = efflux.compressible(
                flow_area, self.discharge_coefficient,
                self.source.state.equation[0].cpmass() /
                self.source.state.equation[0].cvmass(),
                R/self.source.state.equation[0].molar_mass(),
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

            energy_specific = donor.energy_specific[self.of_phase]
            pressure = donor.pressure[self.of_phase]
            density = donor.density[self.of_phase]
            velocity = vapor_mass_flow_rate / density / flow_area

            energy_specific_flow = (
                g*flow_elevation + energy_specific + pressure/density + velocity**2/2)
            energy_flow = energy_specific_flow*vapor_mass_flow_rate

            self.flow.mass_flow_rate[self.of_phase] = vapor_mass_flow_rate
            self.flow.energy_flow[self.of_phase] = energy_specific
            self.flow.pressure[self.of_phase] = pressure
            self.flow.density[self.of_phase] = density
            self.flow.velocity[self.of_phase] = velocity
            self.flow.energy_specific_flow[self.of_phase] = energy_specific_flow
            self.flow.energy_flow[self.of_phase] = energy_flow

    def advance(self):
        self.check_if_locked()
        self.distribute()  # NOTE : this disrupted solver in legacy, be careful
        self.compute_vapor_flow()
        self.propagate_flow()


class OverPass(Conductor):

    # Subclass to represent passage at the top of section formed by crest of weir and
    # internal wall of vessel

    def __init__(self,
                 edge_level: float | float64,
                 discharge_coefficient: float | float64,
                 source: Section | None = None,
                 sink: Section | None = None) -> None:

        # Only vapor and lightest fluid passes
        super().__init__((0, 1), source, sink)
        self.discharge_coefficinet = discharge_coefficient
        self.edge_level = edge_level
        self.is_reached: bool = False

        # Possible TODO : recast into property, simplify total volume computations
        self._vapor_flow_area = (
            meter.area_cs_circle_trunc(self.source.diameter, self.source.diameter) -
            meter.area_cs_circle_trunc(self.source.diameter, self.edge_level))

        self.of_phase: tuple[int, int]
        self.source: Section
        self.sink: Section

    def check_if_reached(self):
        """Switches corresponding flag if liquid level reaches crest of weir or falls
        behind it"""
        self.is_reached = False
        criterial_level = max(max(self.source.state.level[1:]),
                              max(self.sink.state.level[1:]))
        if criterial_level >= self.edge_level:
            self.is_reached = True

    def compute_vapor_flow(self):
        """Determines vapor flow rate with compressible adiabatic flow model"""
        of_phase = self.of_phase[0]

        vapor_mass_flow_rate = efflux.compressible(
            self._vapor_flow_area,
            self.discharge_coefficinet,
            self.source.state.equation[of_phase].cpmass() /
            self.source.state.equation[of_phase].cvmass(),
            R/self.source.state.equation[of_phase].molar_mass(),
            self.source.state.density[of_phase],
            self.source.state.temperature[of_phase],
            self.source.state.pressure[of_phase],
            self.sink.state.density[of_phase],
            self.sink.state.temperature[of_phase],
            self.sink.state.pressure[of_phase]
        )

        if vapor_mass_flow_rate > 0:
            donor = self.source
        else:
            donor = self.sink

        self.flow.mass_flow_rate[of_phase] = vapor_mass_flow_rate
        self.flow.velocity[of_phase] = (
            self.flow.mass_flow_rate[of_phase] /
            self._vapor_flow_area/donor.state.density[of_phase])
        self.flow.energy_specific[of_phase] = donor.state.energy_specific[of_phase]
        self.flow.temperature[of_phase] = donor.state.temperature[of_phase]
        self.flow.density[of_phase] = donor.state.density[of_phase]
        self.flow.energy_specific_flow[of_phase] = (
            self.flow.energy_specific[of_phase] +
            self.flow.pressure[of_phase] / self.flow.density[of_phase] +
            self.flow.velocity[of_phase]**2/2)
        self.flow.energy_flow[of_phase] = (self.flow.energy_specific_flow[of_phase] *
                                           vapor_mass_flow_rate)

    def compute_liquid_overflow(self):
        """Determines flow rate of lightest liquid if weir's crest is reached, sets 0
        otherwise. Computation are based on condition of constant total volume of liquids
        in source, so for correct resolution all flow rates from other conductors must be
        computed"""
        of_phase = self.of_phase[1]  # lightest liquid
        self.flow.mass_flow_rate[of_phase] = 0
        if self.is_reached:
            # collect net flow rates for source formed by other conductors
            other_liquid_flow_rates_inlet = np.array([0.0])
            other_liquid_flow_rates_outlet = np.array([0.0])
            for inlet in self.source.inlets:
                if inlet is not self:
                    other_liquid_flow_rates_inlet += inlet.flow.mass_flow_rate[1:]
            for outlet in self.source.outlets:
                if outlet is not self:
                    other_liquid_flow_rates_outlet += outlet.flow.mass_flow_rate[1:]

            other_liquid_net_flow = (
                other_liquid_flow_rates_inlet - other_liquid_flow_rates_outlet)

            overflow_rate = other_liquid_net_flow[0] - np.sum(
                self.source.state.density[0] / self.source.state.density[1:] *
                other_liquid_net_flow[1:])

            if overflow_rate < 0:
                # NOTE : assigining integer 0 potentially can lead to troubles if numpy
                # will not resolve array elements' types correctly
                overflow_rate = 0

            donor = self.source

            self.flow.mass_flow_rate[of_phase] = overflow_rate
            self.flow.density[of_phase] = donor.state.density[of_phase]
            self.flow.temperature[of_phase] = donor.state.temperature[of_phase]
            self.flow.pressure[of_phase] = donor.state.pressure[of_phase-1]
            self.flow.velocity[of_phase] = 0
            self.flow.energy_specific[of_phase] = donor.state.energy_specific[of_phase]
            self.flow.energy_specific_flow[of_phase] = (
                self.flow.energy_specific[of_phase] +
                self.flow.pressure[of_phase] / self.flow.density[of_phase] +
                self.flow.velocity[of_phase]**2/2)
            self.flow.energy_flow[of_phase] = (self.flow.energy_specific[of_phase] *
                                               overflow_rate)

    def advance(self) -> None:
        self.check_if_reached()
        self.compute_liquid_overflow()
        self.compute_vapor_flow()
        self.propagate_flow()


class FurnacePolynomial(Conductor):

    def __init__(
            self, of_phase: int,
            minmax_fuel_flow: tuple[float, float], elevation: float,
            diameter: float, center_distance: float, in_control_volume: ControlVolume,
            coeffs: NDArray[float64] = np.array([21.62, 10.59])*1e3) -> None:

        super().__init__(of_phase, None, in_control_volume)

        self.of_phase = of_phase
        self.min_fuel_flow, self.max_fuel_flow = minmax_fuel_flow
        self.range_fuel_flow = self.max_fuel_flow-self.min_fuel_flow
        self.elevation = elevation
        self.diameter = diameter
        self.center_distance = center_distance

        self.fuel_flow = self.min_fuel_flow

        self.coeffs = coeffs
        self.controller: PropIntDiff | None

    def compute_heat_flux(self):
        """Compute heat flux produced by furnace with polynomial approximation and fuel
        flow rate"""

        bottom_layer_level = 0
        if self.of_phase < len(self.sink.state.mass)-1:
            bottom_layer_level = self.sink.state.level[self.of_phase+1]
        upper_layer_level = self.sink.state.level[self.of_phase]

        # Furnace activates only when heated layer fully encloses it
        enclosed = (
            (upper_layer_level >= self.elevation+self.diameter/2) and
            (bottom_layer_level <= self.elevation-self.diameter/2))

        output = 1
        if self.controller is not None:
            output = self.controller.signal

        fuel_flow = self.min_fuel_flow+output*self.range_fuel_flow
        heat = np.polynomial.polynomial.polyval(
            fuel_flow, self.coeffs)*enclosed

        self.fuel_flow = fuel_flow*enclosed
        self.flow.energy_flow[self.of_phase] = heat

    def advance(self):
        self.compute_heat_flux()
        self.propagate_flow()


class PhaseInterface(Conductor):

    def __init__(self, of_phase: tuple[int, int],
                 in_control_volume: Section,
                 evaporation_coefficient: float = 0) -> None:

        # This one differs from others in a sense that energy is not moved
        # from one control volume to other, it redistributed between phases
        # in one control volume
        #
        # of_phase should be an iterable of two specifying adjacent phases that
        # form interface. Index listing should correspond with introduced convention
        # (lighter comes first)

        super().__init__(of_phase, None, in_control_volume)
        self.evaporation_coefficient = evaporation_coefficient

        self.saturation_state: StateData
        self.heat_transfer_coeff: float
        self.of_phase: tuple[int, int]
        self.sink: Section

    def compute_flow(self):
        of_light_phase, of_heavy_phase = self.of_phase
        level = self.sink.state.level[of_heavy_phase]

        heat_transfer_area = meter.area_planecut_section_horiz_ellipses(
            self.sink.length_left_semiaxis,
            self.sink.length_cylinder,
            self.sink.length_right_semiaxis,
            self.sink.diameter,
            level)

        light_phase_temperature = self.sink.state.temperature[of_light_phase]
        heavy_phase_temperature = self.sink.state.temperature[of_heavy_phase]
        temperature_difference = light_phase_temperature-heavy_phase_temperature

        # Watch sign carefully!
        heat_flow = heat_transfer_area*self.heat_transfer_coeff*temperature_difference

        self.flow.energy_flow[of_light_phase] = -heat_flow
        self.flow.energy_flow[of_heavy_phase] = heat_flow

        if of_light_phase == 0:
            evaporation_area = heat_transfer_area
            saturation_pressure = self.saturation_state.pressure[of_light_phase]
            saturation_density = self.saturation_state.density[of_light_phase]

            evaporation_rate = efflux.evaporation_heuristic(
                evaporation_area,
                self.evaporation_coefficient,
                saturation_density, saturation_pressure,
                self.sink.state.pressure[of_light_phase])

            self.flow.mass_flow_rate[of_light_phase] = evaporation_rate

    def advance(self):
        self.compute_flow()
        self.propagate_flow()


def _compute_pressure_for(
        elevation: float | float64,
        levels_profile: NDArray[float64],
        pressures_profile: NDArray[float64]) -> float | float64:
    """Computes pressure value for given elevation with provided pressure profile data.
    Profile data should follow introduced convention for phase data storage.

    Parameters
    ----------
    elevation
        level value at which pressure value is desired.
    levels_profile
        level data of pressure profile.
    pressures_profile
        pressure data of pressure profile.

    Returns
    -------
        pressure at given elevation for provided profile.
    """

    # Processing algorithm complies with introduced data storage convention for
    # multiphase situation

    pressure_on_elevation = np.interp(
        elevation,
        [0, *np.flip(levels_profile)],
        [*np.flip(pressures_profile), pressures_profile[0]]
    )

    return pressure_on_elevation
