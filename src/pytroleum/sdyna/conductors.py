# Conductors here
from abc import ABC, abstractmethod
import numpy as np
import pytroleum.tport.efflux as efflux
from pytroleum.sdyna.opdata import FlowData
from pytroleum.sdyna.interfaces import ControlVolume
from pytroleum.sdyna.controllers import PropIntDiff, StartStop

from typing import Callable, Iterable
from numpy import float64


class Conductor(ABC):

    # Abstract base class for conductor

    @abstractmethod
    def __init__(self, phase_index: int,
                 source: ControlVolume | None = None,
                 sink: ControlVolume | None = None) -> None:
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
            elevation: float | float64,
            discharge_coefficient: float | float64,
            opening: float | float64 = 0,
            source: ControlVolume | None = None,
            sink: ControlVolume | None = None) -> None:

        super().__init__(phase_index, source, sink)

        self.diameter_pipe = diameter_pipe
        self.diameter_valve = diameter_valve
        self.discharge_coefficient = discharge_coefficient
        self.elevation = elevation
        self.opening = opening

        self.controller: PropIntDiff | StartStop | None = None

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

    def compute_mass_flow_rate(self):

        # It was implemented somewhat like this in legacy, but here we should find better
        # way to handle multiphase situation and facilitate ODE system solution. Code is
        # brought here in this form for preliminary considerations and should be
        # reimplemented with smart utilization of phase ransparency idea

        upstream_state = self.source.state
        downstream_state = self.sink.state

        valve_opening = 1
        if self.controller is not None:
            # controller for valve is adjusted to return value between 0 and 1,
            # this value is interpreted as an opening of valve without any transform
            valve_opening = self.controller.signal

        if self.phase_index > 0:  # conductor deals with liquid phase

            # This relies heavily on data storage conventions employed for control
            # volumes. This is needed to account for hydrostatic head contributions.
            # If one does not want to compute them - flat pressure profile can be
            # recorded instead of cumulative-sum based pressure profile, buth this
            # should be handled in convolumes

            upstream_pressure = np.interp(
                self.elevation,
                [0, *np.flip(upstream_state.level)],
                [*np.flip(upstream_state.pressure),
                 upstream_state.pressure[0]]
            )

            downstream_pressure = np.interp(
                self.elevation,
                [0, *np.flip(downstream_state.level)],
                [*np.flip(downstream_state.pressure),
                 downstream_state.pressure[0]]
            )
            mass_flow_rate = efflux.incompressible(
                self.area_valve*valve_opening,
                self.area_pipe,
                self.discharge_coefficient,
                upstream_state.density[self.phase_index],
                upstream_pressure,
                downstream_pressure
            )
        else:  # conductor deals with vapor phase

            mass_flow_rate = efflux.compressible(
                self.area_valve*valve_opening,
                self.discharge_coefficient,

                # eos makes storing k and R excessive
                upstream_state.equation_of_state[self.phase_index].cvmass() /
                upstream_state.equation_of_state[self.phase_index].cpmass(),
                upstream_state.equation_of_state[self.phase_index].gas_constant(
                ),

                upstream_state.density[self.phase_index],
                upstream_state.temperature[self.phase_index],
                upstream_state.pressure[self.phase_index],
                downstream_state.density[self.phase_index],
                downstream_state.temperature[self.phase_index],
                downstream_state.pressure[self.phase_index]
            )

        self.flow.mass_flow_rate[self.phase_index] = mass_flow_rate

    def compute_energy_flux(self):
        # j = (u+p/rho+w^2/2)*G
        if self.flow.mass_flow_rate >= 0:
            donor = self.source.state
        else:
            donor = self.sink.state

        # NOTE this violates DRY, find better way to do this later
        valve_opening = 1
        if self.controller is not None:
            valve_opening = self.controller.signal

        # NOTE I am inclined to make FlowData fields scalar now hmm, should thin this
        # through better, pretty large change. Try to write as if single value kept for
        # FlowData and see what happens
        #
        # On the other hand we have a pretty solid reason to keep array interfaces : inlet
        # flow rates are multiphase. This thing alone probably stumps all the reasoning
        # I suppose... Add to it opportunity to account for flow

        (
            self.flow.energy_specific[self.phase_index],
            self.flow.temperature[self.phase_index],
            self.flow.density[self.phase_index],
        ) = (
            donor.energy_specific[self.phase_index],
            donor.temperature[self.phase_index],
            donor.density[self.phase_index]
        )
        self.flow.pressure[self.phase_index] = np.interp(
            self.elevation,
            [0, *np.flip(donor.level)],
            [*np.flip(donor.pressure), donor.pressure[0]]
        )
        self.flow.velocity[self.phase_index] = (self.flow.mass_flow_rate /
                                                self.flow.density /
                                                (self.area_valve*valve_opening))
        self.flow.energy_specific_flow = (self.flow.energy_specific +
                                          self.flow.pressure/self.flow.density +
                                          self.flow.velocity**2/2)
        self.flow.energy_flow = self.flow.energy_specific_flow*self.flow.mass_flow_rate

    def advance(self):
        self.compute_mass_flow_rate()
        self.compute_energy_flux()


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

    def __init__(self, phase_index: int,
                 source: ControlVolume | None = None,
                 sink: ControlVolume | None = None) -> None:
        super().__init__(phase_index, source, sink)

    def advance(self):
        pass


class OverPass(Conductor):

    # Subclass to represent passage at the top of section formed by crest of weir and
    # internal wall of vessel

    def __init__(self, phase_index: int,
                 source: ControlVolume | None = None,
                 sink: ControlVolume | None = None) -> None:
        super().__init__(phase_index, source, sink)

    def advance(self) -> None:
        pass


class FurnaceHeatConduti(Conductor):

    # Subcalss to represent heat flux from furnace

    def __init__(self, phase_index: int,
                 source: ControlVolume | None = None,
                 sink: ControlVolume | None = None) -> None:
        super().__init__(phase_index, source, sink)

    def advance(self):
        pass


class PhaseInterface(Conductor):

    # Subclass to represent interfacial interactinos

    def __init__(self, phase_index: int,
                 source: ControlVolume | None = None,
                 sink: ControlVolume | None = None) -> None:
        super().__init__(phase_index, source, sink)

    def advance(self):
        pass


if __name__ == "__main__":
    vlv = Valve(0, 100e-3, 80e-3, 1, 0.61, 0)
    print(vlv.diameter_pipe*1e3)
