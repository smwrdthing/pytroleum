# Conductors here

import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod
from typing import Callable, Iterable
import pytroleum.sdyna.opdata as opd
import pytroleum.tport.efflux as efflux
from pytroleum.sdyna.interfaces import ControlVolume
from pytroleum.sdyna.controllers import PropIntDiff, StartStop

type Numeric = float | NDArray


class Conductor(ABC):

    # Abstract base class for conductor

    @abstractmethod
    def __init__(self, phase_index: float,
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

    def specify_flow(self, flow: opd.FlowData):
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

    def __init__(self, phase_index: float,
                 diameter_pipe: float,
                 diameter_valve: float,
                 elevation: float,
                 discharge_coefficient: float,
                 opening=0,
                 source: ControlVolume | None = None,
                 sink: ControlVolume | None = None,) -> None:
        super().__init__(phase_index, source, sink)
        self.diameter_pipe = diameter_pipe
        self.diameter_valve = diameter_valve

        self.discharge_coefficient = discharge_coefficient
        self.elevation = elevation

        self.opening = opening

        self.controller: PropIntDiff | StartStop | None = None

    # getter/setter for pipe diameter  ---------------------------------------------------
    @property
    def diameter_pipe(self):
        return self._diameter_pipe

    @diameter_pipe.setter
    def diameter_pipe(self, new_diameter_pipe):
        self._diameter_pipe = new_diameter_pipe
        self._area_pipe = np.pi*new_diameter_pipe**2/4
    # ------------------------------------------------------------------------------------

    # getter/setter for valve diameter ---------------------------------------------------
    @property
    def diameter_valve(self):
        return self._diameter_valve

    @diameter_valve.setter
    def diameter_valve(self, new_diameter_valve):
        self._diameter_valve = new_diameter_valve
        self._area_valve = np.pi*new_diameter_valve**2/4
    # ------------------------------------------------------------------------------------

    # getter/setter for pipe area  -------------------------------------------------------
    @property
    def area_pipe(self):
        return self._area_pipe

    @area_pipe.setter
    def area_pipe(self, new_area_pipe):
        self._area_pipe = new_area_pipe
        self._diameter_pipe = np.sqrt(4*new_area_pipe/np.pi)
    # ------------------------------------------------------------------------------------

    # getter/setter for valve area -------------------------------------------------------
    @property
    def area_valve(self):
        return self._area_valve

    @area_valve.setter
    def area_valve(self, new_area_vale):
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

        mass_flowrate_liquid = efflux.incompressible(
            self.area_valve,
            self.area_pipe,
            self.discharge_coefficient,
            upstream_state.density[1:],
            upstream_state.pressure[1:],
            downstream_state.pressure[1:]
        )

        mass_flowrate_gas = efflux.compressible(
            self.area_valve,
            self.discharge_coefficient,

            # eos makes storing k and R excessive
            upstream_state.equation_of_state[0].cvmass(
            ) / upstream_state.equation_of_state[0].cpmass(),
            upstream_state.equation_of_state[0].gas_constant(),

            upstream_state.density[0],
            upstream_state.temperature[0],
            upstream_state.pressure[0],
            downstream_state.density[0],
            downstream_state.temperature[0],
            downstream_state.pressure[0]
        )

        # Array re-collection always seemed clunky to me
        mass_flowrate = np.array(
            [*mass_flowrate_liquid, *mass_flowrate_gas])  # type: ignore
        #  Union with float causes issues

        self.flow.mass_flowrate = mass_flowrate

    def advance(self):
        pass


class CentrifugalPump(Conductor):

    # Subclass ro representcentrifugal pump

    def __init__(self, phase_index: float,
                 source: ControlVolume | None = None,
                 sink: ControlVolume | None = None) -> None:
        super().__init__(phase_index, source, sink)

    def advance(self) -> None:
        pass


class UnderPass(Conductor):

    # Subclass to represent passage at the bottom of section formed by crest of weir and
    # internal wall of vessel

    def __init__(self, phase_index: float,
                 source: ControlVolume | None = None,
                 sink: ControlVolume | None = None) -> None:
        super().__init__(phase_index, source, sink)

    def advance(self):
        pass


class OverPass(Conductor):

    # Subclass to represent passage at the top of section formed by crest of weir and
    # internal wall of vessel

    def __init__(self, phase_index: float,
                 source: ControlVolume | None = None,
                 sink: ControlVolume | None = None) -> None:
        super().__init__(phase_index, source, sink)

    def advance(self) -> None:
        pass


class FurnaceHeatConduti(Conductor):

    # Subcalss to represent heat flux from furnace

    def __init__(self, phase_index: float,
                 source: ControlVolume | None = None,
                 sink: ControlVolume | None = None) -> None:
        super().__init__(phase_index, source, sink)

    def advance(self):
        pass


class PhaseInterface(Conductor):

    # Subclass to represent interfacial interactinos

    def __init__(self, phase_index: float,
                 source: ControlVolume | None = None,
                 sink: ControlVolume | None = None) -> None:
        super().__init__(phase_index, source, sink)

    def advance(self):
        pass


if __name__ == "__main__":
    vlv = Valve(100e-3, 80e-3, 1, 0.61, 0)
