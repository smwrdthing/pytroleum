from abc import ABC, abstractmethod
from typing import Literal, Callable
from pytroleum.sdyna.convolumes import ControlVolume, SectionHorizontal
from pytroleum.sdyna.conductors import Conductor, Fixed, Valve
from pytroleum.sdyna.controllers import PropIntDiff, StartStop
from pytroleum.sdyna.custom_solvers import ExplicitEuler
import numpy as np
from numpy import float64
from numpy.typing import NDArray


_VALID_OBJECTIVES = ["level", "pressure", "temperature"]


class DynamicNetwork(ABC):

    def __init__(self) -> None:
        self.control_volumes: list[ControlVolume] = []
        self.conductors: list[Conductor] = []
        self.objectives: dict[Conductor, Callable[[], float]] = {}
        self.solver: ExplicitEuler

        self._masses: NDArray[float64]
        self._energies: NDArray[float64]
        self._state_vector: NDArray[float64]

        self._mass_flows: NDArray[float64]
        self._energy_flows: NDArray[float64]
        self._flows: NDArray[float64]

        self._number_of_phases: int
        self._number_of_control_volumes: int
        self._state_vector_size: int

    def add_control_volume(self, control_volume: ControlVolume) -> None:
        """Adds control volume to a list of network control volumes"""
        if control_volume not in self.control_volumes:
            self.control_volumes.append(control_volume)

    def add_conductor(self, conductor: Conductor) -> None:
        """Adds conductor to a list of network control volumes"""
        if conductor not in self.conductors:
            self.conductors.append(conductor)

    def evaluate_size(self) -> None:
        """Determines "size" of system with number of control volumes and number of
        presented phases. Allocates arrays to hold state-vector information"""
        self._number_of_control_volumes = len(self.control_volumes)
        self._number_of_phases = len(self.control_volumes[0].state.mass)
        self._state_vector_size = 2*self._number_of_control_volumes*self._number_of_phases

        self._masses = np.zeros(self._state_vector_size//2)
        self._energies = np.zeros(self._state_vector_size//2)
        self._state_vector = np.zeros(self._state_vector_size)

        self._mass_flows = np.zeros(self._state_vector_size//2)
        self._energy_flows = np.zeros(self._state_vector_size//2)
        self._flows = np.zeros(self._state_vector_size)

    def bind_objective(
            self, objective: tuple[Literal["level", "temperature", "pressure"], int],
            in_control_volume: ControlVolume, conductor: Conductor) -> None:
        """Assigns control variable to track to a controllable conductor in system.

        Parameters
        ----------
        objective
            tuple with first element as a string to specify variable to control and second
            element as integer index to specify phase.
        in_control_volume
            control volume where specified control variable must be maintained.
        conductor
            conductor that affects specified control variable in specified control volume.

        Raises
        ------
        KeyError
            if phase index exceeds size of phase state-arrays
        KeyError
            if provided string does not specify valid control variable to maintain
        KeyError
            If control volume is missing from a system
        KeyError
            if conductor is missing from a system
        """

        parameter, of_phase = objective

        if of_phase > len(in_control_volume.state.mass)-1:
            raise KeyError(f"Phase {of_phase} is not presented")
        if parameter not in _VALID_OBJECTIVES:
            raise KeyError("Invalid objecive")
        if in_control_volume not in self.control_volumes:
            raise KeyError("Control volume not found in the system")
        if conductor not in self.conductors:
            raise KeyError("Conductor not found in the system")
        if parameter == "level":
            self.objectives[conductor] = lambda: (
                in_control_volume.state.level[of_phase])
        if parameter == "temperature":
            self.objectives[conductor] = lambda: (
                in_control_volume.state.temperature[of_phase])
        if parameter == "pressure":
            self.objectives[conductor] = lambda: (
                in_control_volume.state.pressure[of_phase])

    def connect_elements(
            self, connection_map: dict[Conductor, tuple[ControlVolume, ControlVolume]]):
        """Connects elements in system to each other with provided connection map.

        Parameters
        ----------
        connection_map
            A dictionary with conductors as keys and tuples with pair of control volumes
            to connect with conductor as values.

        Raises
        ------
        KeyError
            if conductor is missing from system
        ValueError
            if value in conncetion map contains tuple of same control volumes
        """
        conductors = list(connection_map.keys())
        control_volume_pairs = list(connection_map.values())

        if any(conductor not in self.conductors for conductor in conductors):
            raise KeyError("Map contains unknown conductors")

        if any(sink is source for sink, source in control_volume_pairs):
            raise ValueError("Map contains looped paths")

        # for sink, source in control_volume_pairs:
        #     if sink not in self.control_volumes or source not in self.control_volumes:
        #         raise ValueError("Map contains unknonws control volumes")
        # Unregistered control volumes should be allowed if we want to skip computations
        # for infinite-volume control volumes (atmosphere, reservoir etc)

        for item in connection_map.items():
            conductor, (source, sink) = item
            conductor.connect_source(source)
            conductor.connect_sink(sink)

    def map_state_to_vector(self) -> NDArray[float64]:
        """Maps control volumes' state parameters to solver's state vector,
        i.e. m, E -> y"""
        for i, cv in enumerate(self.control_volumes):
            start = i*self._number_of_phases
            stop = start+self._number_of_phases
            self._masses[start:stop] = cv.state.mass
            self._energies[start:stop] = cv.state.energy

        self._state_vector[:self._state_vector_size//2] = self._masses
        self._state_vector[self._state_vector_size//2:] = self._energies

        return self._state_vector

    def map_vector_to_state(self, y: NDArray[float64]) -> None:
        """Maps solver's state vector to control volumes' state parameters,
        i.e. y -> m, E"""
        if len(y) != self._state_vector_size:
            raise RuntimeError("Incorrect size of state vector")

        self._state_vector[:] = y
        self._masses = y[:self._state_vector_size//2]
        self._energies = y[self._state_vector_size//2:]

        for i, cv in enumerate(self.control_volumes):
            start = i*self._number_of_phases
            stop = start+self._number_of_phases
            cv.state.mass[:] = self._masses[start:stop]
            cv.state.energy[:] = self._energies[start:stop]

    def map_flows_to_vector(self) -> NDArray[float64]:
        """Maps flow rates in system into 1D vector for solver to use"""
        for i, cv in enumerate(self.control_volumes):
            start = i*self._number_of_phases
            stop = start+self._number_of_phases

            self._mass_flows[start:stop] = cv._net_mass_flow
            self._energy_flows[start:stop] = cv._net_energy_flow

        self._flows[:self._state_vector_size//2] = self._mass_flows
        self._flows[self._state_vector_size//2:] = self._energy_flows

        return self._flows

    @abstractmethod
    def ode_system(self, t: float, y: NDArray[float64]) -> NDArray[float64]:
        """Represents right-hand side of system of ordinary differential equations"""
        return self.map_flows_to_vector()

    def prepare_solver(self, time_step, start_time=0) -> None:
        self.solver = ExplicitEuler(
            fun=self.ode_system, t0=start_time,
            y0=self.map_state_to_vector(), time_step=time_step)

    @abstractmethod
    def advance(self):
        """Process dynamic system for current time step"""

        # Base class implementation ignores state-altering conductors bullshit.
        # This expects regular systems with one-to-one mapping of masses and energies
        # to secondary parameters and, consequently, one-to-one mapping of masses and
        # energies into flow rates.
        #
        # For this time-advancement only masses and energies considered to be known at
        # initial time-step. Thus, advancement begins with computations of secondary
        # parameters from new state-vector values and so on.
        #
        # State-altering abomination is handled in time advancement algorithm of class
        # that represents system with this stuff.

        self.map_vector_to_state(self.solver.y)

        for control_volume in self.control_volumes:
            control_volume.advance()

        for conductor in self.conductors:
            # If more control strategies emerge consider moving control logic to
            # conductor advancement algorithm
            if isinstance(conductor.controller, PropIntDiff):
                conductor.controller.control(
                    self.solver.time_step, self.objectives[conductor]())
            if isinstance(conductor.controller, StartStop):
                conductor.controller.control(self.objectives[conductor]())
            conductor.advance()

        self.solver.step()
