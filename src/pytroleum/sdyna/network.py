# System-level functionality goes here
from abc import ABC, abstractmethod
from typing import Literal, Callable, overload
from scipy.integrate import OdeSolver, RK23
from pytroleum.sdyna.convolumes import ControlVolume, SectionHorizontal
from pytroleum.sdyna.conductors import (Conductor, Fixed, Valve, OverPass,
                                        UnderPass, CentrifugalPump)
from pytroleum.sdyna.controllers import PropIntDiff, StartStop
from pytroleum.sdyna.opdata import StateData
from pytroleum.sdyna.custom_solvers import ExplicitEuler
import numpy as np
from numpy import float64
from numpy.typing import NDArray

# NOTE :
# this is WIP, so it is still forming. Few good ideas are of concern now, should try
# implementing them

# For optimisation purposes stick with arrays. Before solution process begins evaluate
# size of system using number of phases and number of control volumes, then create
# system-level state vector for ODE, should be faster
#
# Apparently slice-assignments are slightly slower thane new array creation.
# We still can leverage this to simplify procedure for dynamic system with state-altering
# elements conductors. (see UnderPass)


# NOTE on order of resolution:
# Distribution algorithms of UnderPass must be applied before everything else in
# order to get correct pressure values for other computations.
#
# Don't forget to amend solver's state-vector after distribution is done, or you
# will eventually encounter negative masses and energies in intermediate
# computations. OverPass advancement must be last, as it relies on flow rates of
# other conductors
#
# In legacy it was like that
# * ->
# Perform ODE solver step ->
# Assign new masses and energies to CV states ->
# Invoke UnderPass distribution algorithm ->
# Amend Solver's state vector ->
# Compute secondary parameters in CVs ->
# Log system parameters ->
# Compute control errors ->
# Apply control ->
# Compute system flows ->
# ** ->
#
# First two steps are considered as part of "process" procedure for ODE solution,
# all others as part of "postprocess" procedure.
#
# Sequence above looks like a mess, mostly due to the features of distributing
# conductors (UnderPass) requiering special treatment. Here we should refine
# time-advancement algorithms to make them general enough to cover such unusual
# cases too.

# Advancing conductors first is, most likely, the way :
#
# UnderPass -> externals -> OverPass -> to CV advancement
#
# When we look at this like that, we consider only states in CVs to
# be known (state vector), during system advancement we basically evaluate
# our right side for previous time-step first by condutor's advancement.
# This eliminates time-advancement mess above
# (and probably even eliminates need for solver amendment).
#
# Should implement and see what happens


# NOTE :
# Tried to capture overall advancement here in hope to maintain Runge-Kutta's
# methods accuracy. It occurs to be extremely difficult due to the prescence
# of state-altering conductors. We can alter solver state vector inside of this
# function, but we must be carefull with it
#
# In legacy code right-hand side was evaluated for fixed state
# (i.e. state vector did not affect objects of system at all).
#
# This made usage of higher order methods futile, because they reach accuracy by
# intermediate function evaluations (more evaluations for higher accuracy).
# This looses meaning when we return constant values for each intermediate
# evaluation on time step. That's why RK23 was used, solver with lowest order
# of accuracy (and thus with minimal intermediate function evlautions) among
# out-of-box solvers in scipy.
#
# Scipy solvers can be modified with custom procedures to capture behavior of
# state-altering conductors (such as UnderPass). This should be possible,
# however for now it was decided to make base-class method abstract
# (asking for ode system resolution procedure explicitly) and stick with
# basic solver using explicit Euler's method.
#
# This should not be a catastrophy, because even large commercial applications
# use frist-order methods. HYSYS use Euler's method as well, but they use implicit
# scheme. Implicit scheme is more robust in terms of stability and works better
# with stiff systems, but it requieres iterative solution of nonlinear equation.
# As long as we keep step size reasonably small Euler's method should do the job.


_VALID_OBJECTIVES = ["level", "pressure", "temperature"]


class DynamicNetwork(ABC):

    # Must figure out how to connect objects quick

    def __init__(self) -> None:
        self.control_volumes: list[ControlVolume] = []
        self.conductors: list[Conductor] = []
        self.objectives: dict[Conductor, Callable[[], float]] = {}
        self.solver: ExplicitEuler

        # Convenience parameters to track system-level state variables
        self._masses: NDArray[float64]
        self._energies: NDArray[float64]
        self._state_vector: NDArray[float64]

        # Vectors to hold system-level mass and energy flow rates
        self._mass_flows: NDArray[float64]
        self._energy_flows: NDArray[float64]
        self._flows: NDArray[float64]

        # Convenience attributes
        self._number_of_phases: int
        self._number_of_control_volumes: int
        self._state_vector_size: int

    def add_control_volume(self, control_volume: ControlVolume) -> None:
        if control_volume not in self.control_volumes:
            self.control_volumes.append(control_volume)

    def add_conductor(self, conductor: Conductor) -> None:
        if conductor not in self.conductors:
            self.conductors.append(conductor)

    def evaluate_size(self) -> None:
        # Fixates system's size, allocates arrays for global state tracking

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

        # NOTE :
        # Check if values are updated in dinctinoary for this arrangement, if it does not
        # work and values do not change we can try recasting stuff above into lambdas.
        # If this will not work too we still can record tuples

    def connect_elements(
            self, connection_map: dict[Conductor, tuple[ControlVolume, ControlVolume]]):

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

        # Map control volumes' state parameters to solver's state vector
        # m, E -> y

        for i, cv in enumerate(self.control_volumes):
            start = i*self._number_of_phases
            stop = start+self._number_of_phases
            self._masses[start:stop] = cv.state.mass
            self._energies[start:stop] = cv.state.energy

        self._state_vector[:self._state_vector_size//2] = self._masses
        self._state_vector[self._state_vector_size//2:] = self._energies

        return self._state_vector

    def map_vector_to_state(self, y: NDArray[float64]) -> None:

        # Map state vector into control volumes' state objects
        # y -> m, E

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

        # Cast net flow rates of control volumes into vector that corresponds
        # to state-vector os solver

        for i, cv in enumerate(self.control_volumes):
            start = i*self._number_of_phases
            stop = start+self._number_of_phases

            self._mass_flows[start:stop] = cv.net_mass_flow
            self._energy_flows[start:stop] = cv.net_energy_flow

        self._flows[:self._state_vector_size//2] = self._mass_flows
        self._flows[self._state_vector_size//2:] = self._energy_flows

        return self._flows

    @abstractmethod
    def ode_system(self, t: float, y: NDArray[float64]) -> NDArray[float64]:
        return self.map_flows_to_vector()

    def prepare_solver(self, time_step, start_time=0) -> None:
        self.solver = ExplicitEuler(
            fun=self.ode_system, t0=start_time,
            y0=self.map_state_to_vector(), time_step=time_step)

    @abstractmethod
    def advance(self):

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


class EmulsionTreater(DynamicNetwork):

    def __init__(self) -> None:
        super().__init__()
        self.control_volumes: list[SectionHorizontal]
        self.conductors: list[Valve | OverPass | UnderPass | CentrifugalPump]

    def ode_system(self, t, y):
        return self.map_flows_to_vector()

    def advance(self):
        pass

# Check how this stuff works, try to test some simple systems, look into mapping
# algorithms, investigate how other parts of system behaves, verify against legacy?


if __name__ == "__main__":
    from pytroleum.sdyna.opdata import OperationData, FlowData, fabric_state, fabric_flow
    from pytroleum.sdyna.convolumes import Atmosphere
    from pytroleum.tdyna.eos import factory_eos
    import CoolProp.constants as CoolConst
    from pprint import pprint
    import matplotlib.pyplot as plt

    s1 = SectionHorizontal(0.4, 1, 0, 1, lambda h: 0)
    s2 = SectionHorizontal(0, 1, 0.4, 1, lambda h: 0)
    atm = Atmosphere()
    vlv = Valve(1, 50e-3, 32e-3, 0.61, 1, 0.1, s2, atm)

    thermodynamic_state = (CoolConst.PT_INPUTS, 1e5, 300)

    s1.state = fabric_state(
        [factory_eos({"air": 1}, with_state=thermodynamic_state),
         factory_eos({"water": 1}, with_state=thermodynamic_state)],
        s1.compute_volume_with_level,
        np.array([2e5, 2e5]),
        np.array([300, 300]),
        np.array([1, 0.4]),
        False)

    s2.state = fabric_state(
        [factory_eos({"air": 1}, with_state=thermodynamic_state),
         factory_eos({"water": 1}, with_state=thermodynamic_state)],
        s2.compute_volume_with_level,
        np.array([2e5, 2e5]),
        np.array([300, 300]),
        np.array([1, 0.4]),
        False)

    inlet = Fixed([0, 1], sink=s2)
    inlet.flow = fabric_flow(
        [factory_eos({"air": 1}, with_state=thermodynamic_state),
         factory_eos({"water": 1}, with_state=thermodynamic_state)],
        np.array([1e5, 1e5]),
        np.array([300, 300]),
        np.pi*(50e-3)**2/4,
        0.9,
        np.array([0, 1], dtype=np.float64),
        False
    )

    vlv.opening = 0.7
    vlv.flow = fabric_flow(
        [factory_eos({"air": 1}, with_state=thermodynamic_state),
         factory_eos({"water": 1}, with_state=thermodynamic_state)],
        np.array([1e5, 1e5]),
        np.array([275, 275]),
        vlv.area_valve*vlv.opening,
        vlv.elevation,
        np.array([0, 0], dtype=np.float64),
        False
    )

    class GenericDynamic(DynamicNetwork):
        # Can't instantiate ABC, need dummy subclass
        def __init__(self) -> None:
            super().__init__()

        def ode_system(self, t, y):
            return super().ode_system(t, y)

        def advance(self):
            return super().advance()

    net = GenericDynamic()
    net.add_control_volume(s1)
    net.add_control_volume(s2)
    # net.add_control_volume(atm)
    net.add_conductor(vlv)
    net.add_conductor(inlet)
    net.evaluate_size()

    vlv.controller = PropIntDiff(0.4, 0.015, 0, 100, 0.45, [0, 1])
    vlv.controller.polarity = -1
    vlv.controller.norm_by = vlv.controller.setpoint

    net.bind_objective(("level", 1), s2, vlv)
    net.connect_elements({
        vlv: (s2, atm),
        inlet: (atm, s2)
    })

    s1.advance()
    s2.advance()
    vlv.advance()

    pprint(s2.state)
    print()

    net.prepare_solver(10)
    net.advance()
    # net.advance()

    # Final step should perform mapping too to capture last-step changes
    # or move mapping in advancement algorithm to the bottom of method, whatever
    # suits better.
    # net.map_vector_to_state(net.solver.y)

    pprint(s2.state)

    # Mass decreases, it works. Nice

    # Some simple simulation

    t = [0.]
    h = [s2.state.level[1]]
    signal = [vlv.controller.signal]
    error = [vlv.controller.error]
    for n in range(300):
        net.advance()
        t.append(t[-1]+net.solver.time_step)
        h.append(s2.state.level[1])
        signal.append(vlv.controller.signal)
        error.append(vlv.controller.error)
    t = np.array(t)
    h = np.array(h)
    signal = np.array(signal)
    error = np.array(error)

    fig, ax = plt.subplots()
    ax.set_title("Objective parameter")
    ax.plot(t/60, h*1e3)
    ax.grid(True)

    fig, ax = plt.subplots()
    ax.set_title("Control signal")
    ax.plot(t/60, signal*100)
    ax.grid(True)

    fig, ax = plt.subplots()
    ax.set_title("Control error")
    ax.plot(t/60, error*100)
    ax.grid(True)
    plt.show()
    # General algorithm seems to work fine.
