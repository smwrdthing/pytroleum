import numpy as np
import matplotlib.pyplot as plt


from numpy.typing import NDArray
from pytroleum.sdyna.convolumes import Atmosphere, SectionHorizontal
from pytroleum.sdyna.conductors import Fixed, Valve, PhaseInterface
from pytroleum.sdyna.controllers import PropIntDiff
from pytroleum.sdyna.network import DynamicNetwork

from pytroleum import meter
from pytroleum.tdyna.eos import factory_eos
from pytroleum.sdyna.opdata import factory_state, factory_flow


class SimpleNetwork(DynamicNetwork):

    def __init__(self) -> None:
        super().__init__()

        section = SectionHorizontal(0, 3000e-3, 0, 1000e-3, lambda h: 0)

        inlet = Fixed([0, 1])
        valve_vapor = Valve(0, 45e-3, 32e-3, 0.7, 1.0, 0.0)
        valve_liquid = Valve(1, 45e-3, 25e-3, 0.61, 1.0, 0.0)
        interface = PhaseInterface((0, 1), section, 0)  # type: ignore

        section.state = factory_state(
            [
                factory_eos({"air": 1}),
                factory_eos({"water": 1})
            ],
            section.compute_volume_with_level,
            np.array([1e5, 1e5]),
            np.array([300., 300.]),
            np.array([section.diameter,
                      section.diameter/3]))

        inlet.flow = factory_flow(
            [
                factory_eos({"air": 1}),
                factory_eos({"water": 1})
            ],
            np.array([1e5, 1e5]),
            np.array([300., 300.]),
            np.pi*(80e-3)**2/4,
            0.0,
            np.array([0.05, 0.15]))

        valve_vapor.flow = factory_flow(
            [
                factory_eos({"air": 1}),
                factory_eos({"water": 1})
            ],
            np.array([1e5, 1e5]),
            np.array([300., 300.]),
            valve_vapor.diameter_pipe**2/4,
            0.0,
            np.array([0.0, 0.0]))

        valve_liquid.flow = factory_flow(
            [
                factory_eos({"air": 1}),
                factory_eos({"water": 1})
            ],
            np.array([1e5, 1e5]),
            np.array([300., 300.]),
            valve_liquid.diameter_pipe**2/4,
            0.0,
            np.array([0.0, 0.0]))

        interface.flow = factory_flow(
            [
                factory_eos({"air": 1}),
                factory_eos({"water": 1})
            ],
            np.array([1e5, 1e5]),
            np.array([300, 300]),
            meter.area_planecut_section_horiz_ellipses(
                section.length_left_semiaxis,
                section.length_cylinder,
                section.length_right_semiaxis,
                section.diameter,
                section.state.level[1]),
            0.0,
            np.array([0.0, 0.0]))
        interface.heat_transfer_coeff = 20
        interface.saturation_state = Atmosphere().state

        valve_vapor.controller = PropIntDiff(
            0.25, 0.005, 0, 1,
            2e5, (0, 1), 1/25_000*np.inf, polarity=-1,
            norm_by=2e5)

        valve_liquid.controller = PropIntDiff(
            0.8, 0.002, 0, 1,
            section.diameter*2/3, (0, 1), 1/25_000*np.inf, polarity=-1,
            norm_by=section.diameter*2/3)

        self.add_control_volume(section)

        self.add_conductor(inlet)
        self.add_conductor(valve_vapor)
        self.add_conductor(valve_liquid)
        self.add_conductor(interface)

        self.bind_objective(("pressure", 0), section, valve_vapor)
        self.bind_objective(("level", 1), section, valve_liquid)

        self.connect_elements({
            inlet: (Atmosphere(), section),
            valve_vapor: (section, Atmosphere()),
            valve_liquid: (section, Atmosphere()),
            interface: (Atmosphere(), section),
        })

        self.evaluate_size()

    def ode_system(self, t, y):
        return super().ode_system(t, y)

    def advance(self):
        return super().advance()


net = SimpleNetwork()

net.prepare_solver(step := 5.0)
total = 60*60*4

time, level, pressure, temperature = [], [], [], []
inflow, outflow = [], []
signals, errors = [], []

for _ in range(int(total/step)):
    time.append(net.solver.t)

    level.append(net.control_volumes[0].state.level.copy())
    pressure.append(net.control_volumes[0].state.pressure.copy())
    temperature.append(net.control_volumes[0].state.temperature.copy())

    inflow.append(net.conductors[0].flow.mass_flow_rate.copy())
    outflow.append(net.conductors[2].flow.mass_flow_rate.copy())

    signals.append([
        net.conductors[1].controller._signal,  # type: ignore
        net.conductors[2].controller._signal  # type: ignore
    ])
    errors.append([
        net.conductors[1].controller._error,  # type: ignore
        net.conductors[2].controller._error  # type: ignore
    ])

    net.advance()

time = np.array(time)

level = np.array(level)
pressure = np.array(pressure)
temperature = np.array(temperature)

inflow = np.array(inflow)
outflow = np.array(outflow)

signals = np.array(signals)
errors = np.array(errors)

fig, ax = plt.subplots()
ax.set_xlabel("time, min")
ax.set_ylabel("level, mm")
ax.plot(time/60, level[:, 1]*1e3)
ax.grid(True)

fig, ax = plt.subplots()
ax.set_xlabel("time, min")
ax.set_ylabel(r"temperature, $\degree$C")
ax.set_ylim((24, 30))
ax.plot(time/60, temperature-273.15)
ax.grid(True)

fig, ax = plt.subplots()
ax.plot(time/60, pressure[:, 0]/1e5)
ax.set_xlabel("time, min")
ax.set_ylabel("pressure, bar (abs.)")
ax.grid(True)

fig, ax = plt.subplots()
ax.plot(time/60, inflow[:, 1])
ax.plot(time/60, outflow[:, 1])
ax.plot(time/60, inflow[:, 1]-outflow[:, 1])
ax.set_xlabel("time, min")
ax.set_ylabel("flow rates (vap.), kg/s")
ax.legend(["in", "out", "net"])
ax.grid(True)

fig, ax = plt.subplots()
ax.plot(time/60, signals*100)
ax.set_xlabel("time, min")
ax.set_ylabel("signals in system, %")
ax.legend(["pressure", "level"])
ax.grid(True)

fig, ax = plt.subplots()
ax.plot(time/60, errors*100)
ax.set_xlabel("time, min")
ax.set_ylabel("errors in system, %")
ax.legend(["pressure", "level"])
ax.grid(True)

plt.show()

# Looks like solver works, but there are some noize in the solutuion, we could filetr it,
# out of course, but I am curious about reasons of such instabiliteis
#
# nevermind, it was numpy type shenanigans
#
# We can merge sdyna_content branch to main I suppose
