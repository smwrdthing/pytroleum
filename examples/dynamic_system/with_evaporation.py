from typing import TYPE_CHECKING
from copy import copy

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
if TYPE_CHECKING:
    from pytroleum.tdyna.CoolStub import AbstractState  # type: ignore
else:
    from CoolProp.CoolProp import AbstractState
import CoolProp.constants as CoolConst
from pprint import pprint

from pytroleum.sdyna import network as net
from pytroleum.sdyna import conductors as cds
from pytroleum.sdyna import convolumes as cvs
from pytroleum.sdyna import controllers as cts
from pytroleum.sdyna.opdata import factory_state, factory_flow
from pytroleum.tdyna.eos import factory_eos
from pytroleum import meter


class SimpleNetwork(net.DynamicNetwork):

    def __init__(self) -> None:
        super().__init__()

        self.conductors: list[cds.Conductor | cds.PhaseInterface]

        cv = cvs.SectionHorizontal(
            length_left_semiaxis=700e-3,
            length_cylinder=5000e-3,
            length_right_semiaxis=700e-3,
            diameter=1000e-3,
            volume_modificator=lambda h: 0)

        pressure, temperature = np.ones(
            2)*1e5, np.array([20+273.15, 5+273.15])
        level = np.array([cv.diameter, cv.diameter/2])
        state_equations = [
            factory_eos({"methane": 0.5, "ethane": 0.5}),
            factory_eos({"water": 1})
        ]

        cv.state = factory_state(
            state_equations, cv.compute_volume_with_level,  # type: ignore
            pressure.copy(), temperature.copy(), level.copy())
        cv.advance()

        interface = cds.PhaseInterface(
            of_phase=(0, 1),
            in_control_volume=cv,  # type: ignore
            evaporation_coefficient=1e-3)
        interface.heat_transfer_coeff = 2500

        interface.flow = factory_flow(
            state_equations,  # type: ignore
            pressure.copy(),
            temperature.copy(),
            meter.area_planecut_cover_ellipse(
                cv.length_left_semiaxis, cv.length_cylinder, cv.length_right_semiaxis),
            0.0,
            np.array([0.0, 0.0]))

        sat_pressure = 1.1*pressure.copy()
        sat_temperature = temperature.copy()

        interface.saturation_state = factory_state(
            state_equations, cv.compute_volume_with_level,  # type: ignore
            sat_pressure.copy(), sat_temperature.copy(), level.copy())

        self.control_volumes.append(cv)
        self.conductors.append(interface)

        self.connect_elements({interface: (cvs.Atmosphere(), cv)})

    def ode_system(self, t, y):
        return super().ode_system(t, y)

    def advance(self):
        return super().advance()


simple = SimpleNetwork()

total_time = 2.5
time_step = 0.1
total_steps = int(total_time/time_step)
# total_steps = 100

simple.evaluate_size()
simple.prepare_solver(time_step)
simple.advance()

time = []
evaporation_flow_rate = []
heat_flow = []
temperature = []
pressure = []
vessel_pressure = []

for step in range(total_steps):

    time.append(simple.solver.t)
    heat_flow.append(simple.conductors[0].flow.energy_flow[0])
    pressure.append(simple.control_volumes[0].state.pressure.copy())
    temperature.append(simple.control_volumes[0].state.temperature.copy())
    evaporation_flow_rate.append(simple.control_volumes[0]._net_mass_flow[0])

    # print(f"STEP {step} : {simple.solver.t: .3e} s")
    # pprint(simple.control_volumes[0].state)
    # print()
    # pprint(simple.conductors[0].flow)
    # print()

    simple.advance()

time = np.array(time)
heat_flow = np.array(heat_flow)
pressure = np.array(pressure)
temperature = np.array(temperature)
evaporation_flow_rate = np.array(evaporation_flow_rate)

# Mass and energy flow relation define how temperature changes, some coefficients for
# flow rate and heat flux models may drive system into unresolvable states

marker = ""
plt.plot(
    # Plot heat flow vs time
    # time, heat_flow/1e3, marker=marker,

    # Plot temperatures vs time
    time, temperature-273.15, marker=marker

    # Plot evaporation flow rate vs time
    # time, evaporation_flow_rate*1e3, marker=marker

    # Plot pressure vs time with normalized evaporation flow rate
    # time, pressure/1e5,
    # time, np.ones_like(
    #     time)/1e5*simple.conductors[0].saturation_state.pressure[0], '--r',
    # time, evaporation_flow_rate /
    # np.max(evaporation_flow_rate)*np.max(pressure)/1e5
)
plt.legend(["vapor", "liquid"])
plt.show()

# Coefficients and time step tuning are needed for specific models.
# If wast-pacing processes are presented step must be lower.

# Well, it seems to be adequate (in "expected behaviour" sense, not
# "entierely correct physics" sense) for some inputs. I consider it as a small win
