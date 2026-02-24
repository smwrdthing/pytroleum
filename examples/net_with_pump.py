import numpy as np
from numpy.typing import NDArray
from scipy.constants import g as gravity
import matplotlib.pyplot as plt

from pytroleum.sdyna import network as net
from pytroleum.sdyna import convolumes as cvs
from pytroleum.sdyna import conductors as cds
from pytroleum.sdyna import controllers as cts
from pytroleum.sdyna import opdata as opd
from pytroleum.tdyna import eos

VAPOR, LIQUID = 0, 1

INIT_PRESSURE = 1e5
INIT_TEMPERATURE = 280.0

INIT_LEVEL = 500e-3
MAX_LEVEL = 800e-3
MIN_LEVEL = 300e-3

INLET_DIAMETER = 50e-3
INLET_AREA = np.pi*INLET_DIAMETER**2/4

VAPOR_INFLOW = 0.5
LIQUID_INFLOW = 5.0
BACKPRESSURE = 3e5

RPM_CONVERSION = np.pi/30
FLOW_RATE_CONVERSION = 1e-3
EXTR_FLOW_RATE_FRACTION = 0.3
EXTR_HEAD_FRACTION = 0.22

RPM = 1500
ANGULAR_VELOCITY = RPM_CONVERSION*RPM

MAX_FLOW_RATE = 7*FLOW_RATE_CONVERSION
EXTR_FLOW_RATE = MAX_FLOW_RATE*EXTR_FLOW_RATE_FRACTION

ZERO_HEAD = 30
EXTR_HEAD = ZERO_HEAD*(EXTR_HEAD_FRACTION+1)

RESISTANCE_COEFF = 5.5

PUMP_REF = {"angular_velocity": ANGULAR_VELOCITY,
            "volume_flow_rates": [0.0, EXTR_FLOW_RATE, MAX_FLOW_RATE],
            "heads": [ZERO_HEAD, EXTR_HEAD, 0.0]}

PRESSURE_SETPOINT = 2e5

TIME_STEP = 0.5


class NetWithPump(net.DynamicNetwork):

    def __init__(self) -> None:

        super().__init__()

        self.section = cvs.SectionHorizontal(
            length_left_semiaxis=0.0,
            length_cylinder=5000e-3,
            length_right_semiaxis=0.0,
            diameter=1000e-3,
            volume_modificator=lambda level: 0.0)

        self.inlet = cds.Fixed(
            of_phase=[VAPOR, LIQUID],
            sink=self.section)

        self.pump = cds.CentrifugalPump(
            of_phase=LIQUID,
            source=self.section)

        self.vapor_valve = cds.Valve(
            of_phase=VAPOR,
            diameter_pipe=80e-3,
            diameter_valve=50e-3,
            discharge_coefficient=0.8,
            opening=0.0,
            elevation=self.section.diameter,
            source=self.section)

        self.pump.characteristic_reference(**PUMP_REF)
        self.pump.max_angular_velocity = ANGULAR_VELOCITY
        self.pump.resistance_coeff = RESISTANCE_COEFF
        self.pump.flow_area = INLET_AREA

        self.section.state = opd.factory_state(
            equation_of_state=[eos.factory_eos({"air": 1}),
                               eos.factory_eos({"water": 1})],
            volume_fn=self.section.compute_volume_with_level,
            pressure=np.array([INIT_PRESSURE, INIT_PRESSURE]),
            temperature=np.array([INIT_TEMPERATURE, INIT_TEMPERATURE]),
            level=np.array([self.section.diameter, INIT_LEVEL]))

        self.inlet.flow = opd.factory_flow(
            equation_of_state=[eos.factory_eos({"air": 1}),
                               eos.factory_eos({"water": 1})],
            pressure=np.array([INIT_PRESSURE, INIT_PRESSURE]),
            temperature=np.array([INIT_TEMPERATURE, INIT_TEMPERATURE]),
            flow_cross_area=INLET_AREA,
            elevation=self.section.diameter,
            mass_flowrate=np.array([VAPOR_INFLOW, LIQUID_INFLOW]))

        self.vapor_valve.flow = opd.factory_flow(
            equation_of_state=[eos.factory_eos({"air": 1}),
                               eos.factory_eos({"water": 1})],
            pressure=np.array([INIT_PRESSURE, INIT_PRESSURE]),
            temperature=np.array([INIT_TEMPERATURE, INIT_TEMPERATURE]),
            flow_cross_area=self.vapor_valve.area_pipe,
            elevation=self.section.diameter,
            mass_flowrate=np.array([VAPOR_INFLOW, 0.0]))

        self.vapor_valve.controller = cts.PropIntDiff(
            1.1, 0.02, 0.0, 1.0,
            PRESSURE_SETPOINT, (0, 1),
            polarity=-1, norm_by=PRESSURE_SETPOINT)

        self.pump.flow = opd.factory_flow(
            equation_of_state=[eos.factory_eos({"air": 1}),
                               eos.factory_eos({"water": 1})],
            pressure=np.array([INIT_PRESSURE, INIT_PRESSURE]),
            temperature=np.array([INIT_TEMPERATURE, INIT_TEMPERATURE]),
            flow_cross_area=INLET_AREA,
            elevation=0.0,
            mass_flowrate=np.array([0.0, 0.0]))

        self.pump.controller = cts.StartStop(
            MAX_LEVEL, MIN_LEVEL, 1.0, 0.0)

        self.pump.sink.state.pressure[:] = BACKPRESSURE

        self.add_control_volume(self.section)
        self.add_conductor(self.vapor_valve)
        self.add_conductor(self.pump)

        self.evaluate_size()

    def ode_system(self, t, y):
        return super().ode_system(t, y)

    def advance(self):

        self.map_vector_to_state(self.solver.y)

        self.section.advance()

        self.inlet.advance()

        if isinstance(self.vapor_valve.controller, cts.PropIntDiff):
            self.vapor_valve.controller.control(
                self.solver.time_step, self.section.state.pressure[VAPOR])
        self.vapor_valve.advance()

        if isinstance(self.pump.controller, cts.StartStop):
            self.pump.controller.control(
                probe=self.section.state.level[LIQUID],
                invert=True)
        self.pump.advance()

        self.solver.step()


net_with_pump = NetWithPump()
net_with_pump.prepare_solver(TIME_STEP)

total_time = 60*60*5
num_of_steps = int(total_time/TIME_STEP)
num_of_steps = num_of_steps + 1*(num_of_steps*TIME_STEP < total_time)

time = []
pressure, temperature, level = [], [], []
pump_flow = []
pump_signal = []

valve_flow = []
valve_signal = []
valve_error = []

for _ in range(num_of_steps):

    time.append(net_with_pump.solver.t)

    # Record params
    pressure.append(net_with_pump.section.state.pressure[VAPOR])
    temperature.append(net_with_pump.section.state.temperature[VAPOR])
    level.append(net_with_pump.section.state.level[LIQUID])

    # Record controls
    if net_with_pump.pump.controller:
        pump_signal.append(net_with_pump.pump.controller._signal)

    if isinstance(net_with_pump.vapor_valve.controller, cts.PropIntDiff):
        valve_signal.append(net_with_pump.vapor_valve.controller._signal)
        valve_error.append(net_with_pump.vapor_valve.controller._error)

    # Record outflow
    pump_flow.append(net_with_pump.pump.flow.mass_flow_rate[LIQUID])
    valve_flow.append(net_with_pump.vapor_valve.flow.mass_flow_rate[VAPOR])

    net_with_pump.advance()

time = np.array(time)

pressure = np.array(pressure)
temperature = np.array(temperature)
level = np.array(level)

pump_flow = np.array(pump_flow)

pump_signal = np.array(pump_signal)
valve_signal = np.array(valve_signal)
valve_error = np.array(valve_error)

plot_results = True
if plot_results:
    from typing import Literal

    time_units: Literal["s", "min", "h"] = "min"
    level_units: Literal["m", "cm", "mm"] = "mm"
    pressure_units: Literal["Pa", "kPa", "bar", "MPa"] = "bar"
    temperature_units: Literal["C", "K"] = "C"
    volume_units: Literal["m^3", "l"] = "m^3"
    mass_units: Literal["kg"] = "kg"
    flow_rate_units: Literal["kg/s", "kg/min", "kg/h"] = "kg/s"

    signal_units: Literal["-", "%"] = "%"

    if time_units == "s":
        time_scale = 1
    if time_units == "min":
        time_scale = 1/60
    if time_units == "s":
        time_scale = 1/60/60

    if level_units == "m":
        level_scale = 1
    if level_units == "cm":
        level_scale = 100
    if level_units == "mm":
        level_scale = 1000

    if pressure_units == "Pa":
        pressure_scale = 1
    if pressure_units == "kPa":
        pressure_scale = 1e-3
    if pressure_units == "bar":
        pressure_scale = 1e-5
    if pressure_units == "MPa":
        pressure_scale = 1e-6

    if temperature_units == "K":
        temperature_offset = 0.0
    if temperature_units == "C":
        temperature_offset = -273.15

    if volume_units == "m^3":
        volume_scale = 1
    if volume_units == "l":
        volume_scale = 1000

    if mass_units == "kg":
        mass_scale = 1

    if flow_rate_units == "kg/s":
        flow_rate_scale = 1
    if flow_rate_units == "kg/min":
        flow_rate_scale = 60
    if flow_rate_units == "kg/h":
        flow_rate_scale = 60*60*60

    if signal_units == "-":
        signal_scale = 1
    if signal_units == "%":
        signal_scale = 100

    scaled_time = time*time_scale
    scaled_level = level*level_scale
    sclaed_pressure = pressure*pressure_scale
    scaled_temperature = temperature+temperature_offset

    scaled_pump_flow_rate = pump_flow*flow_rate_scale
    scaled_pump_signal = pump_signal*signal_scale
    scaled_valve_signal = valve_signal*signal_scale

    cm_to_inch = 1/2.54
    fig_width = 17*cm_to_inch
    fig_height = 22*cm_to_inch
    show_grid = True

    # Plot pump & resistance network characteristic
    k1, k2, k3 = net_with_pump.pump.coefficients
    omega = net_with_pump.pump.max_angular_velocity
    xi = net_with_pump.pump.resistance_coeff
    dp = BACKPRESSURE-net_with_pump.section.state.pressure[LIQUID]
    rho = net_with_pump.section.state.density[LIQUID]

    Q = np.linspace(0, np.max(PUMP_REF["volume_flow_rates"]), 300)
    v = Q/net_with_pump.pump.flow_area

    dH_pump = k1*omega**2 + 2*k2*omega*Q - k3*Q**2
    dH_res = dp/rho/gravity + xi*v**2/2/gravity

    fig, ax = plt.subplots()
    ax.set_title("Pump curve")
    ax.set_xlabel("Q, m^3/h")
    ax.set_ylabel("H, m")
    ax.grid(True)

    ax.plot(Q*60*60, dH_pump, Q*60*60, dH_res)
    ax.plot(
        np.array(PUMP_REF["volume_flow_rates"])*60*60,
        np.array(PUMP_REF["heads"]), "ko")
    ax.legend(["pump curve", "resistance"])

    level_row, pressure_row, temperature_row = 0, 1, 2
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), nrows=3)
    ax[level_row].set_ylabel(f"h [{level_units}]")
    ax[level_row].grid(show_grid)
    ax[level_row].plot(scaled_time, scaled_level)

    ax[level_row].plot(
        scaled_time, np.ones_like(time) * MAX_LEVEL*level_scale, "k--")
    ax[level_row].plot(
        scaled_time, np.ones_like(time) * MIN_LEVEL*level_scale, "k--")

    ax[pressure_row].set_ylabel(f"p [{pressure_units}]")
    ax[pressure_row].grid(show_grid)
    ax[pressure_row].plot(scaled_time, sclaed_pressure)

    ax[temperature_row].set_xlabel(f"t [{time_units}]")
    ax[temperature_row].set_ylabel(f"T [{temperature_units}]")
    ax[temperature_row].grid(show_grid)
    ax[temperature_row].plot(scaled_time, scaled_temperature)

    fig_width = 30*cm_to_inch
    fig_height = 12*cm_to_inch
    signal_col, flow_rate_col = 0, 1
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), ncols=2)

    ax[signal_col].set_xlabel(f"t, [{time_units}]")
    ax[signal_col].set_ylabel(f"signal [{signal_units}]")
    ax[signal_col].grid(show_grid)
    ax[signal_col].plot(scaled_time, scaled_pump_signal)
    ax[signal_col].plot(scaled_time, scaled_valve_signal)
    ax[signal_col].legend(["pump", "valve"])

    ax[flow_rate_col].plot(scaled_time, scaled_pump_flow_rate)
    ax[flow_rate_col].set_xlabel(f"t [{time_units}]")
    ax[flow_rate_col].set_ylabel(f"G [{flow_rate_units}]")
    ax[flow_rate_col].grid(show_grid)

    plt.show()

# NOTE
# Noice!
