import numpy as np
from scipy.optimize import fsolve

from enum import IntEnum, auto


# Enums are used for indexing in the flowsheet object iterables to make mapping between
# order of elements in iterables and specific "places" in system clearer
#
# Different enums for different values violate DRY slightly, but much easier to comprehend
# and more robust
#
# Using enums is convenient, because we choose how to store values only ones - during enum
# definition, later on we just rely on chosen indexing rules


DIVIDER_LENGTH = 80
MINOR_DIVIDER = '-'*DIVIDER_LENGTH
MAJOR_DIVIDER = '='*DIVIDER_LENGTH


def _minor_divider():
    print(MINOR_DIVIDER)


def _major_divider():
    print(MAJOR_DIVIDER)


class FlowSpec(IntEnum):
    # No need for type annotations here as we use IntEnum specifically
    O, OVERFLOW = 0, 0
    U, UNDERFLOW = 1, 1
    IN, INLET = 2, 2
    R, RECIRCULATION = 3, 3
    FOR, FORWARD = 4, 4
    REV, REVERSE = 5, 5

    # SIZE is a convenience variable to instantiate containers (lists) for values, using
    # auto ensures that it is larger than last index by one, thus enabling using np.zeros
    # for instantiation
    SIZE = auto()


class PressureSpec(IntEnum):
    OB, OVERFLOW_BACK = 0, 0
    UB, UNDERFLOW_BACK = 1, 1
    O, OVERFLOW = 2, 2
    U, UNDERFLOW = 3, 3
    IN, INLET = 4, 4
    N, NODE = 5, 5

    SIZE = auto()


class ResistanceSpec(IntEnum):
    O, OVERFLOW = 0, 0
    U, UNDERFLOW = 1, 1
    IN, INLET = 2, 2
    OV, OVERFLOW_VALVE = 3, 3
    UV, UNDERFLOW_VALVE = 4, 4

    SIZE = auto()


class FlowSheet:

    # We could define different models for pressure-flow relationships
    # by making this ABC and stipulating inheritance for specific models,
    # this would require decorating some methods with @abstractmethod and moving
    # actual implementations in repsective calsses. For now we leave it as is

    def __init__(self) -> None:
        self.resistance: list[float] = np.zeros(ResistanceSpec.SIZE).tolist()
        self.pressure: list[float] = np.zeros(PressureSpec.SIZE).tolist()
        self.flow_rate: list[float] = np.zeros(FlowSpec.SIZE).tolist()

    def residuals(self) -> tuple:

        # Unpacking hell in 3, 2, 1...

        K_o = self.resistance[ResistanceSpec.OVERFLOW]
        K_u = self.resistance[ResistanceSpec.UNDERFLOW]
        K_in = self.resistance[ResistanceSpec.INLET]

        # Here openings are absorbed into coefficients for valves
        K_ov = self.resistance[ResistanceSpec.OVERFLOW_VALVE]
        K_uv = self.resistance[ResistanceSpec.UNDERFLOW_VALVE]

        P_o = self.pressure[PressureSpec.OVERFLOW]
        P_u = self.pressure[PressureSpec.UNDERFLOW]
        P_in = self.pressure[PressureSpec.INLET]
        P_ob = self.pressure[PressureSpec.OVERFLOW_BACK]
        P_ub = self.pressure[PressureSpec.UNDERFLOW_BACK]
        P_n = self.pressure[PressureSpec.NODE]

        Q_o = self.flow_rate[FlowSpec.OVERFLOW]
        Q_u = self.flow_rate[FlowSpec.UNDERFLOW]
        Q_in = self.flow_rate[FlowSpec.INLET]

        res = (Q_o - K_o*np.sqrt(P_n-P_o),
               Q_u - K_u*np.sqrt(P_n-P_u),
               Q_in - K_in*np.sqrt(P_in-P_n),
               Q_o - K_ov*np.sqrt(P_o-P_ob),
               Q_u - K_uv*np.sqrt(P_u-P_ub))

        return res

    def solve_from_backpressures(
            self, inlet_flow_rate: float, backpressure: tuple[float, float]) -> None:

        # Those two values stay constant, no need to reassign them each time
        # we call objective function
        self.pressure[PressureSpec.OVERFLOW_BACK] = backpressure[PressureSpec.OB]
        self.pressure[PressureSpec.UNDERFLOW_BACK] = backpressure[PressureSpec.UB]
        # ^^^^^ here is a reason to make OVERFLOW_BACK 0 and UNDERFLOW_BACK 1 btw

        self._solve_with_underflow_spec(
            fsolve(lambda underflow_flow_rate:
                   self._solve_with_underflow_spec(underflow_flow_rate) -
                   inlet_flow_rate, inlet_flow_rate)[0])

    def solve_from_outflows(
            self, inlet_pressure: float, outflows: tuple[float, float]) -> None:

        # This set of inputs is easier to handle, here we can get away without nonlinear
        # solver.
        self.pressure[PressureSpec.INLET] = inlet_pressure

        K_o = self.resistance[ResistanceSpec.OVERFLOW]
        K_u = self.resistance[ResistanceSpec.UNDERFLOW]
        K_in = self.resistance[ResistanceSpec.INLET]
        K_ov = self.resistance[ResistanceSpec.OVERFLOW_VALVE]
        K_uv = self.resistance[ResistanceSpec.UNDERFLOW_VALVE]

        self.flow_rate[FlowSpec.OVERFLOW] = Q_o = outflows[FlowSpec.OVERFLOW]
        self.flow_rate[FlowSpec.UNDERFLOW] = Q_u = outflows[FlowSpec.UNDERFLOW]

        self.flow_rate[FlowSpec.INLET] = Q_in = Q_o + Q_u

        self.pressure[PressureSpec.NODE] = P_n = inlet_pressure - \
            (Q_in/K_in)**2
        self.pressure[PressureSpec.OVERFLOW] = P_o = P_n - (Q_o/K_o)**2
        self.pressure[PressureSpec.UNDERFLOW] = P_u = P_n - (Q_u/K_u)**2

        self.pressure[PressureSpec.OVERFLOW_BACK] = P_o - (Q_o/K_ov)**2
        self.pressure[PressureSpec.UNDERFLOW_BACK] = P_u - (Q_u/K_uv)**2

    def account_for_recirculation(self, recirculation_rate: float = 0.02) -> None:

        Q_o = self.flow_rate[FlowSpec.OVERFLOW]
        Q_in = self.flow_rate[FlowSpec.INLET]

        Q_r = Q_in*recirculation_rate
        Q_for = Q_in + Q_r
        Q_rev = Q_o + Q_r

        self.flow_rate[FlowSpec.RECIRCULATION] = Q_r
        self.flow_rate[FlowSpec.FORWARD] = Q_for
        self.flow_rate[FlowSpec.REVERSE] = Q_rev

    def summary(self):

        # Unit conversions and precision are currently hardcoded here, this might be
        # changed later on if needed
        to_bar = 1/1e5
        to_liter_per_min = 60*1e3

        _major_divider()
        print("FLOWSHEET SUMMARY")
        _major_divider()

        print("RESISTANCES :: ")
        print(f"Overflow : {self.resistance[ResistanceSpec.O]:.3e}")
        print(f"Underflow : {self.resistance[ResistanceSpec.U]:.3e}")
        print(f"Inlet : {self.resistance[ResistanceSpec.IN]:.3e}")
        print(f"Overflow valve : {self.resistance[ResistanceSpec.OV]:.3e}")
        print(f"Underflow valve : {self.resistance[ResistanceSpec.UV]:.3e}")

        _minor_divider()

        print("FLOW RATES ::")
        print(
            f"Overflow : {self.flow_rate[FlowSpec.O]*to_liter_per_min:.3f} l/min")
        print(
            f"Underflow : {self.flow_rate[FlowSpec.U]*to_liter_per_min:.3f} l/min")
        print(
            f"Inlet : {self.flow_rate[FlowSpec.IN]*to_liter_per_min:.3f} l/min")
        print(
            f"Recirculation : {self.flow_rate[FlowSpec.R]*to_liter_per_min:.3f} l/min")
        print(
            f"Forward : {self.flow_rate[FlowSpec.FOR]*to_liter_per_min:.3f} l/min")
        print(
            f"Reverse : {self.flow_rate[FlowSpec.REV]*to_liter_per_min:.3f} l/min")

        _minor_divider()

        print("PRESSURES :: ")
        print(
            f"Overflow : {self.pressure[PressureSpec.O]*to_bar: .3f} bar(a)")
        print(
            f"Oveflow (back) : {self.pressure[PressureSpec.OB]*to_bar: .3f} bar(a)")
        print(
            f"Underflow : {self.pressure[PressureSpec.U]*to_bar:.3f} bar(a)")
        print(
            f"Undefrlow (back) : {self.pressure[PressureSpec.UB]*to_bar:.3f} bar(a)")
        print(
            f"Inlet : {self.pressure[PressureSpec.IN]*to_bar:.3f} bar(a)")
        print(
            f"Node : {self.pressure[PressureSpec.N]*to_bar:.3f} bar(a)")

        _minor_divider()

        F = self.flow_rate[FlowSpec.O]/self.flow_rate[FlowSpec.IN]
        PDR = (
            self.pressure[PressureSpec.IN]-self.pressure[PressureSpec.O])/(
            self.pressure[PressureSpec.IN]-self.pressure[PressureSpec.U])

        print("SECONDARIES :: ")
        print(f"Split ratio : {F*100: .3f} %")
        print(f"Pressure difference ratio : {PDR: .3f}")

        _major_divider()
        print("END OF SUMMARY")
        _major_divider()

    def _solve_with_underflow_spec(self, underflow_flow_rate: float) -> float:

        # objective function performs reassignments, so we don't need to do this manually
        # after solution is found

        P_ob = self.pressure[PressureSpec.OVERFLOW_BACK]
        P_ub = self.pressure[PressureSpec.UNDERFLOW_BACK]

        K_o = self.resistance[ResistanceSpec.OVERFLOW]
        K_u = self.resistance[ResistanceSpec.UNDERFLOW]
        K_in = self.resistance[ResistanceSpec.INLET]
        K_ov = self.resistance[ResistanceSpec.OVERFLOW_VALVE]
        K_uv = self.resistance[ResistanceSpec.UNDERFLOW_VALVE]

        k_o = K_ov/K_o  # auxiliary variable

        self.flow_rate[FlowSpec.UNDERFLOW] = Q_u = underflow_flow_rate
        self.pressure[PressureSpec.UNDERFLOW] = P_u = (Q_u/K_uv)**2 + P_ub

        self.pressure[PressureSpec.NODE] = P_n = (Q_u/K_u)**2 + P_u

        self.pressure[PressureSpec.OVERFLOW] = P_o = (
            P_n + k_o**2*P_ob)/(1 + k_o**2)
        self.flow_rate[FlowSpec.OVERFLOW] = Q_o = K_o*np.sqrt(P_n - P_o)

        self.flow_rate[FlowSpec.INLET] = Q_in = Q_o + Q_u
        self.pressure[PressureSpec.INLET] = (Q_in/K_in)**2+P_n

        return Q_in  # for nonlinear solver


if __name__ == "__main__":

    def compute_resistance(density, area, discharge_coeff):
        # auxiliary function
        return discharge_coeff*area/np.sqrt(density)

    # Instantiate object
    flowsheet = FlowSheet()

    # Fluid properties
    oil_fraction = 10/100
    water_fraction = 1-oil_fraction

    oil_density = 830
    water_density = 1000
    inlet_density = water_density*water_fraction + oil_density*oil_fraction

    # Diameters
    overflow_diameter = 3e-3
    underflow_diameter = 12e-3
    inlet_diameter = 25e-3

    overflow_valve_diameter = 6e-3
    underflow_valve_diameter = 12e-3

    # Areas
    overflow_area = np.pi/4*overflow_diameter**2
    underflow_area = np.pi/4*underflow_diameter**2
    inlet_area = np.pi/4*inlet_diameter**2

    overflow_valve_area = np.pi/4*overflow_valve_diameter**2
    underflow_valve_area = np.pi/4*underflow_valve_diameter**2

    # Discharge coefficients
    overflow_discharge_coeff = 0.2
    underflow_discharge_coeff = 0.4
    inlet_discharge_coeff = 0.1

    overflow_valve_discharge_coeff = 0.61
    underflow_valve_discharge_coeff = 0.61

    # Openings
    overflow_valve_opening = 1.0
    underflow_valve_opening = 1.0

    # Collecting inputs
    overflow_inputs = (oil_density,
                       overflow_area,
                       overflow_discharge_coeff)

    underflow_inputs = (water_density,
                        underflow_area,
                        underflow_discharge_coeff)

    inlet_inputs = (inlet_density,
                    inlet_area,
                    inlet_discharge_coeff)

    overflow_valve_inputs = (oil_density,
                             overflow_valve_area*overflow_valve_opening,
                             overflow_valve_discharge_coeff)

    underflow_valve_inputs = (water_density,
                              underflow_valve_area*underflow_valve_opening,
                              underflow_valve_discharge_coeff)

    # Set resistances
    flowsheet.resistance[ResistanceSpec.O] = compute_resistance(
        *overflow_inputs)
    flowsheet.resistance[ResistanceSpec.U] = compute_resistance(
        *underflow_inputs)
    flowsheet.resistance[ResistanceSpec.IN] = compute_resistance(
        *inlet_inputs)
    flowsheet.resistance[ResistanceSpec.OV] = compute_resistance(
        *overflow_valve_inputs)
    flowsheet.resistance[ResistanceSpec.UV] = compute_resistance(
        *underflow_valve_inputs)

    # Case 1 - fixed backpressure & given flow rate
    print("CASE 1")
    Q_in = 1e-3
    backpressures = (1.5e5, 4.5e5)
    flowsheet.solve_from_backpressures(Q_in, backpressures)
    flowsheet.account_for_recirculation()
    flowsheet.summary()

    # Summary looks fine, matches sanbox code results

    print()
    _minor_divider()
    _minor_divider()
    _minor_divider()
    print()

    # Case 2 - fxied outflow and inlet pressure
    # just for fun, let's take values from previous case
    print("CASE 2")
    P_in = flowsheet.pressure[PressureSpec.IN]
    Q_o = flowsheet.flow_rate[FlowSpec.O]
    Q_u = flowsheet.flow_rate[FlowSpec.U]
    outflows = (Q_o, Q_u)

    flowsheet.solve_from_outflows(P_in, outflows)
    flowsheet.account_for_recirculation()
    flowsheet.summary()
    # Both give same result
