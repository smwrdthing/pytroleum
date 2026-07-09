from jet_huang import Design, Loc, OperationConditions, Phase, Requirements

_M_TO_MM = 1e3
_M2_TO_CM2 = 1e4
_PA_TO_MPA = 1e-6
_K_TO_C = 273.15

_LABEL_WIDTH = 60
_DIVIDER_LENGTH = 75
_MINOR_DIVIDER = '-' * _DIVIDER_LENGTH
_MAJOR_DIVIDER = '=' * _DIVIDER_LENGTH


def _major_header(title: str) -> None:
    print(_MAJOR_DIVIDER)
    print(title.center(_DIVIDER_LENGTH))
    print(_MAJOR_DIVIDER)


def _minor_header(title: str) -> None:
    print(_MINOR_DIVIDER)
    print(title.center(_DIVIDER_LENGTH))
    print(_MINOR_DIVIDER)


def print_row(label: str, value: str, unit: str = '') -> None:
    """Print one aligned result row."""
    print(f"{label:<{_LABEL_WIDTH}} {value} {unit}".rstrip())


def report_inputs(req: Requirements) -> None:
    _major_header("INPUTS")
    print_row("Working fluid", req.phase.name())
    print_row(
        "Vapor temperature at the nozzle inlet Tg",
        f"{req.temperature[Phase.P, Loc.IN] - _K_TO_C:.2f} C "
        f"({req.temperature[Phase.P, Loc.IN]:.2f} K)")
    print_row(
        "Vapor pressure at the nozzle inlet Pg",
        f"{req.pressure[Phase.P, Loc.IN] * _PA_TO_MPA:.4f}", "MPa")
    print_row(
        "Vapor temperature at the suction port Te",
        f"{req.temperature[Phase.S, Loc.IN] - _K_TO_C:.2f} C "
        f"({req.temperature[Phase.S, Loc.IN]:.2f} K)")
    print_row(
        "Vapor pressure at the suction port Pe",
        f"{req.pressure[Phase.S, Loc.IN] * _PA_TO_MPA:.4f}", "MPa")


def report_dimensions(design: Design) -> None:
    _major_header("DIMENSIONS")
    print_row(
        "Nozzle throat diameter dt",
        f"{design.diameter[Phase.P, Loc.TH] * _M_TO_MM:.2f}", "mm")
    print_row(
        "Nozzle exit diameter dp1",
        f"{design.diameter[Phase.P, Loc.EX] * _M_TO_MM:.2f}", "mm")
    print(_MINOR_DIVIDER)
    print_row(
        "Nozzle throat area At",
        f"{design.area[Phase.P, Loc.TH] * _M2_TO_CM2:.2f}", "cm2")
    print_row(
        "Nozzle exit area Ap1",
        f"{design.area[Phase.P, Loc.EX] * _M2_TO_CM2:.2f}", "cm2")
    print_row(
        "Primary flow area at plane y-y Apy",
        f"{design.area[Phase.P, Loc.CH] * _M2_TO_CM2:.2f}", "cm2")
    print_row(
        "Suction flow area at plane y-y Asy",
        f"{design.area[Phase.S, Loc.CH] * _M2_TO_CM2:.2f}", "cm2")
    print_row(
        "Constant-area section area A3",
        f"{design.area[Phase.M, Loc.AM] * _M2_TO_CM2:.2f}", "cm2")
    print_row(
        "Constant-area section diameter d3",
        f"{design.diameter[Phase.M, Loc.AM] * _M_TO_MM:.2f}", "mm")
    print_row(
        "Area ratio A3/At",
        f"{(design.area[Phase.M, Loc.AM] /
            design.area[Phase.P, Loc.TH]):.4f}")


def report_conditions(conditions: OperationConditions) -> None:
    _major_header("OPERATION")
    print_row(
        "Primary mass flow rate mp",
        f"{conditions.mass_flow_rate[Phase.P]:.4e}", "kg/s")
    print_row(
        "Suction mass flow rate ms",
        f"{conditions.mass_flow_rate[Phase.S]:.4e}", "kg/s")
    print_row(
        "Total mass flow rate mp + ms",
        f"{(conditions.mass_flow_rate[Phase.P] +
            conditions.mass_flow_rate[Phase.S]):.4e}",
        "kg/s")
    print_row(
        "Entrainment ratio omega = ms/mp",
        f"{(conditions.mass_flow_rate[Phase.S] /
            conditions.mass_flow_rate[Phase.P]):.4f}")
    print_row(
        "Compression ratio Pc/Pe",
        f"{(conditions.pressure[Phase.M, Loc.D] /
            conditions.pressure[Phase.S, Loc.IN]):.4f}")

    _minor_header("SECTION p1")
    print_row(
        "Pressure at nozzle exit Pp1",
        f"{conditions.pressure[Phase.P, Loc.EX] * _PA_TO_MPA:.4f}",
        "MPa")
    print_row(
        "Mach number at nozzle exit Mp1",
        f"{conditions.mach[Phase.P, Loc.EX]:.4f}")

    _minor_header("SECTION y-y")
    print_row(
        "Primary pressure at plane y-y Ppy",
        f"{conditions.pressure[Phase.P, Loc.CH] * _PA_TO_MPA:.4f}",
        "MPa")
    print_row(
        "Primary temperature at plane y-y Tpy",
        f"{conditions.temperature[Phase.P, Loc.CH] - _K_TO_C:.2f}", "C")
    print_row(
        "Primary Mach number at plane y-y Mpy",
        f"{conditions.mach[Phase.P, Loc.CH]:.4f}")
    print_row(
        "Primary velocity at plane y-y Vpy",
        f"{conditions.velocity[Phase.P, Loc.CH]:.2f}", "m/s")
    print_row(
        "Suction pressure at plane y-y Psy",
        f"{conditions.pressure[Phase.S, Loc.CH] * _PA_TO_MPA:.4f}",
        "MPa")
    print_row(
        "Suction temperature at plane y-y Tsy",
        f"{conditions.temperature[Phase.S, Loc.CH] - _K_TO_C:.2f}",
        "C")
    print_row(
        "Suction velocity at plane y-y Vsy",
        f"{conditions.velocity[Phase.S, Loc.CH]:.2f}", "m/s")

    _minor_header("SECTION m")
    print_row(
        "Mixed-flow pressure before shock Pm",
        f"{conditions.pressure[Phase.M, Loc.PS] * _PA_TO_MPA:.4f}",
        "MPa")
    print_row(
        "Mixed-flow temperature before shock Tm",
        f"{conditions.temperature[Phase.M, Loc.PS] - _K_TO_C:.2f}", "C")
    print_row(
        "Mixed-flow Mach number before shock Mm",
        f"{conditions.mach[Phase.M, Loc.PS]:.4f}")
    print_row(
        "Mixed-flow velocity before shock Vm",
        f"{conditions.velocity[Phase.M, Loc.PS]:.2f}", "m/s")

    _minor_header("SECTION 3")
    print_row(
        "Pressure at constant-area section exit P3",
        f"{conditions.pressure[Phase.M, Loc.AM] * _PA_TO_MPA:.4f}",
        "MPa")
    print_row(
        "Mach number at constant-area section exit M3",
        f"{conditions.mach[Phase.M, Loc.AM]:.4f}")

    _minor_header("SECTION c")
    print_row(
        "Discharge pressure at ejector exit Pc",
        f"{conditions.pressure[Phase.M, Loc.D] * _PA_TO_MPA:.4f}",
        "MPa")

    print(_MAJOR_DIVIDER)
