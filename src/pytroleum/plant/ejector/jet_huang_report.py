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
        f"{req.temperature[Phase.PRIMARY, Loc.INLET] - _K_TO_C:.2f} C "
        f"({req.temperature[Phase.PRIMARY, Loc.INLET]:.2f} K)")
    print_row(
        "Vapor pressure at the nozzle inlet Pg",
        f"{req.pressure[Phase.PRIMARY, Loc.INLET] * _PA_TO_MPA:.4f}", "MPa")
    print_row(
        "Vapor temperature at the suction port Te",
        f"{req.temperature[Phase.SECONDARY, Loc.INLET] - _K_TO_C:.2f} C "
        f"({req.temperature[Phase.SECONDARY, Loc.INLET]:.2f} K)")
    print_row(
        "Vapor pressure at the suction port Pe",
        f"{req.pressure[Phase.SECONDARY, Loc.INLET] * _PA_TO_MPA:.4f}", "MPa")


def report_dimensions(design: Design) -> None:
    _major_header("DIMENSIONS")
    print_row(
        "Nozzle throat diameter dt",
        f"{design.diameter[Phase.PRIMARY, Loc.THROAT] * _M_TO_MM:.2f}", "mm")
    print_row(
        "Nozzle exit diameter dp1",
        f"{design.diameter[Phase.PRIMARY, Loc.EXHAUST] * _M_TO_MM:.2f}", "mm")
    print(_MINOR_DIVIDER)
    print_row(
        "Nozzle throat area At",
        f"{design.area[Phase.PRIMARY, Loc.THROAT] * _M2_TO_CM2:.2f}", "cm2")
    print_row(
        "Nozzle exit area Ap1",
        f"{design.area[Phase.PRIMARY, Loc.EXHAUST] * _M2_TO_CM2:.2f}", "cm2")
    print_row(
        "Primary flow area at plane y-y Apy",
        f"{design.area[Phase.PRIMARY, Loc.CHOKE] * _M2_TO_CM2:.2f}", "cm2")
    print_row(
        "Suction flow area at plane y-y Asy",
        f"{design.area[Phase.SECONDARY, Loc.CHOKE] * _M2_TO_CM2:.2f}", "cm2")
    print_row(
        "Constant-area section area A3",
        f"{design.area[Phase.MIX, Loc.AFTERMIX] * _M2_TO_CM2:.2f}", "cm2")
    print_row(
        "Constant-area section diameter d3",
        f"{design.diameter[Phase.MIX, Loc.AFTERMIX] * _M_TO_MM:.2f}", "mm")
    print_row(
        "Area ratio A3/At",
        f"{(design.area[Phase.MIX, Loc.AFTERMIX] /
            design.area[Phase.PRIMARY, Loc.THROAT]):.4f}")


def report_conditions(conditions: OperationConditions) -> None:
    _major_header("OPERATION")
    print_row(
        "Primary mass flow rate mp",
        f"{conditions.mass_flow_rate[Phase.PRIMARY]:.4e}", "kg/s")
    print_row(
        "Suction mass flow rate ms",
        f"{conditions.mass_flow_rate[Phase.SECONDARY]:.4e}", "kg/s")
    print_row(
        "Total mass flow rate mp + ms",
        f"{(conditions.mass_flow_rate[Phase.PRIMARY] +
            conditions.mass_flow_rate[Phase.SECONDARY]):.4e}",
        "kg/s")
    print_row(
        "Entrainment ratio omega = ms/mp",
        f"{(conditions.mass_flow_rate[Phase.SECONDARY] /
            conditions.mass_flow_rate[Phase.PRIMARY]):.4f}")
    print_row(
        "Compression ratio Pc/Pe",
        f"{(conditions.pressure[Phase.MIX, Loc.DRAIN] /
            conditions.pressure[Phase.SECONDARY, Loc.INLET]):.4f}")

    _minor_header("SECTION p1")
    print_row(
        "Pressure at nozzle exit Pp1",
        f"{conditions.pressure[Phase.PRIMARY, Loc.EXHAUST] * _PA_TO_MPA:.4f}",
        "MPa")
    print_row(
        "Mach number at nozzle exit Mp1",
        f"{conditions.mach[Phase.PRIMARY, Loc.EXHAUST]:.4f}")

    _minor_header("SECTION y-y")
    print_row(
        "Primary pressure at plane y-y Ppy",
        f"{conditions.pressure[Phase.PRIMARY, Loc.CHOKE] * _PA_TO_MPA:.4f}",
        "MPa")
    print_row(
        "Primary temperature at plane y-y Tpy",
        f"{conditions.temperature[Phase.PRIMARY, Loc.CHOKE] - _K_TO_C:.2f}", "C")
    print_row(
        "Primary Mach number at plane y-y Mpy",
        f"{conditions.mach[Phase.PRIMARY, Loc.CHOKE]:.4f}")
    print_row(
        "Primary velocity at plane y-y Vpy",
        f"{conditions.velocity[Phase.PRIMARY, Loc.CHOKE]:.2f}", "m/s")
    print_row(
        "Suction pressure at plane y-y Psy",
        f"{conditions.pressure[Phase.SECONDARY, Loc.CHOKE] * _PA_TO_MPA:.4f}",
        "MPa")
    print_row(
        "Suction temperature at plane y-y Tsy",
        f"{conditions.temperature[Phase.SECONDARY, Loc.CHOKE] - _K_TO_C:.2f}",
        "C")
    print_row(
        "Suction velocity at plane y-y Vsy",
        f"{conditions.velocity[Phase.SECONDARY, Loc.CHOKE]:.2f}", "m/s")

    _minor_header("SECTION m")
    print_row(
        "Mixed-flow pressure before shock Pm",
        f"{conditions.pressure[Phase.MIX, Loc.PRE_SHOCK] * _PA_TO_MPA:.4f}",
        "MPa")
    print_row(
        "Mixed-flow temperature before shock Tm",
        f"{conditions.temperature[Phase.MIX, Loc.PRE_SHOCK] - _K_TO_C:.2f}", "C")
    print_row(
        "Mixed-flow Mach number before shock Mm",
        f"{conditions.mach[Phase.MIX, Loc.PRE_SHOCK]:.4f}")
    print_row(
        "Mixed-flow velocity before shock Vm",
        f"{conditions.velocity[Phase.MIX, Loc.PRE_SHOCK]:.2f}", "m/s")

    _minor_header("SECTION 3")
    print_row(
        "Pressure at constant-area section exit P3",
        f"{conditions.pressure[Phase.MIX, Loc.AFTERMIX] * _PA_TO_MPA:.4f}",
        "MPa")
    print_row(
        "Mach number at constant-area section exit M3",
        f"{conditions.mach[Phase.MIX, Loc.AFTERMIX]:.4f}")

    _minor_header("SECTION c")
    print_row(
        "Discharge pressure at ejector exit Pc",
        f"{conditions.pressure[Phase.MIX, Loc.DRAIN] * _PA_TO_MPA:.4f}",
        "MPa")

    print(_MAJOR_DIVIDER)
