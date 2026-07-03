from jet_huang import Loc, Phase

_M_TO_MM = 1e3
_M2_TO_CM2 = 1e4
_PA_TO_MPA = 1e-6
_K_TO_C = 273.15

_SECTIONS = (
    ("Inlet", Loc.INLET),
    ("Throat", Loc.THROAT),
    ("Nozzle exit", Loc.EXIT_NOZZLE),
    ("Premix", Loc.PREMIX),
    ("Choke", Loc.CHOKE),
    ("Pre-shock", Loc.PRE_SHOCK),
    ("Shock", Loc.SHOCK),
    ("Aftermix", Loc.AFTERMIX),
    ("Drain", Loc.DRAIN),
)


def _report_field(arr, title, unit, transform, fmt) -> None:
    print()
    print(title)
    suffix = f" [{unit}]" if unit else ""
    for label, loc in _SECTIONS:
        print(
            f"{label:<12}: {transform(arr[Phase.JET, loc]):{fmt}}"
            f" / {transform(arr[Phase.CARRY, loc]):{fmt}}"
            f" / {transform(arr[Phase.MIX, loc]):{fmt}}{suffix}")


def report_conditions(conditions) -> None:
    print("         jet / carry / mix")
    print("Mass flow rates")
    print(
        f"         : {conditions.mass_flow_rate[Phase.JET]:.3e}"
        f" / {conditions.mass_flow_rate[Phase.CARRY]:.3e}"
        f" / {conditions.mass_flow_rate[Phase.MIX]:.3e} [kg/s]")

    _report_field(
        conditions.temperature, "temperatures", "C",
        lambda value: value - _K_TO_C, ".2f")
    _report_field(
        conditions.pressure, "pressure", "MPa",
        lambda value: value * _PA_TO_MPA, ".4f")
    _report_field(
        conditions.velocity, "velocity", "m/s",
        lambda value: value, ".2f")
    _report_field(
        conditions.mach, "Mach numbers", "",
        lambda value: value, ".4f")


def report_geometry(design) -> None:
    print("         jet / carry / mix")
    _report_field(
        design.diameter, "diameters", "mm",
        lambda value: value * _M_TO_MM, ".2f")
    _report_field(
        design.area, "areas", "cm2",
        lambda value: value * _M2_TO_CM2, ".2f")
    _report_field(
        design.length, "lengths", "mm",
        lambda value: value * _M_TO_MM, ".2f")
