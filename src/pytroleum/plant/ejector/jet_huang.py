from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytroleum.tdyna.CoolStub import AbstractState  # type: ignore
else:
    from CoolProp import AbstractState
from CoolProp.constants import PT_INPUTS

import numpy as np
from scipy.optimize import fsolve
from enum import IntEnum


_R_UNIV = 8.314462618

# Huang et al. (1999), Eqs. (1), (5), (7); Section 4: np, ns, fp; Eq. (19): fm
PRIMARY_NOZZLE_EFFICIENCY = 0.95
CARRY_NOZZLE_EFFICIENCY = 0.85
PRIMARY_CORE_AREA_FACTOR = 0.88

MACH_GUESS = 2.0
_DA3 = 1e-8  # ΔA3, m² (Fig. 3)
_A3_INITIAL_RATIO = 7.0
_PC_REL_TOLERANCE = 1e-3
_MAX_ITER = 500


class Phase(IntEnum):
    """Stream indices (jet, carry, mixed)."""

    JET = 0
    CARRY = 1
    MIX = 2


class Loc(IntEnum):
    """Ejector section indices along the flow path (Huang et al., Fig. 2)."""

    INLET = 0
    THROAT = 1
    EXIT_NOZZLE = 2
    PREMIX = 3
    CHOKE = 4
    PRE_SHOCK = 5
    SHOCK = 6
    AFTERMIX = 7
    DRAIN = 8


_LAST_PHASE = len(Phase)
_LAST_LOC = len(Loc)
_SHAPE = (_LAST_PHASE, _LAST_LOC)
CONTAINER = np.full(_SHAPE, np.nan)

type Requirements = OperationConditions


@dataclass
class OperationConditions:

    phase: AbstractState
    mass_flow_rate: np.ndarray

    pressure: np.ndarray = field(default_factory=lambda: CONTAINER.copy())
    temperature: np.ndarray = field(default_factory=lambda: CONTAINER.copy())
    velocity: np.ndarray = field(default_factory=lambda: CONTAINER.copy())
    mach: np.ndarray = field(default_factory=lambda: CONTAINER.copy())

    def __post_init__(self) -> None:
        self.mass_flow_rate = np.array(
            [*self.mass_flow_rate, np.sum(self.mass_flow_rate)])

    def report(self) -> None:
        from jet_huang_report import report_conditions
        report_conditions(self)


@dataclass
class Design:
    """Ejector geometry from Huang et al. critical-mode analysis."""

    diameter: np.ndarray
    area: np.ndarray
    length: np.ndarray

    def report(self) -> None:
        from jet_huang_report import report_geometry
        report_geometry(self)


def solve_dimensions(
        req: Requirements,
        design: Design,
        Pc_star: float,
        Pc_rel_tolerance: float = _PC_REL_TOLERANCE,
        max_iter: int = _MAX_ITER,
) -> None:
    """Huang et al. (1999) critical-mode analysis, Eqs. (1)–(18), Fig. 3."""

    gamma, R, cp = _perfect_gas(
        req.phase,
        req.pressure[Phase.JET, Loc.INLET],
        req.temperature[Phase.JET, Loc.INLET],
    )

    # Eq. (1): mp
    req.mass_flow_rate[Phase.JET] = (
        req.pressure[Phase.JET, Loc.INLET] *
        design.area[Phase.JET, Loc.THROAT] /
        np.sqrt(req.temperature[Phase.JET, Loc.INLET]) *
        np.sqrt(gamma / R * (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (gamma - 1.0))) *
        np.sqrt(PRIMARY_NOZZLE_EFFICIENCY))

    # Eq. (2): Mp1
    req.mach[Phase.JET, Loc.EXIT_NOZZLE] = fsolve(
        lambda x: (
            1.0 / x[0] ** 2 *
            (2.0 / (gamma + 1.0) * (1.0 + (gamma - 1.0) / 2.0 * x[0] ** 2)) **
            ((gamma + 1.0) / (gamma - 1.0)) -
            (design.area[Phase.JET, Loc.EXIT_NOZZLE] /
             design.area[Phase.JET, Loc.THROAT]) ** 2),
        [MACH_GUESS],
    )[0]

    # Eq. (3): Pp1
    req.pressure[Phase.JET, Loc.EXIT_NOZZLE] = (
        req.pressure[Phase.JET, Loc.INLET] /
        (1.0 + (gamma - 1.0) / 2.0 * req.mach[Phase.JET, Loc.EXIT_NOZZLE] ** 2) **
        (gamma / (gamma - 1.0)))

    # Eq. (6): Msy
    req.mach[Phase.CARRY, Loc.CHOKE] = 1.0

    # Eq. (6): Psy
    req.pressure[Phase.CARRY, Loc.CHOKE] = (
        req.pressure[Phase.CARRY, Loc.INLET] /
        (1.0 + (gamma - 1.0) / 2.0 * req.mach[Phase.CARRY, Loc.CHOKE] ** 2) **
        (gamma / (gamma - 1.0)))

    design.area[Phase.MIX, Loc.AFTERMIX] = (
        design.area[Phase.JET, Loc.THROAT] * _A3_INITIAL_RATIO)

    for _ in range(max_iter):
        _solve_entrainment_areas(req, design)
        _solve_mixing_to_drain(req, design)

        # Fig. 3, step 12 — Pc vs Pc*; подбор A3
        if (abs(req.pressure[Phase.MIX, Loc.DRAIN] - Pc_star) / Pc_star <=
                Pc_rel_tolerance):
            break

        if req.pressure[Phase.MIX, Loc.DRAIN] >= Pc_star:
            design.area[Phase.MIX, Loc.AFTERMIX] += _DA3
        else:
            design.area[Phase.MIX, Loc.AFTERMIX] -= _DA3
    else:
        raise RuntimeError(
            f"solve_dimensions: Fig. 3 did not converge to Pc = Pc* "
            f"within {max_iter} iterations.")

    design.diameter[Phase.MIX, Loc.AFTERMIX] = np.sqrt(
        4.0 * design.area[Phase.MIX, Loc.AFTERMIX] / np.pi)

    for loc in (Loc.PREMIX, Loc.CHOKE, Loc.PRE_SHOCK, Loc.SHOCK):
        design.diameter[Phase.MIX, loc] = (
            design.diameter[Phase.MIX, Loc.AFTERMIX])

    req.mass_flow_rate[Phase.MIX] = (
        req.mass_flow_rate[Phase.JET] + req.mass_flow_rate[Phase.CARRY])


def _solve_entrainment_areas(
        req: Requirements,
        design: Design,
) -> None:
    """Fig. 3, steps 4–5 — Eqs. (4), (5), (8): Mpy, Apy, Asy."""

    gamma, _, _ = _perfect_gas(
        req.phase,
        req.pressure[Phase.JET, Loc.INLET],
        req.temperature[Phase.JET, Loc.INLET],
    )

    while True:
        # Eq. (4): Ppy
        req.pressure[Phase.JET, Loc.CHOKE] = (
            req.pressure[Phase.CARRY, Loc.CHOKE])

        # Eq. (4): Mpy
        req.mach[Phase.JET, Loc.CHOKE] = fsolve(
            lambda x: (
                (1.0 + (gamma - 1.0) / 2.0 *
                 req.mach[Phase.JET, Loc.EXIT_NOZZLE] ** 2) **
                (gamma / (gamma - 1.0)) /
                (1.0 + (gamma - 1.0) / 2.0 * x[0] ** 2) **
                (gamma / (gamma - 1.0)) -
                req.pressure[Phase.CARRY, Loc.CHOKE] /
                req.pressure[Phase.JET, Loc.EXIT_NOZZLE]
            ),
            [MACH_GUESS],
        )[0]

        # Eq. (5): Apy
        design.area[Phase.JET, Loc.CHOKE] = (
            design.area[Phase.JET, Loc.EXIT_NOZZLE] *
            (PRIMARY_CORE_AREA_FACTOR / req.mach[Phase.JET, Loc.CHOKE] *
             (2.0 / (gamma + 1.0) *
              (1.0 + (gamma - 1.0) / 2.0 *
               req.mach[Phase.JET, Loc.CHOKE] ** 2)) **
             ((gamma + 1.0) / (2.0 * (gamma - 1.0)))) /
            (1.0 / req.mach[Phase.JET, Loc.EXIT_NOZZLE] *
             (2.0 / (gamma + 1.0) *
              (1.0 + (gamma - 1.0) / 2.0 *
               req.mach[Phase.JET, Loc.EXIT_NOZZLE] ** 2)) **
             ((gamma + 1.0) / (2.0 * (gamma - 1.0)))))

        # Eq. (8): Asy
        design.area[Phase.CARRY, Loc.CHOKE] = (
            design.area[Phase.MIX, Loc.AFTERMIX] -
            design.area[Phase.JET, Loc.CHOKE])

        if design.area[Phase.CARRY, Loc.CHOKE] < 0.0:
            # Fig. 3, step 5: Asy < 0 → A3 + ΔA3
            design.area[Phase.MIX, Loc.AFTERMIX] = (
                design.area[Phase.JET, Loc.CHOKE] + _DA3)
            continue
        break


def _solve_mixing_to_drain(
        req: Requirements,
        design: Design,
) -> None:
    """Fig. 3, steps 6–11 — Eqs. (7)–(18): ms … Pc."""

    gamma, R, cp = _perfect_gas(
        req.phase,
        req.pressure[Phase.JET, Loc.INLET],
        req.temperature[Phase.JET, Loc.INLET],
    )

    # Eq. (7): ms
    req.mass_flow_rate[Phase.CARRY] = (
        req.pressure[Phase.CARRY, Loc.INLET] *
        design.area[Phase.CARRY, Loc.CHOKE] /
        np.sqrt(req.temperature[Phase.CARRY, Loc.INLET]) *
        np.sqrt(gamma / R * (2.0 / (gamma + 1.0)) **
                ((gamma + 1.0) / (gamma - 1.0))) *
        np.sqrt(CARRY_NOZZLE_EFFICIENCY))

    # Eq. (9): Tpy
    req.temperature[Phase.JET, Loc.CHOKE] = (
        req.temperature[Phase.JET, Loc.INLET] /
        (1.0 + (gamma - 1.0) / 2.0 * req.mach[Phase.JET, Loc.CHOKE] ** 2))

    # Eq. (10): Tsy
    req.temperature[Phase.CARRY, Loc.CHOKE] = (
        req.temperature[Phase.CARRY, Loc.INLET] /
        (1.0 + (gamma - 1.0) / 2.0 * req.mach[Phase.CARRY, Loc.CHOKE] ** 2))

    # Eq. (13): Vpy
    req.velocity[Phase.JET, Loc.CHOKE] = (
        req.mach[Phase.JET, Loc.CHOKE] *
        np.sqrt(gamma * R * req.temperature[Phase.JET, Loc.CHOKE]))

    # Eq. (13): Vsy
    req.velocity[Phase.CARRY, Loc.CHOKE] = (
        req.mach[Phase.CARRY, Loc.CHOKE] *
        np.sqrt(gamma * R * req.temperature[Phase.CARRY, Loc.CHOKE]))

    # Eq. (11): Pm
    req.pressure[Phase.MIX, Loc.PRE_SHOCK] = (
        req.pressure[Phase.CARRY, Loc.CHOKE])

    # Eq. (19): fm
    fm = _mixing_coeff(
        design.area[Phase.MIX, Loc.AFTERMIX] /
        design.area[Phase.JET, Loc.THROAT])

    # Eq. (11): Vm
    req.velocity[Phase.MIX, Loc.PRE_SHOCK] = fm * (
        req.mass_flow_rate[Phase.JET] * req.velocity[Phase.JET, Loc.CHOKE] +
        req.mass_flow_rate[Phase.CARRY] * req.velocity[Phase.CARRY, Loc.CHOKE]
    ) / (req.mass_flow_rate[Phase.JET] + req.mass_flow_rate[Phase.CARRY])

    # Eq. (12): Tm
    req.temperature[Phase.MIX, Loc.PRE_SHOCK] = (
        req.mass_flow_rate[Phase.JET] * (
            cp * req.temperature[Phase.JET, Loc.CHOKE] +
            req.velocity[Phase.JET, Loc.CHOKE] ** 2 / 2.0) +
        req.mass_flow_rate[Phase.CARRY] * (
            cp * req.temperature[Phase.CARRY, Loc.CHOKE] +
            req.velocity[Phase.CARRY, Loc.CHOKE] ** 2 / 2.0) -
        (req.mass_flow_rate[Phase.JET] + req.mass_flow_rate[Phase.CARRY]) *
        req.velocity[Phase.MIX, Loc.PRE_SHOCK] ** 2 / 2.0) / (
            req.mass_flow_rate[Phase.JET] * cp +
            req.mass_flow_rate[Phase.CARRY] * cp)

    # Eq. (15): Mm
    req.mach[Phase.MIX, Loc.PRE_SHOCK] = (
        req.velocity[Phase.MIX, Loc.PRE_SHOCK] /
        np.sqrt(gamma * R * req.temperature[Phase.MIX, Loc.PRE_SHOCK]))

    # Eq. (16): P3
    req.pressure[Phase.MIX, Loc.AFTERMIX] = (
        req.pressure[Phase.MIX, Loc.PRE_SHOCK] *
        (1.0 + 2.0 * gamma / (gamma + 1.0) *
         (req.mach[Phase.MIX, Loc.PRE_SHOCK] ** 2 - 1.0)))

    # Eq. (17): M3
    req.mach[Phase.MIX, Loc.AFTERMIX] = np.sqrt(
        (1.0 + (gamma - 1.0) / 2.0 * req.mach[Phase.MIX, Loc.PRE_SHOCK] ** 2) /
        (gamma * req.mach[Phase.MIX, Loc.PRE_SHOCK] ** 2 -
         (gamma - 1.0) / 2.0))

    # Eq. (18): Pc
    req.pressure[Phase.MIX, Loc.DRAIN] = (
        req.pressure[Phase.MIX, Loc.AFTERMIX] *
        (1.0 + (gamma - 1.0) / 2.0 *
         req.mach[Phase.MIX, Loc.AFTERMIX] ** 2) **
        (gamma / (gamma - 1.0)))


def _perfect_gas(
        eos: AbstractState, pressure: float, temperature: float,
) -> tuple[float, float, float]:
    """Constant gamma, R, cp at inlet state (Huang et al.)."""

    eos.update(PT_INPUTS, pressure, temperature)
    cp = eos.cpmass()
    gamma = cp / eos.cvmass()
    R = _R_UNIV / eos.molar_mass()
    return gamma, R, cp


def _mixing_coeff(area_ratio: float) -> float:
    """Return fm for A3/At (Huang et al., Eq. 19)."""

    if area_ratio > 8.3:
        return 0.80
    if area_ratio > 6.9:
        return 0.82
    return 0.84
