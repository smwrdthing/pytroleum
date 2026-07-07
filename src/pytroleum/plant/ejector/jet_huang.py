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
PRIMARY_NOZZLE_EFF = 0.95
SECONDARY_NOZZLE_EFF = 0.85
PRIMARY_CORE_AREA_FACTOR = 0.88

MACH_GUESS = 2.0
_DA3 = 1e-8  # ΔA3, m² (Fig. 3)
_A3_INITIAL_RATIO = 7.0
_PC_REL_TOLERANCE = 1e-3
_MAX_ITER = 500


class Phase(IntEnum):
    """Stream indices (jet, carry, mixed)."""

    PRIMARY = 0
    SECONDARY = 1
    MIX = 2


class Loc(IntEnum):
    """Ejector section indices along the flow path (Huang et al., Fig. 2)."""

    INLET = 0
    THROAT = 1
    EXHAUST = 2
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

    gamma, R, cp = _extract_properties_for(
        req.phase,
        req.pressure[Phase.PRIMARY, Loc.INLET],
        req.temperature[Phase.PRIMARY, Loc.INLET],
    )

    _primary_mass_flow(req, design, gamma, R)
    _primary_exhaust_state(req, design, gamma)
    _secondary_choke_state(req, gamma)

    design.area[Phase.MIX, Loc.AFTERMIX] = (
        design.area[Phase.PRIMARY, Loc.THROAT] * _A3_INITIAL_RATIO)

    for _ in range(max_iter):
        # Fig. 3: Asy < 0 → A3 = Apy + ΔA3 → Eq. (4)
        while True:
            _primary_core_state(req, design, gamma)
            _secondary_choke_area(design)
            if design.area[Phase.SECONDARY, Loc.CHOKE] >= 0.0:
                break
            design.area[Phase.MIX, Loc.AFTERMIX] = (
                design.area[Phase.PRIMARY, Loc.CHOKE] + _DA3)

        _secondary_mass_flow(req, design, gamma, R)
        _choke_temperatures(req, gamma)
        _mix_pre_shock_velocity_pressure(req, design, gamma, R)
        _mix_pre_shock_temperature_mach(req, gamma, R, cp)
        _aftermix_state(req, gamma)
        _mix_drain_pressure(req, gamma)

        # Fig. 3: Pc vs Pc* → подбор A3 → Eq. (4)
        if (abs(req.pressure[Phase.MIX, Loc.DRAIN] - Pc_star) / Pc_star <=
                Pc_rel_tolerance):
            # NOTE вместо for - цикла с breake здесь можно сделать while-цикл
            # NOTE с похожим условием, должно получиться чуть покороче
            break

        if req.pressure[Phase.MIX, Loc.DRAIN] >= Pc_star:
            design.area[Phase.MIX, Loc.AFTERMIX] += _DA3
        else:
            design.area[Phase.MIX, Loc.AFTERMIX] -= _DA3
    else:
        raise RuntimeError(
            f"solve_dimensions: Fig. 3 did not converge to Pc = Pc* "
            f"within {max_iter} iterations.")

    _finalize_mix_geometry(req, design)


# --- Fig. 3 (Huang et al., Eqs. 1–18) ---


def _primary_mass_flow(
        req: Requirements,
        design: Design,
        gamma: float,
        R: float,
) -> None:
    """Fig. 3 — Eq. (1): mp."""

    req.mass_flow_rate[Phase.PRIMARY] = _mass_flow_rate(
        req.pressure[Phase.PRIMARY, Loc.INLET],
        req.temperature[Phase.PRIMARY, Loc.INLET],
        design.area[Phase.PRIMARY, Loc.THROAT],
        gamma, R, PRIMARY_NOZZLE_EFF)


def _primary_exhaust_state(
        req: Requirements,
        design: Design,
        gamma: float,
) -> None:
    """Fig. 3 — Eqs. (2), (3): Mp1, Pp1."""

    req.mach[Phase.PRIMARY, Loc.EXHAUST] = fsolve(
        _primary_exhaust_mach_residual,
        [MACH_GUESS],
        args=(
            gamma,
            design.area[Phase.PRIMARY, Loc.EXHAUST] /
            design.area[Phase.PRIMARY, Loc.THROAT]),
    )[0]
    req.pressure[Phase.PRIMARY, Loc.EXHAUST] = (
        req.pressure[Phase.PRIMARY, Loc.INLET] /
        _isentropic_relation(gamma, req.mach[Phase.PRIMARY, Loc.EXHAUST]) **
        (gamma / (gamma - 1.0)))


def _secondary_choke_state(
        req: Requirements,
        gamma: float,
) -> None:
    """Fig. 3 — Eq. (6): Msy, Psy."""

    req.mach[Phase.SECONDARY, Loc.CHOKE] = 1.0
    req.pressure[Phase.SECONDARY, Loc.CHOKE] = (
        req.pressure[Phase.SECONDARY, Loc.INLET] /
        _isentropic_relation(gamma, req.mach[Phase.SECONDARY, Loc.CHOKE]) **
        (gamma / (gamma - 1.0)))


def _primary_core_state(
        req: Requirements,
        design: Design,
        gamma: float,
) -> None:
    """Fig. 3 — Eqs. (4), (5): Mpy, Apy."""

    req.pressure[Phase.PRIMARY, Loc.CHOKE] = (
        req.pressure[Phase.SECONDARY, Loc.CHOKE])
    req.mach[Phase.PRIMARY, Loc.CHOKE] = fsolve(
        _primary_choke_mach_residual,
        [MACH_GUESS],
        args=(
            gamma,
            req.mach[Phase.PRIMARY, Loc.EXHAUST],
            req.pressure[Phase.SECONDARY, Loc.CHOKE] /
            req.pressure[Phase.PRIMARY, Loc.EXHAUST]),
    )[0]
    design.area[Phase.PRIMARY, Loc.CHOKE] = (
        design.area[Phase.PRIMARY, Loc.EXHAUST] *
        (PRIMARY_CORE_AREA_FACTOR / req.mach[Phase.PRIMARY, Loc.CHOKE] *
         (2.0 / (gamma + 1.0) *
          _isentropic_relation(gamma, req.mach[Phase.PRIMARY, Loc.CHOKE])) **
         ((gamma + 1.0) / (2.0 * (gamma - 1.0)))) /
        (1.0 / req.mach[Phase.PRIMARY, Loc.EXHAUST] *
         (2.0 / (gamma + 1.0) *
          _isentropic_relation(gamma, req.mach[Phase.PRIMARY, Loc.EXHAUST])) **
         ((gamma + 1.0) / (2.0 * (gamma - 1.0)))))


def _secondary_choke_area(
        design: Design,
) -> None:
    """Fig. 3 — Eq. (8): Asy."""

    design.area[Phase.SECONDARY, Loc.CHOKE] = (
        design.area[Phase.MIX, Loc.AFTERMIX] -
        design.area[Phase.PRIMARY, Loc.CHOKE])


def _secondary_mass_flow(
        req: Requirements,
        design: Design,
        gamma: float,
        R: float,
) -> None:
    """Fig. 3 — Eq. (7): ms."""

    req.mass_flow_rate[Phase.SECONDARY] = _mass_flow_rate(
        req.pressure[Phase.SECONDARY, Loc.INLET],
        req.temperature[Phase.SECONDARY, Loc.INLET],
        design.area[Phase.SECONDARY, Loc.CHOKE],
        gamma, R, SECONDARY_NOZZLE_EFF)


def _choke_temperatures(
        req: Requirements,
        gamma: float,
) -> None:
    """Fig. 3 — Eqs. (9), (10): Tpy, Tsy."""

    req.temperature[Phase.PRIMARY, Loc.CHOKE] = (
        req.temperature[Phase.PRIMARY, Loc.INLET] /
        _isentropic_relation(gamma, req.mach[Phase.PRIMARY, Loc.CHOKE]))
    req.temperature[Phase.SECONDARY, Loc.CHOKE] = (
        req.temperature[Phase.SECONDARY, Loc.INLET] /
        _isentropic_relation(gamma, req.mach[Phase.SECONDARY, Loc.CHOKE]))


def _mix_pre_shock_velocity_pressure(
        req: Requirements,
        design: Design,
        gamma: float,
        R: float,
) -> None:
    """Fig. 3 — Eqs. (11), (13), (14), (19): Pm, Vpy, Vsy, Vm."""

    req.velocity[Phase.PRIMARY, Loc.CHOKE] = _velocity_from_mach(
        req.mach[Phase.PRIMARY, Loc.CHOKE], gamma, R,
        req.temperature[Phase.PRIMARY, Loc.CHOKE])
    req.velocity[Phase.SECONDARY, Loc.CHOKE] = _velocity_from_mach(
        req.mach[Phase.SECONDARY, Loc.CHOKE], gamma, R,
        req.temperature[Phase.SECONDARY, Loc.CHOKE])
    req.pressure[Phase.MIX, Loc.PRE_SHOCK] = (
        req.pressure[Phase.SECONDARY, Loc.CHOKE])
    fm = _mixing_coeff(
        design.area[Phase.MIX, Loc.AFTERMIX] /
        design.area[Phase.PRIMARY, Loc.THROAT])
    req.velocity[Phase.MIX, Loc.PRE_SHOCK] = fm * (
        req.mass_flow_rate[Phase.PRIMARY] * req.velocity[Phase.PRIMARY, Loc.CHOKE] +
        req.mass_flow_rate[Phase.SECONDARY] *
        req.velocity[Phase.SECONDARY, Loc.CHOKE]
    ) / (req.mass_flow_rate[Phase.PRIMARY] + req.mass_flow_rate[Phase.SECONDARY])


def _mix_pre_shock_temperature_mach(
        req: Requirements,
        gamma: float,
        R: float,
        cp: float,
) -> None:
    """Fig. 3 — Eqs. (12), (15): Tm, Mm."""

    primary_choke_energy = (
        cp * req.temperature[Phase.PRIMARY, Loc.CHOKE] +
        req.velocity[Phase.PRIMARY, Loc.CHOKE] ** 2 / 2.0)
    secondary_choke_energy = (
        cp * req.temperature[Phase.SECONDARY, Loc.CHOKE] +
        req.velocity[Phase.SECONDARY, Loc.CHOKE] ** 2 / 2.0)

    req.temperature[Phase.MIX, Loc.PRE_SHOCK] = 1/cp*(
        (req.mass_flow_rate[Phase.PRIMARY] * primary_choke_energy +
         req.mass_flow_rate[Phase.SECONDARY] * secondary_choke_energy) /
        (req.mass_flow_rate[Phase.PRIMARY] +
         req.mass_flow_rate[Phase.SECONDARY]) -
        req.velocity[Phase.MIX, Loc.PRE_SHOCK] ** 2 / 2.0)

    req.mach[Phase.MIX, Loc.PRE_SHOCK] = (
        req.velocity[Phase.MIX, Loc.PRE_SHOCK] /
        np.sqrt(gamma * R * req.temperature[Phase.MIX, Loc.PRE_SHOCK]))


def _aftermix_state(
        req: Requirements,
        gamma: float,
) -> None:
    """Fig. 3 — Eqs. (16), (17): P3, M3."""

    req.pressure[Phase.MIX, Loc.AFTERMIX] = (
        req.pressure[Phase.MIX, Loc.PRE_SHOCK] *
        (1.0 + 2.0 * gamma / (gamma + 1.0) *
         (req.mach[Phase.MIX, Loc.PRE_SHOCK] ** 2 - 1.0)))
    req.mach[Phase.MIX, Loc.AFTERMIX] = np.sqrt(
        _isentropic_relation(gamma, req.mach[Phase.MIX, Loc.PRE_SHOCK]) /
        (gamma * req.mach[Phase.MIX, Loc.PRE_SHOCK] ** 2 -
         (gamma - 1.0) / 2.0))


def _mix_drain_pressure(
        req: Requirements,
        gamma: float,
) -> None:
    """Fig. 3 — Eq. (18): Pc."""

    req.pressure[Phase.MIX, Loc.DRAIN] = (
        req.pressure[Phase.MIX, Loc.AFTERMIX] *
        _isentropic_relation(gamma, req.mach[Phase.MIX, Loc.AFTERMIX]) **
        (gamma / (gamma - 1.0)))


def _finalize_mix_geometry(
        req: Requirements,
        design: Design,
) -> None:
    """Set mix-section diameters"""

    design.diameter[Phase.MIX, Loc.AFTERMIX] = np.sqrt(
        4.0 * design.area[Phase.MIX, Loc.AFTERMIX] / np.pi)

    for loc in (Loc.PREMIX, Loc.CHOKE, Loc.PRE_SHOCK, Loc.SHOCK):
        design.diameter[Phase.MIX, loc] = (
            design.diameter[Phase.MIX, Loc.AFTERMIX])


# --- Helpers ---


def _velocity_from_mach(
        mach: float,
        gamma: float,
        R: float,
        temperature: float,
) -> float:
    """Return V = M · sqrt(γRT) (Huang et al., Eqs. 13, 14)."""

    return mach * np.sqrt(gamma * R * temperature)


def _mass_flow_rate(
        pressure: float,
        temperature: float,
        area: float,
        gamma: float,
        R: float,
        nozzle_efficiency: float,
) -> float:
    """Mass flow at choking condition (Huang et al., Eqs. 1, 7)."""

    return (
        pressure * area / np.sqrt(temperature) *
        np.sqrt(gamma / R * (2.0 / (gamma + 1.0)) **
                ((gamma + 1.0) / (gamma - 1.0))) *
        np.sqrt(nozzle_efficiency))


def _isentropic_relation(
        gamma: float, mach: float | np.ndarray,
) -> float | np.ndarray:
    """Return 1 + (γ − 1)/2 · M²."""

    return 1.0 + (gamma - 1.0) / 2.0 * mach ** 2


def _primary_exhaust_mach_residual(
        mach: np.ndarray,
        gamma: float,
        area_ratio: float,
) -> float | np.ndarray:
    """Eq. (2) residual: (A/A*)² − f(M) for primary nozzle exit Mach."""

    return (
        1.0 / mach ** 2 *
        (2.0 / (gamma + 1.0) * _isentropic_relation(gamma, mach)) **
        ((gamma + 1.0) / (gamma - 1.0)) -
        area_ratio ** 2)


def _primary_choke_mach_residual(
        mach: np.ndarray,
        gamma: float,
        mach_exhaust: float,
        pressure_ratio: float,
) -> float | np.ndarray:
    """Eq. (4) residual: isentropic pressure ratio match at primary choke."""

    return (
        _isentropic_relation(gamma, mach_exhaust) **
        (gamma / (gamma - 1.0)) /
        _isentropic_relation(gamma, mach) ** (gamma / (gamma - 1.0)) -
        pressure_ratio)


def _extract_properties_for(
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
