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

    # NOTE большая функция, будет трудно рефакторить, следует разбить на подфункции
    # NOTE каждая подфункция будет решать свою задачу из блок-схемы в статье, потом
    # NOTE можно просто вызывать эти функции здесь

    gamma, R, cp = _extract_properties_for(
        req.phase,
        req.pressure[Phase.PRIMARY, Loc.INLET],
        req.temperature[Phase.PRIMARY, Loc.INLET],
    )

    # Eq. (1): mp
    req.mass_flow_rate[Phase.PRIMARY] = (
        req.pressure[Phase.PRIMARY, Loc.INLET] *
        design.area[Phase.PRIMARY, Loc.THROAT] /
        np.sqrt(req.temperature[Phase.PRIMARY, Loc.INLET]) *
        np.sqrt(gamma / R * (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (gamma - 1.0))) *
        np.sqrt(PRIMARY_NOZZLE_EFFICIENCY))

    # Eq. (2): Mp1
    req.mach[Phase.PRIMARY, Loc.EXHAUST] = fsolve(
        lambda x: (
            1.0 / x[0] ** 2 *
            (2.0 / (gamma + 1.0) * _isentropic_relation(gamma, x[0])) **
            ((gamma + 1.0) / (gamma - 1.0)) -
            (design.area[Phase.PRIMARY, Loc.EXHAUST] /
             design.area[Phase.PRIMARY, Loc.THROAT]) ** 2),
        [MACH_GUESS],
    )[0]
    # NOTE возможно будет легче вынести нелиненйное уравнение в отдельную функцию вместо
    # NOTE лямбды, будет полегче читать и проверять

    # Eq. (3): Pp1
    req.pressure[Phase.PRIMARY, Loc.EXHAUST] = (
        req.pressure[Phase.PRIMARY, Loc.INLET] /
        _isentropic_relation(gamma, req.mach[Phase.PRIMARY, Loc.EXHAUST]) **
        (gamma / (gamma - 1.0)))

    # Eq. (6): Msy
    req.mach[Phase.SECONDARY, Loc.CHOKE] = 1.0

    # Eq. (6): Psy
    req.pressure[Phase.SECONDARY, Loc.CHOKE] = (
        req.pressure[Phase.SECONDARY, Loc.INLET] /
        _isentropic_relation(gamma, req.mach[Phase.SECONDARY, Loc.CHOKE]) **
        (gamma / (gamma - 1.0)))
    # NOTE уравнения 6 и 3 одинаковые, хороший кандидат на отдельную функцию
    # NOTE может внутреннюю (с _ в начале)

    design.area[Phase.MIX, Loc.AFTERMIX] = (
        design.area[Phase.PRIMARY, Loc.THROAT] * _A3_INITIAL_RATIO)

    for _ in range(max_iter):
        _solve_entrainment_areas(req, design)
        _solve_mixing_to_drain(req, design)

        # Fig. 3, step 12 — Pc vs Pc*; подбор A3
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

    design.diameter[Phase.MIX, Loc.AFTERMIX] = np.sqrt(
        4.0 * design.area[Phase.MIX, Loc.AFTERMIX] / np.pi)

    for loc in (Loc.PREMIX, Loc.CHOKE, Loc.PRE_SHOCK, Loc.SHOCK):
        design.diameter[Phase.MIX, loc] = (
            design.diameter[Phase.MIX, Loc.AFTERMIX])

    req.mass_flow_rate[Phase.MIX] = (
        req.mass_flow_rate[Phase.PRIMARY] + req.mass_flow_rate[Phase.SECONDARY])


def _solve_entrainment_areas(
        req: Requirements,
        design: Design,
) -> None:
    """Fig. 3, steps 4–5 — Eqs. (4), (5), (8): Mpy, Apy, Asy."""

    gamma, _, _ = _extract_properties_for(
        req.phase,
        req.pressure[Phase.PRIMARY, Loc.INLET],
        req.temperature[Phase.PRIMARY, Loc.INLET],
    )

    while True:
        # Eq. (4): Ppy
        req.pressure[Phase.PRIMARY, Loc.CHOKE] = (
            req.pressure[Phase.SECONDARY, Loc.CHOKE])

        # Eq. (4): Mpy
        req.mach[Phase.PRIMARY, Loc.CHOKE] = fsolve(
            lambda x: (
                _isentropic_relation(
                    gamma, req.mach[Phase.PRIMARY, Loc.EXHAUST]) **
                (gamma / (gamma - 1.0)) /
                _isentropic_relation(gamma, x[0]) ** (gamma / (gamma - 1.0)) -
                req.pressure[Phase.SECONDARY, Loc.CHOKE] /
                req.pressure[Phase.PRIMARY, Loc.EXHAUST]
            ),
            [MACH_GUESS],
        )[0]

        # Eq. (5): Apy
        design.area[Phase.PRIMARY, Loc.CHOKE] = (
            design.area[Phase.PRIMARY, Loc.EXHAUST] *
            (PRIMARY_CORE_AREA_FACTOR / req.mach[Phase.PRIMARY, Loc.CHOKE] *
             (2.0 / (gamma + 1.0) *
              _isentropic_relation(
                  gamma, req.mach[Phase.PRIMARY, Loc.CHOKE])) **
             ((gamma + 1.0) / (2.0 * (gamma - 1.0)))) /
            (1.0 / req.mach[Phase.PRIMARY, Loc.EXHAUST] *
             (2.0 / (gamma + 1.0) *
              _isentropic_relation(
                  gamma, req.mach[Phase.PRIMARY, Loc.EXHAUST])) **
             ((gamma + 1.0) / (2.0 * (gamma - 1.0)))))

        # Eq. (8): Asy
        design.area[Phase.SECONDARY, Loc.CHOKE] = (
            design.area[Phase.MIX, Loc.AFTERMIX] -
            design.area[Phase.PRIMARY, Loc.CHOKE])

        if design.area[Phase.SECONDARY, Loc.CHOKE] < 0.0:
            # Fig. 3, step 5: Asy < 0 → A3 + ΔA3
            design.area[Phase.MIX, Loc.AFTERMIX] = (
                design.area[Phase.PRIMARY, Loc.CHOKE] + _DA3)
            continue
        # NOTE конструкция с continue и breake выглядит немного странно,
        # NOTE лучше вынести условие наверх в while
        # NOTE если пропустить строчку с continue, то кажется, что цикл выполняется
        # NOTE один раз и сразу заканчивается
        break


def _solve_mixing_to_drain(
        req: Requirements,
        design: Design,
) -> None:
    """Fig. 3, steps 6–11 — Eqs. (7)–(18): ms … Pc."""

    # NOTE тоже очень большая функция, я бы разбил на функции поменьше

    gamma, R, cp = _extract_properties_for(
        req.phase,
        req.pressure[Phase.PRIMARY, Loc.INLET],
        req.temperature[Phase.PRIMARY, Loc.INLET],
    )

    # Eq. (7): ms
    req.mass_flow_rate[Phase.SECONDARY] = (
        req.pressure[Phase.SECONDARY, Loc.INLET] *
        design.area[Phase.SECONDARY, Loc.CHOKE] /
        np.sqrt(req.temperature[Phase.SECONDARY, Loc.INLET]) *
        np.sqrt(gamma / R * (2.0 / (gamma + 1.0)) **
                ((gamma + 1.0) / (gamma - 1.0))) *
        np.sqrt(CARRY_NOZZLE_EFFICIENCY))
    # NOTE уравнения 1 и 7 одинаковые, хороший кандидат в отдельную функцию

    # Eq. (9): Tpy
    req.temperature[Phase.PRIMARY, Loc.CHOKE] = (
        req.temperature[Phase.PRIMARY, Loc.INLET] /
        _isentropic_relation(gamma, req.mach[Phase.PRIMARY, Loc.CHOKE]))

    # Eq. (10): Tsy
    req.temperature[Phase.SECONDARY, Loc.CHOKE] = (
        req.temperature[Phase.SECONDARY, Loc.INLET] /
        _isentropic_relation(gamma, req.mach[Phase.SECONDARY, Loc.CHOKE]))

    # Eq. (13): Vpy
    req.velocity[Phase.PRIMARY, Loc.CHOKE] = (
        req.mach[Phase.PRIMARY, Loc.CHOKE] *
        np.sqrt(gamma * R * req.temperature[Phase.PRIMARY, Loc.CHOKE]))

    # Eq. (14): Vsy
    req.velocity[Phase.SECONDARY, Loc.CHOKE] = (
        req.mach[Phase.SECONDARY, Loc.CHOKE] *
        np.sqrt(gamma * R * req.temperature[Phase.SECONDARY, Loc.CHOKE]))
    # NOTE уравнения 13 и 14 одинаковые, хороший кандидат в отдельную функцию

    # Eq. (11): Pm
    req.pressure[Phase.MIX, Loc.PRE_SHOCK] = (
        req.pressure[Phase.SECONDARY, Loc.CHOKE])

    # Eq. (19): fm
    fm = _mixing_coeff(
        design.area[Phase.MIX, Loc.AFTERMIX] /
        design.area[Phase.PRIMARY, Loc.THROAT])

    # Eq. (11): Vm
    req.velocity[Phase.MIX, Loc.PRE_SHOCK] = fm * (
        req.mass_flow_rate[Phase.PRIMARY] * req.velocity[Phase.PRIMARY, Loc.CHOKE] +
        req.mass_flow_rate[Phase.SECONDARY] *
        req.velocity[Phase.SECONDARY, Loc.CHOKE]
    ) / (req.mass_flow_rate[Phase.PRIMARY] + req.mass_flow_rate[Phase.SECONDARY])

    # Eq. (12): Tm
    req.temperature[Phase.MIX, Loc.PRE_SHOCK] = (
        req.mass_flow_rate[Phase.PRIMARY] * (
            cp * req.temperature[Phase.PRIMARY, Loc.CHOKE] +
            req.velocity[Phase.PRIMARY, Loc.CHOKE] ** 2 / 2.0) +
        req.mass_flow_rate[Phase.SECONDARY] * (
            cp * req.temperature[Phase.SECONDARY, Loc.CHOKE] +
            req.velocity[Phase.SECONDARY, Loc.CHOKE] ** 2 / 2.0) -
        (req.mass_flow_rate[Phase.PRIMARY] + req.mass_flow_rate[Phase.SECONDARY]) *
        req.velocity[Phase.MIX, Loc.PRE_SHOCK] ** 2 / 2.0) / (
            req.mass_flow_rate[Phase.PRIMARY] * cp +
            req.mass_flow_rate[Phase.SECONDARY] * cp)

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
        _isentropic_relation(gamma, req.mach[Phase.MIX, Loc.PRE_SHOCK]) /
        (gamma * req.mach[Phase.MIX, Loc.PRE_SHOCK] ** 2 -
         (gamma - 1.0) / 2.0))

    # Eq. (18): Pc
    req.pressure[Phase.MIX, Loc.DRAIN] = (
        req.pressure[Phase.MIX, Loc.AFTERMIX] *
        _isentropic_relation(gamma, req.mach[Phase.MIX, Loc.AFTERMIX]) **
        (gamma / (gamma - 1.0)))


def _isentropic_relation(gamma: float, mach: float) -> float:
    return 1.0 + (gamma - 1.0) / 2.0 * mach ** 2


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
