from __future__ import annotations
from report import report_conditions, report_dimensions, report_inputs

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytroleum.tdyna.CoolStub import AbstractState  # type: ignore
else:
    from CoolProp import AbstractState
from CoolProp.constants import PT_INPUTS

import numpy as np
from scipy.optimize import fsolve

from interfaces import Loc, Phase, _CONTAINER

_R_UNIV = 8.314462618

# Huang et al. (1999), Int. J. Refrigeration — 1-D critical-mode ejector model.
PRIMARY_NOZZLE_EFF = 0.95
SECONDARY_NOZZLE_EFF = 0.85
PRIMARY_CORE_AREA_FACTOR = 0.88

MACH_GUESS = 2.0
_DA3 = 1e-6  # ΔA3, m² (Fig. 3)
_A3_INITIAL_RATIO = 7.0
_PC_REL_TOLERANCE = 1e-3
_MAX_ITER = 1000


@dataclass
class Requirements:
    """Boundary conditions from the technical specification (ТЗ)."""

    phase: AbstractState
    Pc_star: float
    pressure: np.ndarray = field(default_factory=lambda: _CONTAINER.copy())
    temperature: np.ndarray = field(default_factory=lambda: _CONTAINER.copy())

    def report(self) -> None:
        """Print boundary conditions from the technical specification."""
        report_inputs(self)


@dataclass
class OperationConditions:

    phase: AbstractState
    gamma: float = field(init=False)
    R: float = field(init=False)
    cp: float = field(init=False)
    mass_flow_rate: np.ndarray = field(
        default_factory=lambda: np.zeros(Phase.M))

    pressure: np.ndarray = field(default_factory=lambda: _CONTAINER.copy())
    temperature: np.ndarray = field(default_factory=lambda: _CONTAINER.copy())
    velocity: np.ndarray = field(default_factory=lambda: _CONTAINER.copy())
    mach: np.ndarray = field(default_factory=lambda: _CONTAINER.copy())

    def __post_init__(self) -> None:
        self.mass_flow_rate = np.array(
            [*self.mass_flow_rate, np.sum(self.mass_flow_rate)])

    def read_requirements(self, req: Requirements) -> None:
        """Load boundary conditions from req and set Cp, γ, and R."""
        self.phase = req.phase
        # NOTE надёжнее будет сконструировать новые объекты для уравнений состояний,
        # NOTE здесь может оказаться так, что self.phase и req.phase - одно и то же,
        # NOTE тогда при изменении self.phase будет меняться и req.phase, при копировании
        # NOTE объектов из полей классов такую связь нужно исключить

        self.pressure = req.pressure.copy()
        self.temperature = req.temperature.copy()
        self._extract_properties_for()

    def _extract_properties_for(self) -> None:
        """Set constant Cp, γ, and R at the nozzle inlet state."""
        self.phase.update(
            PT_INPUTS,
            self.pressure[Phase.P, Loc.IN],
            self.temperature[Phase.P, Loc.IN],
        )
        self.cp = self.phase.cpmass()
        self.gamma = self.cp / self.phase.cvmass()
        self.R = _R_UNIV / self.phase.molar_mass()

    def _secondary_choke_state(self) -> None:
        """Set secondary-stream Mach number and pressure at the choke plane."""

        self.mach[Phase.S, Loc.CH] = 1.0
        self.pressure[Phase.S, Loc.CH] = (
            self.pressure[Phase.S, Loc.IN] /
            _isentropic_relation(self.gamma, self.mach[Phase.S, Loc.CH]) **
            (self.gamma / (self.gamma - 1.0)))

    def _choke_temperatures(self) -> None:
        """Set primary and secondary stream temperatures at the choke plane."""

        self.temperature[Phase.P, Loc.CH] = (
            self.temperature[Phase.P, Loc.IN] /
            _isentropic_relation(self.gamma, self.mach[Phase.P, Loc.CH]))
        self.temperature[Phase.S, Loc.CH] = (
            self.temperature[Phase.S, Loc.IN] /
            _isentropic_relation(self.gamma, self.mach[Phase.S, Loc.CH]))

    def _mix_pre_shock_temperature_mach(self) -> None:
        """Set mixed-stream temperature and Mach number before the shock."""

        primary_choke_energy = (
            self.cp * self.temperature[Phase.P, Loc.CH] +
            self.velocity[Phase.P, Loc.CH] ** 2 / 2.0)
        secondary_choke_energy = (
            self.cp * self.temperature[Phase.S, Loc.CH] +
            self.velocity[Phase.S, Loc.CH] ** 2 / 2.0)

        self.temperature[Phase.M, Loc.PS] = 1/self.cp*(
            (self.mass_flow_rate[Phase.P] * primary_choke_energy +
             self.mass_flow_rate[Phase.S] * secondary_choke_energy) /
            (self.mass_flow_rate[Phase.P] +
             self.mass_flow_rate[Phase.S]) -
            self.velocity[Phase.M, Loc.PS] ** 2 / 2.0)

        self.mach[Phase.M, Loc.PS] = (
            self.velocity[Phase.M, Loc.PS] /
            np.sqrt(self.gamma * self.R *
                    self.temperature[Phase.M, Loc.PS]))

    def _aftermix_state(self) -> None:
        """Set mixed-stream pressure and Mach number after the shock."""

        self.pressure[Phase.M, Loc.AM] = (
            self.pressure[Phase.M, Loc.PS] *
            (1.0 + 2.0 * self.gamma / (self.gamma + 1.0) *
             (self.mach[Phase.M, Loc.PS] ** 2 - 1.0)))
        self.mach[Phase.M, Loc.AM] = np.sqrt(
            _isentropic_relation(self.gamma, self.mach[Phase.M, Loc.PS]) /
            (self.gamma * self.mach[Phase.M, Loc.PS] ** 2 -
             (self.gamma - 1.0) / 2.0))

    def _mix_drain_pressure(self) -> None:
        """Set discharge pressure at the ejector exit."""

        self.pressure[Phase.M, Loc.D] = (
            self.pressure[Phase.M, Loc.AM] *
            _isentropic_relation(self.gamma, self.mach[Phase.M, Loc.AM]) **
            (self.gamma / (self.gamma - 1.0)))

    def converged(self, Pc_star: float, rel_tolerance: float) -> bool:
        """Return True if discharge pressure matches the target within tolerance."""
        Pc = self.pressure[Phase.M, Loc.D]
        return abs(Pc - Pc_star) / Pc_star <= rel_tolerance

    def report(self) -> None:
        """Print calculated flow states and performance parameters."""
        report_conditions(self)


@dataclass
class Design:
    """Ejector flow-passage geometry (areas, diameters, lengths)."""

    diameter: np.ndarray
    area: np.ndarray
    length: np.ndarray

    def _secondary_choke_area(self) -> None:
        """Set secondary choke area from mixed and primary core areas."""

        self.area[Phase.S, Loc.CH] = (
            self.area[Phase.M, Loc.AM] -
            self.area[Phase.P, Loc.CH])

    def _finalize_mix_geometry(self) -> None:
        """Set mix-section diameters."""

        self.diameter[Phase.M, Loc.AM] = np.sqrt(
            4.0 * self.area[Phase.M, Loc.AM] / np.pi)

        for loc in (Loc.PM, Loc.CH, Loc.PS, Loc.SH):
            self.diameter[Phase.M, loc] = (
                self.diameter[Phase.M, Loc.AM])

    def _area_ratio(self, loc_num: Loc, loc_denom: Loc) -> float:
        return self.area[Phase.P, loc_num] / self.area[Phase.P, loc_denom]

    def report(self) -> None:
        """Print ejector geometry (areas and diameters, lenghts)."""
        report_dimensions(self)


def solve_dimensions(
        req: Requirements,

        design: Design,
        # NOTE design лучше создавать внутри функции, которая выполняет проектный расчёт
        # NOTE и возвращать вместе с OperationConditions (см. заметки в __init__.py)

        Pc_rel_tolerance: float = _PC_REL_TOLERANCE,
        max_iter: int = _MAX_ITER,
) -> OperationConditions:  # NOTE сигнатура вызова и имя функции противоречат друг-другу
    """Run a jet ejector at critical mode"""

    conditions = OperationConditions(phase=req.phase)
    conditions.read_requirements(req)

    _primary_mass_flow(conditions, design)
    _primary_exhaust_state(conditions, design)
    conditions._secondary_choke_state()

    design.area[Phase.M, Loc.AM] = (
        design.area[Phase.P, Loc.TH] * _A3_INITIAL_RATIO)

    iter_count = 0
    while not conditions.converged(req.Pc_star, Pc_rel_tolerance):
        # Asy < 0 → A3 = Apy + ΔA3 см. Huang et al. (1999)
        valid_secondary_area = False
        while not valid_secondary_area:
            _primary_core_state(conditions, design)
            design._secondary_choke_area()
            valid_secondary_area = design.area[Phase.S, Loc.CH] >= 0.0

            if not valid_secondary_area:
                design.area[Phase.M, Loc.AM] = design.area[Phase.P, Loc.CH]
                design.area[Phase.M, Loc.AM] += _DA3

        _secondary_mass_flow(conditions, design)
        conditions._choke_temperatures()
        _mix_pre_shock_velocity_pressure(conditions, design)
        conditions._mix_pre_shock_temperature_mach()
        conditions._aftermix_state()
        conditions._mix_drain_pressure()

        if iter_count >= max_iter:
            raise RuntimeError(
                f"Solution algorithm did not converge in {max_iter} iterations, "
                f"outer loop is abandoned.")

        if not conditions.converged(req.Pc_star, Pc_rel_tolerance):
            # Pc vs Pc* → подбор A3 см. Huang et al. (1999)
            Pc = conditions.pressure[Phase.M, Loc.D]

            if Pc >= req.Pc_star:
                design.area[Phase.M, Loc.AM] += _DA3
            else:
                design.area[Phase.M, Loc.AM] -= _DA3

            iter_count += 1

    design._finalize_mix_geometry()
    return conditions


def _primary_mass_flow(
        conditions: OperationConditions,
        design: Design,
) -> None:
    """Compute primary mass flow rate"""

    conditions.mass_flow_rate[Phase.P] = _mass_flow_rate(
        conditions.pressure[Phase.P, Loc.IN],
        conditions.temperature[Phase.P, Loc.IN],
        design.area[Phase.P, Loc.TH],
        conditions.gamma, conditions.R, PRIMARY_NOZZLE_EFF)


def _primary_exhaust_state(
        conditions: OperationConditions,
        design: Design,
) -> None:
    """Compute primary nozzle exit Mach number and pressure."""

    conditions.mach[Phase.P, Loc.EX] = fsolve(
        _primary_exhaust_mach_residual,
        [MACH_GUESS],
        args=(
            conditions.gamma,
            design._area_ratio(Loc.EX, Loc.TH)),
    )[0]

    conditions.pressure[Phase.P, Loc.EX] = (
        conditions.pressure[Phase.P, Loc.IN] /
        _isentropic_relation(conditions.gamma, conditions.mach[Phase.P, Loc.EX]) **
        (conditions.gamma / (conditions.gamma - 1.0)))


def _primary_core_state(
        conditions: OperationConditions,
        design: Design,
) -> None:
    """Compute primary choke Mach number and core flow area."""

    conditions.pressure[Phase.P, Loc.CH] = (
        conditions.pressure[Phase.S, Loc.CH])
    conditions.mach[Phase.P, Loc.CH] = fsolve(
        _primary_choke_mach_residual,
        [MACH_GUESS],
        args=(
            conditions.gamma,
            conditions.mach[Phase.P, Loc.EX],
            conditions.pressure[Phase.S, Loc.CH] /
            conditions.pressure[Phase.P, Loc.EX]),
    )[0]
    design.area[Phase.P, Loc.CH] = (
        design.area[Phase.P, Loc.EX] *
        (PRIMARY_CORE_AREA_FACTOR / conditions.mach[Phase.P, Loc.CH] *
         (2.0 / (conditions.gamma + 1.0) *
          _isentropic_relation(conditions.gamma, conditions.mach[Phase.P, Loc.CH])) **
         ((conditions.gamma + 1.0) / (2.0 * (conditions.gamma - 1.0)))) /
        (1.0 / conditions.mach[Phase.P, Loc.EX] *
         (2.0 / (conditions.gamma + 1.0) *
          _isentropic_relation(conditions.gamma, conditions.mach[Phase.P, Loc.EX])) **
         ((conditions.gamma + 1.0) / (2.0 * (conditions.gamma - 1.0)))))


def _secondary_mass_flow(
        conditions: OperationConditions,
        design: Design,
) -> None:
    """Compute secondary mass flow rate"""

    conditions.mass_flow_rate[Phase.S] = _mass_flow_rate(
        conditions.pressure[Phase.S, Loc.IN],
        conditions.temperature[Phase.S, Loc.IN],
        design.area[Phase.S, Loc.CH],
        conditions.gamma, conditions.R, SECONDARY_NOZZLE_EFF)


def _mix_pre_shock_velocity_pressure(
        conditions: OperationConditions,
        design: Design,
) -> None:
    """Compute mixed-stream state before the shock."""

    conditions.velocity[Phase.P, Loc.CH] = _velocity_from_mach(
        conditions.mach[Phase.P, Loc.CH], conditions.gamma, conditions.R,
        conditions.temperature[Phase.P, Loc.CH])

    conditions.velocity[Phase.S, Loc.CH] = _velocity_from_mach(
        conditions.mach[Phase.S, Loc.CH], conditions.gamma, conditions.R,
        conditions.temperature[Phase.S, Loc.CH])

    conditions.pressure[Phase.M, Loc.PS] = (
        conditions.pressure[Phase.S, Loc.CH])

    fm = _mixing_coeff(
        design.area[Phase.M, Loc.AM] /
        design.area[Phase.P, Loc.TH])

    conditions.velocity[Phase.M, Loc.PS] = fm * (
        conditions.mass_flow_rate[Phase.P] *
        conditions.velocity[Phase.P, Loc.CH] +
        conditions.mass_flow_rate[Phase.S] *
        conditions.velocity[Phase.S, Loc.CH]
    ) / (conditions.mass_flow_rate[Phase.P] +
         conditions.mass_flow_rate[Phase.S])


# --- Helpers ---


def _velocity_from_mach(
        mach: float,
        gamma: float,
        R: float,
        temperature: float,
) -> float:
    """Return flow velocity from Mach number"""

    return mach * np.sqrt(gamma * R * temperature)


def _mass_flow_rate(
        pressure: float,
        temperature: float,
        area: float,
        gamma: float,
        R: float,
        nozzle_efficiency: float,
) -> float:
    """Return mass flow rate."""

    return (
        pressure * area / np.sqrt(temperature) *
        np.sqrt(gamma / R * (2.0 / (gamma + 1.0)) **
                ((gamma + 1.0) / (gamma - 1.0))) *
        np.sqrt(nozzle_efficiency))


def _isentropic_relation(
        gamma: float, mach: float | np.ndarray,
) -> float | np.ndarray:
    """Return the isentropic compressibility factor"""

    return 1.0 + (gamma - 1.0) / 2.0 * mach ** 2


def _primary_exhaust_mach_residual(
        mach: np.ndarray,
        gamma: float,
        area_ratio: float,
) -> float | np.ndarray:
    """Return residual for the primary nozzle exit Mach number."""

    residual = (
        1.0 / mach ** 2 * (2.0 / (gamma + 1.0) * _isentropic_relation(gamma, mach)) **
        ((gamma + 1.0) / (gamma - 1.0)) - area_ratio ** 2)
    return residual


def _primary_choke_mach_residual(
        mach: np.ndarray,
        gamma: float,
        mach_exhaust: float,
        pressure_ratio: float,
) -> float | np.ndarray:
    """Return residual for the primary choke Mach number."""

    return (
        _isentropic_relation(gamma, mach_exhaust) **
        (gamma / (gamma - 1.0)) /
        _isentropic_relation(gamma, mach) ** (gamma / (gamma - 1.0)) -
        pressure_ratio)


def _mixing_coeff(area_ratio: float) -> float:
    """Return empirical mixing coefficient"""

    if area_ratio > 8.3:
        return 0.80
    if area_ratio > 6.9:
        return 0.82
    return 0.84

# NOTE Почитать:
# NOTE https://gist.github.com/sloria/7001839
# NOTE + глянуть ссылки внизу страницы
