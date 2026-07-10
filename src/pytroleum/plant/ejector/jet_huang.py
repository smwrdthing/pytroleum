from __future__ import annotations
from jet_huang_report import report_inputs
from jet_huang_report import report_dimensions
from jet_huang_report import report_conditions

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytroleum.tdyna.CoolStub import AbstractState  # type: ignore
else:
    from CoolProp import AbstractState
from CoolProp.constants import PT_INPUTS

import numpy as np
from scipy.optimize import fsolve

from jet_huang_interfaces import Loc, Phase, _CONTAINER

_R_UNIV = 8.314462618

# Huang et al. (1999), Eqs. (1), (5), (7); Section 4: np, ns, fp; Eq. (19): fm
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
        report_inputs(self)


@dataclass
class OperationConditions:

    phase: AbstractState
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
        self.phase = req.phase
        # NOTE надёжнее будет сконструировать новые объекты для уравнений состояний,
        # NOTE здесь может оказаться так, что self.phase и req.phase - одно и то же,
        # NOTE тогда при изменении self.phase будет меняться и req.phase, при копировании
        # NOTE объектов из полей классов такую связь нужно исключить

        self.pressure = req.pressure.copy()
        self.temperature = req.temperature.copy()

    def report(self) -> None:
        report_conditions(self)


@dataclass
class Design:
    """Ejector geometry from Huang et al. critical-mode analysis."""

    diameter: np.ndarray
    area: np.ndarray
    length: np.ndarray

    def report(self) -> None:
        report_dimensions(self)


def solve_dimensions(
        req: Requirements,

        design: Design,
        # NOTE design лучше создавать внутри функции, которая выполняет проектный расчёт
        # NOTE и возвращать вместе с OperationConditions (см. заметки в __init__.py)

        Pc_rel_tolerance: float = _PC_REL_TOLERANCE,
        max_iter: int = _MAX_ITER,
) -> OperationConditions:  # NOTE сигнатура вызова и имя функции противоречат друг-другу
    """Huang et al. (1999) critical-mode analysis, Eqs. (1)–(18), Fig. 3."""

    conditions = OperationConditions(phase=req.phase)
    conditions.read_requirements(req)

    gamma, R, cp = _extract_properties_for(
        conditions.phase,
        conditions.pressure[Phase.P, Loc.IN],
        conditions.temperature[Phase.P, Loc.IN],
    )
    # NOTE если мы дальше везде тащим эти параметры - лучше записать их один раз
    # NOTE в conditions и передавать только conditions
    _primary_mass_flow(conditions, design, gamma, R)  # NOTE здесь будет чище
    _primary_exhaust_state(conditions, design, gamma)
    _secondary_choke_state(conditions, gamma)

    design.area[Phase.M, Loc.AM] = (
        design.area[Phase.P, Loc.TH] * _A3_INITIAL_RATIO)

    iter_count = 0
    while True:
        while True:
            _primary_core_state(conditions, design, gamma)
            _secondary_choke_area(design)

            if design.area[Phase.S, Loc.CH] >= 0.0:
                # NOTE Условие цикла сразу в while, break здесь не нужен
                break

            # A3 = Apy + ΔA3: mixed-section area from primary core area.
            design.area[Phase.M, Loc.AM] = design.area[Phase.P, Loc.CH]
            design.area[Phase.M, Loc.AM] += _DA3

        _secondary_mass_flow(conditions, design, gamma, R)
        _choke_temperatures(conditions, gamma)
        _mix_pre_shock_velocity_pressure(conditions, design, gamma, R)
        _mix_pre_shock_temperature_mach(conditions, gamma, R, cp)
        _aftermix_state(conditions, gamma)
        _mix_drain_pressure(conditions, gamma)

        Pc = conditions.pressure[Phase.M, Loc.D]
        # NOTE условие можно в функцию от conditoins, будет чище + не надо вытаскивать Pc
        if abs(Pc - req.Pc_star) / req.Pc_star <= Pc_rel_tolerance:
            # NOTE Условие цикла сразу в while, break здесь не нужен
            break
        # NOTE в итоге должно быть что-то вроде
        # NOTE >> while not converged(conditions):
        # NOTE >>     # код цикла
        # NOTE может быть
        # NOTE >> while not conditions.converged(): # если сделать converged методом
        # NOTE >>     # код цикла
        # NOTE
        # NOTE Сигнатура функции, проверящей сходимость, может быть сложнее, но в целом
        # NOTE можно добиться вида сверху, если записать недостающее в conditions

        if iter_count >= max_iter:
            raise RuntimeError(
                f"Solution algorithm did not converge in {max_iter} iterations: "
                f"discharge pressure Pc did not reach target Pc_star "
                f"({req.Pc_star:.4g} Pa) within relative tolerance "
                f"{Pc_rel_tolerance:.4g}.")

        if Pc >= req.Pc_star:
            design.area[Phase.M, Loc.AM] += _DA3
        else:
            design.area[Phase.M, Loc.AM] -= _DA3
        iter_count += 1

    _finalize_mix_geometry(design)
    return conditions


def _primary_mass_flow(
        conditions: OperationConditions,
        design: Design,
        gamma: float,
        R: float,
) -> None:
    """Fig. 3 — Eq. (1): mp."""
    # NOTE докстринги нужны для документации функции, короткие ссылки на внешние
    # NOTE источники вне кода - комментариями (либо вовсе исключить, см. заметки выше)
    # NOTE
    # NOTE Либо для полного докстринга можно оставить ссылку внизу в качестве
    # NOTE дополнительной информации, оформление таких штук можно подсмотреть в
    # NOTE библиотеках типа numpy, scipy

    conditions.mass_flow_rate[Phase.P] = _mass_flow_rate(
        conditions.pressure[Phase.P, Loc.IN],
        conditions.temperature[Phase.P, Loc.IN],
        design.area[Phase.P, Loc.TH],
        gamma, R, PRIMARY_NOZZLE_EFF)


def _primary_exhaust_state(
        conditions: OperationConditions,
        design: Design,
        gamma: float,
) -> None:
    """Fig. 3 — Eqs. (2), (3): Mp1, Pp1."""

    conditions.mach[Phase.P, Loc.EX] = fsolve(
        _primary_exhaust_mach_residual,
        [MACH_GUESS],
        args=(
            gamma,
            design.area[Phase.P, Loc.EX] /
            design.area[Phase.P, Loc.TH]),
    )[0]
    # NOTE Идея, чтобы повысить читаемость:
    # NOTE Для отношения площадей можно в design завести метод area_ratio, который будет
    # NOTE принимать целочисленные значения, получится что-то типа
    # NOTE >> design.area_ratio(Loc.EX, Loc.TH)

    conditions.pressure[Phase.P, Loc.EX] = (
        conditions.pressure[Phase.P, Loc.IN] /
        _isentropic_relation(gamma, conditions.mach[Phase.P, Loc.EX]) **
        (gamma / (gamma - 1.0)))


def _secondary_choke_state(
        conditions: OperationConditions,
        gamma: float,
) -> None:
    """Fig. 3 — Eq. (6): Msy, Psy."""

    conditions.mach[Phase.S, Loc.CH] = 1.0
    conditions.pressure[Phase.S, Loc.CH] = (
        conditions.pressure[Phase.S, Loc.IN] /
        _isentropic_relation(gamma, conditions.mach[Phase.S, Loc.CH]) **
        (gamma / (gamma - 1.0)))


def _primary_core_state(
        conditions: OperationConditions,
        design: Design,
        gamma: float,
) -> None:
    """Fig. 3 — Eqs. (4), (5): Mpy, Apy."""

    conditions.pressure[Phase.P, Loc.CH] = (
        conditions.pressure[Phase.S, Loc.CH])
    conditions.mach[Phase.P, Loc.CH] = fsolve(
        _primary_choke_mach_residual,
        [MACH_GUESS],
        args=(
            gamma,
            conditions.mach[Phase.P, Loc.EX],
            conditions.pressure[Phase.S, Loc.CH] /
            conditions.pressure[Phase.P, Loc.EX]),
    )[0]
    design.area[Phase.P, Loc.CH] = (
        design.area[Phase.P, Loc.EX] *
        (PRIMARY_CORE_AREA_FACTOR / conditions.mach[Phase.P, Loc.CH] *
         (2.0 / (gamma + 1.0) *
          _isentropic_relation(gamma, conditions.mach[Phase.P, Loc.CH])) **
         ((gamma + 1.0) / (2.0 * (gamma - 1.0)))) /
        (1.0 / conditions.mach[Phase.P, Loc.EX] *
         (2.0 / (gamma + 1.0) *
          _isentropic_relation(gamma, conditions.mach[Phase.P, Loc.EX])) **
         ((gamma + 1.0) / (2.0 * (gamma - 1.0)))))


def _secondary_choke_area(
        design: Design,
) -> None:
    """Fig. 3 — Eq. (8): Asy."""

    design.area[Phase.S, Loc.CH] = (
        design.area[Phase.M, Loc.AM] -
        design.area[Phase.P, Loc.CH])


def _secondary_mass_flow(
        conditions: OperationConditions,
        design: Design,
        gamma: float,
        R: float,
) -> None:
    """Fig. 3 — Eq. (7): ms."""

    conditions.mass_flow_rate[Phase.S] = _mass_flow_rate(
        conditions.pressure[Phase.S, Loc.IN],
        conditions.temperature[Phase.S, Loc.IN],
        design.area[Phase.S, Loc.CH],
        gamma, R, SECONDARY_NOZZLE_EFF)


def _choke_temperatures(
        conditions: OperationConditions,
        gamma: float,
) -> None:
    """Fig. 3 — Eqs. (9), (10): Tpy, Tsy."""

    conditions.temperature[Phase.P, Loc.CH] = (
        conditions.temperature[Phase.P, Loc.IN] /
        _isentropic_relation(gamma, conditions.mach[Phase.P, Loc.CH]))
    conditions.temperature[Phase.S, Loc.CH] = (
        conditions.temperature[Phase.S, Loc.IN] /
        _isentropic_relation(gamma, conditions.mach[Phase.S, Loc.CH]))


def _mix_pre_shock_velocity_pressure(
        conditions: OperationConditions,
        design: Design,
        gamma: float,
        R: float,
) -> None:
    """Fig. 3 — Eqs. (11), (13), (14), (19): Pm, Vpy, Vsy, Vm."""

    conditions.velocity[Phase.P, Loc.CH] = _velocity_from_mach(
        conditions.mach[Phase.P, Loc.CH], gamma, R,
        conditions.temperature[Phase.P, Loc.CH])

    conditions.velocity[Phase.S, Loc.CH] = _velocity_from_mach(
        conditions.mach[Phase.S, Loc.CH], gamma, R,
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


def _mix_pre_shock_temperature_mach(
        conditions: OperationConditions,
        gamma: float,
        R: float,
        cp: float,
) -> None:
    """Fig. 3 — Eqs. (12), (15): Tm, Mm."""

    primary_choke_energy = (
        cp * conditions.temperature[Phase.P, Loc.CH] +
        conditions.velocity[Phase.P, Loc.CH] ** 2 / 2.0)
    secondary_choke_energy = (
        cp * conditions.temperature[Phase.S, Loc.CH] +
        conditions.velocity[Phase.S, Loc.CH] ** 2 / 2.0)

    conditions.temperature[Phase.M, Loc.PS] = 1/cp*(
        (conditions.mass_flow_rate[Phase.P] * primary_choke_energy +
         conditions.mass_flow_rate[Phase.S] * secondary_choke_energy) /
        (conditions.mass_flow_rate[Phase.P] +
         conditions.mass_flow_rate[Phase.S]) -
        conditions.velocity[Phase.M, Loc.PS] ** 2 / 2.0)

    conditions.mach[Phase.M, Loc.PS] = (
        conditions.velocity[Phase.M, Loc.PS] /
        np.sqrt(gamma * R * conditions.temperature[Phase.M, Loc.PS]))


def _aftermix_state(
        conditions: OperationConditions,
        gamma: float,
) -> None:
    """Fig. 3 — Eqs. (16), (17): P3, M3."""

    conditions.pressure[Phase.M, Loc.AM] = (
        conditions.pressure[Phase.M, Loc.PS] *
        (1.0 + 2.0 * gamma / (gamma + 1.0) *
         (conditions.mach[Phase.M, Loc.PS] ** 2 - 1.0)))
    conditions.mach[Phase.M, Loc.AM] = np.sqrt(
        _isentropic_relation(gamma, conditions.mach[Phase.M, Loc.PS]) /
        (gamma * conditions.mach[Phase.M, Loc.PS] ** 2 -
         (gamma - 1.0) / 2.0))


def _mix_drain_pressure(
        conditions: OperationConditions,
        gamma: float,
) -> None:
    """Fig. 3 — Eq. (18): Pc."""

    conditions.pressure[Phase.M, Loc.D] = (
        conditions.pressure[Phase.M, Loc.AM] *
        _isentropic_relation(gamma, conditions.mach[Phase.M, Loc.AM]) **
        (gamma / (gamma - 1.0)))


def _finalize_mix_geometry(design: Design) -> None:
    """Set mix-section diameters"""

    design.diameter[Phase.M, Loc.AM] = np.sqrt(
        4.0 * design.area[Phase.M, Loc.AM] / np.pi)

    for loc in (Loc.PM, Loc.CH, Loc.PS, Loc.SH):
        design.diameter[Phase.M, loc] = (
            design.diameter[Phase.M, Loc.AM])


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

# NOTE Идея для рефакторинга: многие функции работают только с conditions, их можно
# NOTE cделать методами класса OperationConditions

# NOTE Почитать:
# NOTE https://gist.github.com/sloria/7001839
# NOTE + глянуть ссылки внизу страницы

# NOTE Привести файлы в пакете в порядок: убать лишнее, пересмотреть имена нужных файлов
# NOTE имена файлов - часть API (то, что видит пользователь), они должны быть как можно
# NOTE более ёмкими и репрезентативными
