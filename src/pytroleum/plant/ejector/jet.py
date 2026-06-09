from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytroleum.tdyna.CoolStub import AbstractState  # type: ignore
else:
    from CoolProp import AbstractState
from CoolProp.constants import PT_INPUTS, iSmass, iP, iT

import numpy as np
from scipy.optimize import fsolve

# Location indices
I = INLET = 0  # noqa: E741
L = LOBBY = 1
PM = PREMIX = 2
AM = AFTERMIX = 3
D = DRAIN = 4

# Phase indices
J = JET = 0
C = CARRY = 1
M = MIX = 2
A = ANY = -1

# Template matrix (phase × location) filled with nan — copy before use
_LAST_PHASE = M + 1
_LAST_LOC = D + 1
_SHAPE = (_LAST_PHASE, _LAST_LOC)
CONTAINER = np.full(_SHAPE, np.nan)

type Requirements = OperationConditions  # to separate semantically

# Diffuser geometry constants
S = 2.0
ALPHA = np.radians(6.0)


@dataclass
class OperationConditions:

    phase: list[AbstractState]
    flow_rate: np.ndarray

    pressure: np.ndarray = field(default_factory=lambda: CONTAINER.copy())
    temperature: np.ndarray = field(default_factory=lambda: CONTAINER.copy())

    velocity: np.ndarray = field(default_factory=lambda: CONTAINER.copy())
    velocity_head: np.ndarray = field(default_factory=lambda: CONTAINER.copy())

    def __post_init__(self) -> None:
        self._setup_mixture()

    def _setup_mixture(self) -> None:
        """Auxiliary method to set up mixture EoS"""

        self._validate_eos()

        self.flow_rate = np.array([*self.flow_rate, np.sum(self.flow_rate)])

        # .fluid_names() always returns list, even for pure fluids, we take advantage of
        # that to build string for mixture
        self.phase.append(
            AbstractState(
                self.phase[J].backend_name(),
                "&".join(self.phase[J].fluid_names() +
                         self.phase[C].fluid_names()),
            )
        )
        self.phase[MIX].set_mass_fractions(
            self.flow_rate[:MIX] / self.flow_rate[MIX])

    def _validate_eos(self):
        if self.phase[J].backend_name() != self.phase[C].backend_name():
            raise AttributeError(
                "Inconsistent backend for phases, for jet phase" +
                self.phase[J].backend_name() + " was provided while carry phase uses " +
                self.phase[C].backend_name())


@dataclass
class EjectorDesign:
    """Ejector geometry computed by solve_dimensions."""

    # Area ratios
    m: float   # area[MIX, AM] / area[JET, I]
    n: float   # area[MIX, AM] / area[CARRY, I]

    # Cross-sectional areas [m²]
    area_jet_inlet: float
    area_carry_inlet: float
    area_mix_aftermix: float

    # Diameters [m]
    diameter_jet_inlet: float
    diameter_carry_inlet: float
    diameter_mix_aftermix: float

    # Diffuser outlet pressure [Pa]
    pressure_mix_drain: float


def solve_dimensions(
        req: Requirements, area_jet_inlet: float,
        s: float = S, alpha: float = ALPHA) -> EjectorDesign:
    """Solve ejector equation for design dimensions with known requirements and jet nozzle area"""

    q = req.flow_rate[CARRY] / req.flow_rate[JET]
    dp = req.pressure[MIX, AFTERMIX] - req.pressure[JET, INLET]

    jet_density, carry_density, mix_density = _get_densities(req)

    # hj from given nozzle area — same approach as in the analysis problem
    req.velocity[JET, INLET] = req.flow_rate[JET] / \
        (jet_density * area_jet_inlet)
    req.velocity_head[JET, INLET] = jet_density * \
        req.velocity[JET, INLET] ** 2 / 2
    hj = req.velocity_head[JET, INLET]

    # find n from ejector momentum equation:
    # dp/hj = 1/2 * 1 / [(1+q)²·ρj/ρm − q²·n·ρj/ρc]
    # → (1+q)²·ρj/ρm − q²·n·ρj/ρc = hj/(2·dp)
    n = ((1 + q) ** 2 * jet_density / mix_density - hj / (2 * dp)) / (
        q ** 2 * jet_density / carry_density
    )

    # find m: m = 2·(1+q)²·ρj/ρm − q²·n·ρj/ρc
    m = (2 * (1 + q) ** 2 * jet_density / mix_density
         - q ** 2 * n * jet_density / carry_density)

    # remaining areas and diameters from m and n
    area_mix_aftermix = m * area_jet_inlet
    area_carry_inlet = area_mix_aftermix / n

    diameter_jet_inlet = _diameter(area_jet_inlet)
    diameter_mix_aftermix = _diameter(area_mix_aftermix)
    diameter_carry_inlet = _diameter(area_carry_inlet)

    # diffuser pressure recovery: p_drain = p_am + φ·ρ_mix·u_mix²/2
    # φ = 1 − (ε_tr + ε_r + ε_out)
    velocity_mix_aftermix = req.flow_rate[MIX] / \
        (mix_density * area_mix_aftermix)

    epsilon_tr = 0.002 / np.sin(alpha / 2) * (s ** 2 - 1) / s
    epsilon_r = np.sin(alpha) * (s - 1) / s ** 2
    epsilon_out = 1.0 / s ** 2
    phi = 1.0 - (epsilon_tr + epsilon_r + epsilon_out)

    pressure_mix_drain = (
        req.pressure[MIX, AFTERMIX]
        + phi * mix_density * velocity_mix_aftermix ** 2 / 2
    )

    return EjectorDesign(
        m=m,
        n=n,
        area_jet_inlet=area_jet_inlet,
        area_carry_inlet=area_carry_inlet,
        area_mix_aftermix=area_mix_aftermix,
        diameter_jet_inlet=diameter_jet_inlet,
        diameter_carry_inlet=diameter_carry_inlet,
        diameter_mix_aftermix=diameter_mix_aftermix,
        pressure_mix_drain=pressure_mix_drain,
    )


def report_design(design: EjectorDesign) -> None:
    print("         jet / carry / mix")
    print("diameters")
    print(
        f"Inlet    : {design.diameter_jet_inlet * 1e3:.2f}"
        f" / {design.diameter_carry_inlet * 1e3:.2f}"
        f" / nan [mm]")
    print(
        f"Aftermix : nan"
        f" / nan"
        f" / {design.diameter_mix_aftermix * 1e3:.2f} [mm]")
    print()
    print("areas")
    print(
        f"Inlet    : {design.area_jet_inlet * 1e4:.4f}"
        f" / {design.area_carry_inlet * 1e4:.4f}"
        f" / nan [cm²]")
    print(
        f"Aftermix : nan"
        f" / nan"
        f" / {design.area_mix_aftermix * 1e4:.4f} [cm²]")
    print()
    print("area ratios")
    print(f"m (area[MIX,AM] / area[JET,I])   = {design.m:.4f}")
    print(f"n (area[MIX,AM] / area[CARRY,I]) = {design.n:.4f}")
    print()
    print("pressure")
    print(
        f"Drain    : nan / nan / {design.pressure_mix_drain / 1e5:.4f} [bar]")


# auxiliary functions // should be moved to separate module

def _diameter(area: float) -> float:
    return np.sqrt(4 * area / np.pi)


def _get_densities(
    conditions: OperationConditions,
) -> tuple[float, float, float]:
    """Return (jet, carry, mix) mass densities [kg/m³]."""

    conditions.phase[JET].update(
        PT_INPUTS,
        conditions.pressure[JET, INLET],
        conditions.temperature[JET, INLET],
    )
    jet_density = conditions.phase[JET].rhomass()

    conditions.phase[CARRY].update(
        PT_INPUTS,
        conditions.pressure[CARRY, INLET],
        conditions.temperature[CARRY, INLET],
    )
    carry_density = conditions.phase[CARRY].rhomass()

    conditions.phase[MIX].update(
        PT_INPUTS,
        conditions.pressure[MIX, LOBBY],
        conditions.temperature[MIX, LOBBY],
    )
    mix_entropy = conditions.phase[MIX].smass()
    _isentropic_mixture_jump(
        conditions.phase[MIX], conditions.pressure[MIX, AFTERMIX], mix_entropy
    )
    mix_density = conditions.phase[MIX].rhomass()

    return jet_density, carry_density, mix_density


def _isentropic_mixture_jump_residual(
    mix: AbstractState, p: np.ndarray, T: np.ndarray, smass: float
) -> float:
    p, T = np.atleast_1d(p), np.atleast_1d(T)
    mix.update(PT_INPUTS, p[0], T[0])
    return mix.smass() - smass


def _isentropic_mixture_jump(
    mix: AbstractState, p: float, smass: float
) -> AbstractState:
    """Update *mix* state isentropically to pressure *p*."""
    T_original = mix.T()
    p_original = mix.p()

    dSdP = mix.first_partial_deriv(iSmass, iP, iT)
    dSdT = mix.first_partial_deriv(iSmass, iT, iP)
    T_guess = T_original + dSdP / dSdT * (p - p_original)

    T_goal = fsolve(
        lambda T: _isentropic_mixture_jump_residual(mix, p, T, smass),
        T_guess,
    )[0]

    mix.update(PT_INPUTS, p, T_goal)
    return mix


if __name__ == "__main__":
    jet_phase = AbstractState("HEOS", "N2")
    carry_phase = AbstractState("HEOS", "CH4")

    q = 2.0
    G_carry = 0.07  # kg/s

    requirements = OperationConditions(
        phase=[jet_phase, carry_phase],
        flow_rate=np.array([G_carry / q, G_carry]),
    )

    # Boundary conditions — задаём давления и температуры
    requirements.pressure[JET, INLET] = 3e5       # Pa  — задано
    requirements.pressure[CARRY, INLET] = 3e5     # Pa
    requirements.pressure[MIX, AFTERMIX] = 1.1e5  # Pa  — задано
    requirements.temperature[:, INLET] = 25 + 273.15  # K

    # Initial mixture state for isentropic jump
    requirements.pressure[MIX, LOBBY] = requirements.pressure[JET, INLET]
    requirements.temperature[MIX, LOBBY] = requirements.temperature[JET, INLET]

    # Jet nozzle diameter — задан
    diameter_jet_inlet = 15e-3  # m
    area_jet_inlet = np.pi * diameter_jet_inlet ** 2 / 4

    design = solve_dimensions(requirements, area_jet_inlet)
    report_design(design)
