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


_M_TO_MM = 1e3
_M2_TO_CM2 = 1e4
_PA_TO_BAR = 1e-5
_K_TO_C = 273.15


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

    def report(self) -> None:
        print("         jet / carry / mix")
        print("flow rates")
        print(
            f"         : {self.flow_rate[J]:.2f}"
            f" / {self.flow_rate[C]:.2f}"
            f" / {self.flow_rate[M]:.2f} [kg/s]")
        print()
        print("temperatures")
        print(
            f"Inlet    : {self.temperature[J, I]-_K_TO_C:.2f}"
            f" / {self.temperature[C, I]-_K_TO_C:.2f}"
            f" / {self.temperature[M, I]-_K_TO_C:.2f} [C]")
        print(
            f"Lobby    : {self.temperature[J, L]-_K_TO_C:.2f}"
            f" / {self.temperature[C, L]-_K_TO_C:.2f}"
            f" / {self.temperature[M, L]-_K_TO_C:.2f} [C]")
        print(
            f"Premix   : {self.temperature[J, PM]-_K_TO_C:.2f}"
            f" / {self.temperature[C, PM]-_K_TO_C:.2f}"
            f" / {self.temperature[M, PM]-_K_TO_C:.2f} [C]")
        print(
            f"Aftermix : {self.temperature[J, AM]-_K_TO_C:.2f}"
            f" / {self.temperature[C, AM]-_K_TO_C:.2f}"
            f" / {self.temperature[M, AM]-_K_TO_C:.2f} [C]")
        print(
            f"Drain    : {self.temperature[J, D]-_K_TO_C:.2f}"
            f" / {self.temperature[C, D]-_K_TO_C:.2f}"
            f" / {self.temperature[M, D]-_K_TO_C:.2f} [C]")
        print()
        print("pressure")
        print(
            f"Inlet    : {self.pressure[J, I]*_PA_TO_BAR:.2f}"
            f" / {self.pressure[C, I]*_PA_TO_BAR:.2f}"
            f" / {self.pressure[M, I]*_PA_TO_BAR:.2f} [bar]")
        print(
            f"Lobby    : {self.pressure[J, L]*_PA_TO_BAR:.2f}"
            f" / {self.pressure[C, L]*_PA_TO_BAR:.2f}"
            f" / {self.pressure[M, L]*_PA_TO_BAR:.2f} [bar]")
        print(
            f"Premix   : {self.pressure[J, PM]*_PA_TO_BAR:.2f}"
            f" / {self.pressure[C, PM]*_PA_TO_BAR:.2f}"
            f" / {self.pressure[M, PM]*_PA_TO_BAR:.2f} [bar]")
        print(
            f"Aftermix : {self.pressure[J, AM]*_PA_TO_BAR:.2f}"
            f" / {self.pressure[C, AM]*_PA_TO_BAR:.2f}"
            f" / {self.pressure[M, AM]*_PA_TO_BAR:.2f} [bar]")
        print(
            f"Drain    : {self.pressure[J, D]*_PA_TO_BAR:.2f}"
            f" / {self.pressure[C, D]*_PA_TO_BAR:.2f}"
            f" / {self.pressure[M, D]*_PA_TO_BAR:.2f} [bar]")


@dataclass
class Design:
    """Ejector geometry computed by solve_dimensions."""

    # Full (phase x location) matrices, nan where not computed
    diameter: np.ndarray
    area: np.ndarray
    length: np.ndarray

    def __post_init__(self) -> None:
        self.m = self.area[MIX, AFTERMIX] / self.area[JET, INLET]
        self.n = self.area[MIX, AFTERMIX] / self.area[CARRY, INLET]

    def report(self) -> None:
        print("area ratios")
        print(f"m (area[MIX, AM] / area[JET,   I]) = {self.m:.2f}")
        print(f"n (area[MIX, AM] / area[CARRY, I]) = {self.n:.2f}")
        print()
        print("         jet / carry / mix")
        print("diameters")
        print(
            f"Inlet    : {self.diameter[J, I]*_M_TO_MM:.2f}"
            f" / {self.diameter[C, I]*_M_TO_MM:.2f}"
            f" / {self.diameter[M, I]*_M_TO_MM:.2f} [mm]")
        print(
            f"Lobby    : {self.diameter[J, L]*_M_TO_MM:.2f}"
            f" / {self.diameter[C, L]*_M_TO_MM:.2f}"
            f" / {self.diameter[M, L]*_M_TO_MM:.2f} [mm]")
        print(
            f"Premix   : {self.diameter[J, PM]*_M_TO_MM:.2f}"
            f" / {self.diameter[C, PM]*_M_TO_MM:.2f}"
            f" / {self.diameter[M, PM]*_M_TO_MM:.2f} [mm]")
        print(
            f"Aftermix : {self.diameter[J, AM]*_M_TO_MM:.2f}"
            f" / {self.diameter[C, AM]*_M_TO_MM:.2f}"
            f" / {self.diameter[M, AM]*_M_TO_MM:.2f} [mm]")
        print(
            f"Drain    : {self.diameter[J, D]*_M_TO_MM:.2f}"
            f" / {self.diameter[C, D]*_M_TO_MM:.2f}"
            f" / {self.diameter[M, D]*_M_TO_MM:.2f} [mm]")
        print()
        print("areas")
        print(
            f"Inlet    : {self.area[J, I]*_M2_TO_CM2:.2f}"
            f" / {self.area[C, I]*_M2_TO_CM2:.2f}"
            f" / {self.area[M, I]*_M2_TO_CM2:.2f} [cm²]")
        print(
            f"Lobby    : {self.area[J, L]*_M2_TO_CM2:.2f}"
            f" / {self.area[C, L]*_M2_TO_CM2:.2f}"
            f" / {self.area[M, L]*_M2_TO_CM2:.2f} [cm²]")
        print(
            f"Premix   : {self.area[J, PM]*_M2_TO_CM2:.2f}"
            f" / {self.area[C, PM]*_M2_TO_CM2:.2f}"
            f" / {self.area[M, PM]*_M2_TO_CM2:.2f} [cm²]")
        print(
            f"Aftermix : {self.area[J, AM]*_M2_TO_CM2:.2f}"
            f" / {self.area[C, AM]*_M2_TO_CM2:.2f}"
            f" / {self.area[M, AM]*_M2_TO_CM2:.2f} [cm²]")
        print(
            f"Drain    : {self.area[J, D]*_M2_TO_CM2:.2f}"
            f" / {self.area[C, D]*_M2_TO_CM2:.2f}"
            f" / {self.area[M, D]*_M2_TO_CM2:.2f} [cm²]")
        print()
        print("lengths")
        print(
            f"Inlet    : {self.length[J, I]*_M_TO_MM:.2f}"
            f" / {self.length[C, I]*_M_TO_MM:.2f}"
            f" / {self.length[M, I]*_M_TO_MM:.2f} [mm]")
        print(
            f"Lobby    : {self.length[J, L]*_M_TO_MM:.2f}"
            f" / {self.length[C, L]*_M_TO_MM:.2f}"
            f" / {self.length[M, L]*_M_TO_MM:.2f} [mm]")
        print(
            f"Premix   : {self.length[J, PM]*_M_TO_MM:.2f}"
            f" / {self.length[C, PM]*_M_TO_MM:.2f}"
            f" / {self.length[M, PM]*_M_TO_MM:.2f} [mm]")
        print(
            f"Aftermix : {self.length[J, AM]*_M_TO_MM:.2f}"
            f" / {self.length[C, AM]*_M_TO_MM:.2f}"
            f" / {self.length[M, AM]*_M_TO_MM:.2f} [mm]")
        print(
            f"Drain    : {self.length[J, D]*_M_TO_MM:.2f}"
            f" / {self.length[C, D]*_M_TO_MM:.2f}"
            f" / {self.length[M, D]*_M_TO_MM:.2f} [mm]")
        print()


def solve_dimensions(
        req: Requirements,
        s: float = S, alpha: float = ALPHA) -> Design:
    """Solve ejector equation for design dimensions with known requirements."""

    q = req.flow_rate[CARRY] / req.flow_rate[JET]
    dp = req.pressure[MIX, AFTERMIX] - req.pressure[JET, INLET]

    jet_density, carry_density, mix_density = _get_densities(req)

    hj = req.velocity_head[JET, INLET]
    req.velocity[JET, INLET] = np.sqrt(2 * hj / jet_density)

    # n was derived from the basic ejection equation by substituting m_opt
    n = ((1 + q) ** 2 * jet_density / mix_density - hj / (2 * dp)) / (
        q ** 2 * jet_density / carry_density
    )

    m = (2 * (1 + q) ** 2 * jet_density / mix_density -
         q ** 2 * n * jet_density / carry_density)

    # diameters
    diameter = CONTAINER.copy()
    diameter[JET, INLET] = np.sqrt(
        4 / np.pi * req.flow_rate[JET] / (jet_density * req.velocity[JET, INLET]))
    diameter[MIX, AFTERMIX] = diameter[JET, INLET] * np.sqrt(m)
    diameter[MIX, PREMIX] = diameter[MIX, AFTERMIX]
    diameter[MIX, LOBBY] = diameter[MIX, PREMIX] / 0.9
    diameter[CARRY, INLET] = diameter[MIX, AFTERMIX] / np.sqrt(n)
    diameter[MIX, DRAIN] = diameter[MIX, AFTERMIX] * np.sqrt(s)

    # areas
    area = np.pi * diameter ** 2 / 4

    # lengths
    length_nozzle_to_wall_contact = diameter[JET, INLET] * (4 * (1 + q) - 1.8)
    length_mixing_chamber = 2.5 * diameter[MIX, AFTERMIX]
    length_nozzle_to_premix = length_nozzle_to_wall_contact - \
        0.5 * diameter[MIX, AFTERMIX]

    # length[:, LOC] holds the distance from the *previous* location to LOC;
    # length[:, INLET] is nan (no preceding section).
    length = CONTAINER.copy()
    length[:, PREMIX] = (                                           # lobby  -> premix
        diameter[MIX, LOBBY] - diameter[MIX, PREMIX]) / (2 * np.tan(alpha/2))
    length[:, LOBBY] = length_nozzle_to_premix - \
        length[:, PREMIX]                                           # inlet-> lobby
    length[:, AFTERMIX] = (                                         # premix -> aftermix
        length_nozzle_to_wall_contact + length_mixing_chamber - length_nozzle_to_premix)
    length[:, DRAIN] = (                                            # aftermix -> drain
        diameter[MIX, DRAIN] - diameter[MIX, AFTERMIX]) / (2 * np.tan(alpha/2))

    # diffuser pressure
    velocity_mix_aftermix = req.flow_rate[MIX] / \
        (mix_density * area[MIX, AFTERMIX])

    pressure_recovery_coeff = _recovery_coeff(s, alpha)

    req.pressure[MIX, PREMIX] = req.pressure[MIX, AFTERMIX]
    req.temperature[MIX, PREMIX] = req.temperature[MIX, AFTERMIX]

    req.pressure[MIX, DRAIN] = (
        req.pressure[MIX, AFTERMIX] +
        pressure_recovery_coeff * mix_density * velocity_mix_aftermix ** 2 / 2
    )

    mix_entropy_aftermix = req.phase[MIX].smass()
    _isentropic_mixture_jump(
        req.phase[MIX], req.pressure[MIX, DRAIN], mix_entropy_aftermix)
    req.temperature[MIX, DRAIN] = req.phase[MIX].T()

    return Design(
        diameter=diameter,
        area=area,
        length=length,
    )


# auxiliary functions
def _recovery_coeff(s: float, alpha: float) -> float:
    """Return pressure recovery coefficient for a diffuser."""
    friction_coeff = 0.002 / np.sin(alpha / 2) * (s ** 2 - 1) / s
    expansion_coeff = np.sin(alpha) * ((s - 1) / s) ** 2
    outlet_coeff = 1.0 / s ** 2
    return 1.0 - (friction_coeff + expansion_coeff + outlet_coeff)


def _get_densities(  # NOTE более подробное имя для функции, см. ниже
    conditions: OperationConditions,
) -> tuple[float, float, float]:
    """Return (jet, carry, mix) mass densities [kg/m³]."""

    # NOTE эта функция снимает плотности в определённом сечении, то есть мы ещё "прыгаем"
    # NOTE всеми уравнениями состояния
    # NOTE
    # NOTE от _get_densities ожидаешь чего-то такого:
    # NOTE >> def _get_densities(conditions):
    # NOTE >>     return [eos.rhomass() for eos in condition.phase]
    # NOTE
    # NOTE в нашем случае функция делает больше, поэтоиу лучше отразить это как-то в
    # NOTE идентификаторе

    # jet phase
    conditions.phase[JET].update(
        PT_INPUTS,
        conditions.pressure[JET, INLET],
        conditions.temperature[JET, INLET],
    )
    jet_density = conditions.phase[JET].rhomass()

    # carried phase
    conditions.phase[CARRY].update(
        PT_INPUTS,
        conditions.pressure[CARRY, INLET],
        conditions.temperature[CARRY, INLET],
    )
    carry_density = conditions.phase[CARRY].rhomass()

    # mixture

    # We need mixture density in the section after mixing. We jump here isentropically
    # from conditions in the lobby
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
    # Save the temperature that the isentropic jump already computed.
    conditions.temperature[MIX, AFTERMIX] = conditions.phase[MIX].T()

    return jet_density, carry_density, mix_density


def _isentropic_mixture_jump_residual(
        mix: AbstractState, p: np.ndarray, T: np.ndarray, smass: float) -> float:
    p, T = np.atleast_1d(p), np.atleast_1d(T)
    mix.update(PT_INPUTS, p[0], T[0])
    return mix.smass() - smass


def _isentropic_mixture_jump(
        mix: AbstractState, p: float, smass: float) -> AbstractState:
    """Update *mix* state isentropically to pressure *p*."""
    T_original = mix.T()

    # Use linear approximation as an initial guess
    p_original = mix.p()
    dSdP = mix.first_partial_deriv(iSmass, iP, iT)
    dSdT = mix.first_partial_deriv(iSmass, iT, iP)
    T_guess = T_original + dSdP / dSdT * (p - p_original)

    T_goal = fsolve(
        lambda T: _isentropic_mixture_jump_residual(
            mix, p, T, smass),  # type: ignore
        T_guess,
    )[0]

    mix.update(PT_INPUTS, p, T_goal)
    return mix


if __name__ == "__main__":

    # Phases and parameters specification
    jet_phase = AbstractState("HEOS", "N2")
    carry_phase = AbstractState("HEOS", "CH4")

    q = 2.0
    carried_phase_flow_rate = 0.07  # kg/s

    requirements = OperationConditions(
        phase=[jet_phase, carry_phase],
        flow_rate=np.array(
            [carried_phase_flow_rate / q, carried_phase_flow_rate]),
    )

    # Boundary conditions
    requirements.pressure[JET, INLET] = 3e5        # Pa
    requirements.pressure[CARRY, INLET] = 2e5        # Pa
    requirements.pressure[MIX, AFTERMIX] = 1.1e5      # Pa
    requirements.temperature[JET, INLET] = 25 + _K_TO_C
    requirements.temperature[CARRY, INLET] = 25 + _K_TO_C

    # Initial mixture state for isentropic jump
    requirements.pressure[MIX, LOBBY] = requirements.pressure[JET, INLET]
    requirements.temperature[MIX, LOBBY] = requirements.temperature[JET, INLET]

    # Velocity head at jet inlet — given as a requirement
    requirements.velocity_head[JET, INLET] = 5000.0  # Pa

    design = solve_dimensions(requirements)
    design.report()
    requirements.report()

    # NOTE надо выводить больше знаков после запятой для расхода, если всё в м^3/ч
    # NOTE и лучше воспользоваться форматированием с применением научной нотации
    # NOTE print(f"flow rate : {flow_rate : .5e}") <- пример

    # NOTE потом надо поисктьа ещё литератуту и закрыть пробелы в методике
