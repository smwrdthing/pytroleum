from __future__ import annotations
from dataclasses import dataclass, field
from CoolProp import constants as CoolConst

import numpy as np
from numpy import float64
from numpy.typing import NDArray
from pytroleum.tdyna import eos
from typing import Literal

# Индексы фаз
A, ACTIVE = 0, 0
P, PASSIVE = 1, 1
M, MIXTURE = 2, 2

N_PHASE = 3

# Стандартное термодинамическое состояние
STANDARD_TEMPERATURE = 273.15 + 15  # К (15°C)
STANDARD_PRESSURE = 101325          # Па
STANDARD_STATE = (CoolConst.PT_INPUTS, STANDARD_PRESSURE, STANDARD_TEMPERATURE)

# Объекты EOS по умолчанию
ACTIVE_EOS = eos.factory_eos({"Air": 1}, with_state=STANDARD_STATE)
PASSIVE_EOS = eos.factory_eos({"Air": 1}, with_state=STANDARD_STATE)
MIXTURE_EOS = eos.factory_eos({"Air": 1}, with_state=STANDARD_STATE)

# Список EOS для всех фаз
PHASE_EOS_INTERFACES: PhaseEOSInterfaces = [ACTIVE_EOS,
                                            PASSIVE_EOS,
                                            MIXTURE_EOS]

# Псевдонимы типов
type ThreePhase = NDArray[float64]

type EOSInterface = (
    eos.AbstractState |
    eos.AbstractStateImitator |
    eos.CrudeOilHardcoded |
    eos.CrudeOilReferenced)
type CoolPropInputPair = int
type ThermodynamicState = tuple[CoolPropInputPair, float, float]
type PhaseEOSInterfaces = list[EOSInterface]
type EjectorIndex = Literal[0, 1, 2]


@dataclass
class OperationConditions:

    phase: PhaseEOSInterfaces = field(
        default_factory=lambda: PHASE_EOS_INTERFACES)

    pressure: ThreePhase = field(
        default_factory=lambda: np.zeros(N_PHASE))
    temperature: ThreePhase = field(
        default_factory=lambda: np.zeros(N_PHASE))
    mass_flow_rate: ThreePhase = field(
        default_factory=lambda: np.zeros(N_PHASE))

    def update_state(self, new_state: ThermodynamicState,
                     index: EjectorIndex,
                     upd_containers: bool = False) -> None:
        """Обновляет термодинамическое состояние одной фазы."""
        self.phase[index].update(*new_state)

        if upd_containers:
            if new_state[0] != CoolConst.PT_INPUTS:
                raise ValueError(
                    "Can't update pressure and temperature containers, "
                    "incorrect pair provided")
            self.pressure[index] = new_state[1]
            self.temperature[index] = new_state[2]


@dataclass
class Requirements:
    """Общие входные параметры"""
    num_stages: int
    outlet_pressure: float        # Па
    outlet_diameter: float        # м
    active_inlet_diameter: float  # м
    passive_inlet_diameter: float  # м
