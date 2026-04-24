from __future__ import annotations
from abc import ABC
from dataclasses import dataclass, field

from typing import Iterable, overload, Literal
from numpy import float64
from numpy.typing import NDArray

import numpy as np
from CoolProp import constants as CoolConst

from pytroleum.tdyna import eos


# Индексы для контейнров с параметрами фаз, с которыми мы работаем в трёхфазном сепараторе
V, VAPOR = 0, 0
O, OIL = 1, 1
W, WATER = 2, 2
# полное еоличество фаз - переменная для удобства
N_FLOWS = 3

# Умолчательные интерфейсы к уравнениям состояния
STANDARD_TEMPERATURE = 273.15 + 15  # C
STANDARD_PRESSURE = 101325  # Pa
STANDARD_STATE = (eos.CoolConst.PT_INPUTS,
                  STANDARD_PRESSURE, STANDARD_TEMPERATURE)

VAPOR_EOS = eos.factory_eos({"Methane": 1}, with_state=STANDARD_STATE)
OIL_EOS = eos.CrudeOilHardcoded()
OIL_EOS.update(*STANDARD_STATE)
WATER_EOS = eos.factory_eos({"Water": 1}, with_state=STANDARD_STATE)

PHASE_EOS_INTERFACES = [VAPOR_EOS, OIL_EOS, WATER_EOS]

# Декларация псевдонимов типов для удобной аннотации
type ThreePhaseFlow = NDArray[float64]

# Эти три вообще должны быть в tdyna, но для примера ок
type EOSInterface = (
    eos.AbstractState |
    eos.AbstractStateImitator |
    eos.CrudeOilHardcoded |
    eos.CrudeOilReferenced)
type CoolPropInputPair = int
type ThermodynamicState = tuple[CoolPropInputPair, float, float]

type PhaseEOSInterfaces = list[EOSInterface]

type TPSIndex = Literal[0, 1, 2]


# Код ниже предоставляет те же возможности, что и OperationConditions с PhysicalProperties
# в inputs.py


@dataclass
class OperationConditions:

    # default_factory нужны из-за изменяемости списков, нельзя давать полям значения по
    # умолчанию с изменяемыми типами

    phase: PhaseEOSInterfaces = field(
        default_factory=lambda: PHASE_EOS_INTERFACES)

    # Две переменные состояния для удобства и промежуточных расчётов можно оставить тут,
    # всё остальное достаётся через уравнения состояния в self.phase
    pressure: ThreePhaseFlow = field(
        default_factory=lambda: np.zeros(N_FLOWS))
    temperature: ThreePhaseFlow = field(
        default_factory=lambda: np.zeros(N_FLOWS))

    vol_flow_rate: ThreePhaseFlow = field(
        default_factory=lambda: np.zeros(N_FLOWS))
    mass_flow_rate: ThreePhaseFlow = field(
        default_factory=lambda: np.zeros(N_FLOWS))

    def update_state(self, new_state: ThermodynamicState, upd_containers: bool = False):
        '''Convenience method to change state of all phases with new state'''
        for phase in self.phase:
            phase.update(*new_state)

        if upd_containers:
            # Если хотим автоматически обновлять контенйреы в этом классе передаём
            # true в upd_containers, по умолчанию False
            if new_state[0] != CoolConst.PT_INPUTS:
                # Проверяем идентификтаор пары входных параметров, если PT,
                # то всё в порядке, иначе входные данные невалидны
                raise ValueError(
                    "Can't update pressure and temperature containers," +
                    "incorrect pair provided")
            # Если не зашли в предыдущий блок - всё в порядке, обновляем давления и
            # температуры
            self.pressure[:] = new_state[1]
            self.temperature[:] = new_state[2]
            # ^^^ этот метод ставит всем фазам одинаковое состояние, в общем случае
            # в потоке у разных фаз могут быть разные параметры


def reference_state_flow_rate(
        conditions: OperationConditions,
        reference_state: ThermodynamicState) -> NDArray[float64]:
    '''Recompute flow rates for given refernce state'''

    # В conditions есть pressure и temperature, информация о рабочих параметрах
    # сохраняется в этих атрибутах, можем обновлять уравнения состояния компонентов
    # безопасно (флаг upd_containers False по умолчанию)

    # Пересчитывается только расход газовой фазы

    # Обновляем уравнение состояния для опорного термодинамического состояния,
    # запоминаем плотность в опроном состоянии, возвращаем всё назад

    conditions.phase[V].update(*reference_state)
    refstate_density = conditions.phase[V].rhomass()
    conditions.phase[V].update(
        CoolConst.PT_INPUTS, conditions.pressure[V], conditions.temperature[V])

    vol_flow_rate = np.array([
        conditions.mass_flow_rate[V]/refstate_density,
        conditions.vol_flow_rate[O],
        conditions.vol_flow_rate[W]
    ])

    return vol_flow_rate


def flow_based_water_cut(condition: OperationConditions):
    '''Compute water cut based on volumetric flow rates in
    multiphase vapor-oil-water flow'''

    cut = condition.vol_flow_rate[W]/(
        condition.vol_flow_rate[W]+condition.vol_flow_rate[O])

    return cut


def flow_based_vapor_quality(condition: OperationConditions,
                             reference_state: ThermodynamicState | None = None):
    '''Compute vapor quality (i.e. gas factor) based on volumetric flow rates in
    multiphase vapor-oil-water flow'''

    vol_flow_rates = condition.vol_flow_rate
    if reference_state is not None:
        vol_flow_rates = reference_state_flow_rate(condition, reference_state)

    return vol_flow_rates[V]/np.sum(vol_flow_rates)


def flow_velocity(conditions: OperationConditions,
                  effective_area: float | NDArray[float64]) -> NDArray[float64]:
    # Логика может быть более сложной
    return conditions.vol_flow_rate/effective_area

# Решения сверху - тоже не истина в последней инстанции, думаю ещё есть пространство для
# улучшения
#
# Можно, например, улучшить механизм установки новых значений расхода, сделать функцию
# для укащания расходов через газовый фактор и обводнённость и т.д.
#
# Основная цель организации кода в том, чтобы сделать его использование максимально
# простым и инутитвино понятным, насколько это возможно - принмать архитектурные решения
# можно руководствуясь этой установкой


if __name__ == "__main__":

    # Пример использования

    conditions = OperationConditions()
    conditions.vol_flow_rate = np.array([
        100_000/24/60/60,
        20_000/24/60/60,
        30_000/24/60/60,
    ])
    conditions.update_state((CoolConst.PT_INPUTS, 3e5, 293.15))

    # Хотим нефть с другими свойствами
    conditions.phase[OIL].change(750, 8e-3, 1722)  # type: ignore
    # type checker ругается, поэтому type: ignore нужно починить аннотацию типов в tdyna,
    # но это несрочно

    # Если хотим совсем другое уравнение состояния - его можно подготовить заранее
    # и установить через =

    # Массовый расход сейчас нужно ставить "вручную", если он нам нужен, тут можно
    # улучшить (как варинат - сделать через property)
    conditions.mass_flow_rate = conditions.vol_flow_rate*(
        np.array(list([phase.rhomass() for phase in conditions.phase]))
    )

    diameter = 300e-3
    area = np.pi*diameter**2/4

    print(f"Массовый расход {conditions.mass_flow_rate} кг/с")
    print(f"Объёмный расход {conditions.vol_flow_rate} м^3/c")
    print(f"Скорости {flow_velocity(conditions, area)} м/с")
