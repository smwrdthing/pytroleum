from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy import float64
from numpy.typing import NDArray
from CoolProp import constants as CoolConst

from pytroleum.tdyna import eos
from pytroleum.meter import volume_cover_elliptic_trunc, volume_section_horiz_ellipses

# Индексы фаз
V, VAPOR = 0, 0
O, OIL = 1, 1
W, WATER = 2, 2

# полное количество фаз
N_FLOWS = 3

# Стандартное термодинамическое состояние
STANDARD_TEMPERATURE = 273.15 + 15  # К (15°C)
STANDARD_PRESSURE = 101325          # Па
STANDARD_STATE = (CoolConst.PT_INPUTS, STANDARD_PRESSURE, STANDARD_TEMPERATURE)

# Объекты EOS по умолчанию
VAPOR_EOS = eos.factory_eos({"Methane": 1}, with_state=STANDARD_STATE)
OIL_EOS = eos.CrudeOilHardcoded()
OIL_EOS.update(*STANDARD_STATE)
WATER_EOS = eos.factory_eos({"Water": 1}, with_state=STANDARD_STATE)

# Список EOS для всех фаз
PHASE_EOS_INTERFACES = [VAPOR_EOS, OIL_EOS, WATER_EOS]

# Псевдонимы типов
type ThreePhaseFlow = NDArray[float64]

type EOSInterface = (
    eos.AbstractState |
    eos.AbstractStateImitator |
    eos.CrudeOilHardcoded |
    eos.CrudeOilReferenced)
type CoolPropInputPair = int
type ThermodynamicState = tuple[CoolPropInputPair, float, float]
type PhaseEOSInterfaces = list[EOSInterface]
type TPSIndex = Literal[0, 1, 2]


@dataclass
class OperationConditions:

    phase: PhaseEOSInterfaces = field(
        default_factory=lambda: PHASE_EOS_INTERFACES)

    pressure: ThreePhaseFlow = field(
        default_factory=lambda: np.zeros(N_FLOWS))
    temperature: ThreePhaseFlow = field(
        default_factory=lambda: np.zeros(N_FLOWS))

    vol_flow_rate: ThreePhaseFlow = field(
        default_factory=lambda: np.zeros(N_FLOWS))
    mass_flow_rate: ThreePhaseFlow = field(
        default_factory=lambda: np.zeros(N_FLOWS))

    def update_state(self, new_state: ThermodynamicState,
                     upd_containers: bool = False) -> None:
        """Обновляет термодинамическое состояние всех фаз."""
        for phase in self.phase:
            phase.update(*new_state)

        if upd_containers:
            if new_state[0] != CoolConst.PT_INPUTS:
                raise ValueError(
                    "Can't update pressure and temperature containers, "
                    "incorrect pair provided")
            self.pressure[:] = new_state[1]
            self.temperature[:] = new_state[2]


def reference_state_flow_rate(
        conditions: OperationConditions,
        reference_state: ThermodynamicState) -> NDArray[float64]:
    """Пересчитывает объёмные расходы для заданного опорного состояния."""
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


def flow_based_water_cut(conditions: OperationConditions) -> float:
    """Обводнённость по объёмным расходам фаз, д.е.

    w = Q_в / (Q_в + Q_н)

    где Q_в, Q_н — объёмные расходы воды и нефти, м³/с.
    """
    cut = conditions.vol_flow_rate[W]/(
        conditions.vol_flow_rate[W]+conditions.vol_flow_rate[O])

    return cut


def flow_velocity(conditions: OperationConditions,
                  effective_area: float | NDArray[float64]) -> NDArray[float64]:
    """Скорости фаз в поперечном сечении, м/с.

    u = Q / F

    где Q — объёмный расход фазы, м³/с, F — площадь сечения для фазы, м².
    """
    return conditions.vol_flow_rate / effective_area


@dataclass
class CoalescerPacking:
    coalescer_top_gap: float     # расстояние между пластинами в верхнем коалесцере, м
    coalescer_bottom_gap: float  # расстояние между пластинами в нижнем коалесцере, м
    angle: float = 45.0          # угол наклона пластин, градусы


@dataclass
class GeometryCyclone:
    width_inlet_cyclone: float   # м - ширина входа в циклон
    height_inlet_cyclone: float  # м - высота входа в циклон

    def __post_init__(self):
        """Площадь сечения спирального канала одного циклона, м².

        F_кан = b * h

        где b — ширина входа, h — высота входа в циклон.
        """
        self.area_spiral_channel = self.width_inlet_cyclone * self.height_inlet_cyclone


@dataclass
class SeparatorDesign:
    inner_diameter: float           # внутренний диаметр
    length_cylindrical_part: float  # длина цилиндрической части сепаратора
    length_semiaxis: float          # длина полуоси эллиптического днища, м
    length_first_section: float     # длина цилиндрической части первой секции
    length_second_section: float    # длина секции после перегородки
    length_to_baffle: float  # расстояние от решетки до сливной перегородки

    def __post_init__(self):
        """Производные геометрические характеристики сепаратора.

        F_сеч = π * D² / 4  — площадь поперечного сечения, м².

        V_сек_1 = F_сеч * L_1  — объём первой секции, м³.

        V_сек_2 = F_сеч * L_2 + V_эллипт  — объём второй секции
        (с эллиптическим днищем), м³.
        """
        self.volume_ell_head = volume_cover_elliptic_trunc(
            self.length_semiaxis,
            self.inner_diameter,
            self.inner_diameter
        )
        self.volume_separator = volume_section_horiz_ellipses(
            length_semiaxis_left=self.length_semiaxis,
            length_cylinder=self.length_cylindrical_part,
            length_semiaxis_right=self.length_semiaxis,
            diameter=self.inner_diameter,
            level=self.inner_diameter
        )
        self.section_area = np.pi * self.inner_diameter ** 2 / 4
        self.volume = (
            self.section_area * self.length_first_section,
            self.section_area * self.length_second_section + self.volume_ell_head,
        )


# ============================================================
# Пример использования
# ============================================================

if __name__ == "__main__":
    from pytroleum.plant.tps.utils import (PA_TO_MPA, PERCENT, KG_S_TO_T_H,
                                           SECONDS_PER_DAY,
                                           KELVIN_TO_CELSIUS, _major_header)

    pressure = 2e6
    temperature = 273.15 + 50
    vol_flow_gas_norm = 300_000 / SECONDS_PER_DAY
    vol_flow_oil = 200 / SECONDS_PER_DAY
    vol_flow_water = 300 / SECONDS_PER_DAY

    oil_density = 933
    oil_viscosity = 3.073e-3

    conditions = OperationConditions()
    conditions.phase[OIL].change(oil_density, oil_viscosity)  # type: ignore
    conditions.phase[VAPOR].update(*STANDARD_STATE)
    gas_density_norm = conditions.phase[VAPOR].rhomass()

    conditions.update_state((CoolConst.PT_INPUTS, pressure, temperature),
                            upd_containers=True)

    mass_flow_gas = vol_flow_gas_norm * gas_density_norm
    vol_flow_gas_work = mass_flow_gas / conditions.phase[VAPOR].rhomass()

    conditions.vol_flow_rate = np.array(
        [vol_flow_gas_work, vol_flow_oil, vol_flow_water])
    conditions.mass_flow_rate = conditions.vol_flow_rate * np.array(
        [phase.rhomass() for phase in conditions.phase])

    mass_flow_oil = conditions.mass_flow_rate[OIL]
    mass_flow_water = conditions.mass_flow_rate[WATER]
    mass_flow_liquid = mass_flow_oil + mass_flow_water
    mass_flow_total = np.sum(conditions.mass_flow_rate)

    _major_header("УСЛОВИЯ РАБОТЫ")
    print(f"Давление при н.у.: {STANDARD_PRESSURE / PA_TO_MPA:.1f} МПа")
    print(f"Температура при н.у.: {STANDARD_TEMPERATURE} К")
    print(f"Рабочее давление: {pressure / PA_TO_MPA:.1f} МПа")
    print(f"Рабочая температура: {temperature} К "
          f"({temperature - KELVIN_TO_CELSIUS:.0f} °C)")
    print(f"Объемный расход газа при н.у.: "
          f"{vol_flow_gas_norm * SECONDS_PER_DAY:,.0f} м³/сут".replace(",", " "))

    vol_flow_liquid = conditions.vol_flow_rate[OIL] + \
        conditions.vol_flow_rate[WATER]
    print(f"Объемный расход жидкости: "
          f"{vol_flow_liquid * SECONDS_PER_DAY:.0f} м³/сут")

    _major_header("СВОЙСТВА ФЛЮИДА")
    print(f"Плотность газа при н.у.: {gas_density_norm:.3f} кг/м³")
    print(f"Плотность нефти: {conditions.phase[OIL].rhomass():.0f} кг/м³")
    print(f"Плотность воды: {conditions.phase[WATER].rhomass():.0f} кг/м³")
    print(f"Обводненность: {flow_based_water_cut(conditions) * PERCENT:.0f}%")

    _major_header("ОБЪЕМНЫЕ РАСХОДЫ")
    print(f"Объемный расход газа при р.у.: "
          f"{conditions.vol_flow_rate[VAPOR] * SECONDS_PER_DAY:.0f} м³/сут")
    print(f"Объемный расход по нефти: "
          f"{conditions.vol_flow_rate[OIL] * SECONDS_PER_DAY:.0f} м³/сут")
    print(f"Объемный расход по воде: "
          f"{conditions.vol_flow_rate[WATER] * SECONDS_PER_DAY:.0f} м³/сут")

    _major_header("МАССОВЫЕ РАСХОДЫ (кг/с)")
    print(f"Массовый расход газа: {mass_flow_gas:.2f} кг/с")
    print(f"Массовый расход нефти: {mass_flow_oil:.2f} кг/с")
    print(f"Массовый расход воды: {mass_flow_water:.2f} кг/с")
    print(f"Массовый расход жидкости (Н+В): {mass_flow_liquid:.2f} кг/с")
    print(f"Массовый суммарный расход (Г+Н+В): {mass_flow_total:.2f} кг/с")

    _major_header("МАССОВЫЕ РАСХОДЫ (т/ч)")
    print(f"Массовый расход газа: {mass_flow_gas * KG_S_TO_T_H:.2f} т/ч")
    print(f"Массовый расход нефти: {mass_flow_oil * KG_S_TO_T_H:.2f} т/ч")
    print(f"Массовый расход воды: {mass_flow_water * KG_S_TO_T_H:.2f} т/ч")
    print(
        f"Массовый расход жидкости (Н+В): {mass_flow_liquid * KG_S_TO_T_H:.2f} т/ч")
    print(
        f"Массовый суммарный расход (Г+Н+В): {mass_flow_total * KG_S_TO_T_H:.2f} т/ч")

    _major_header("ФИЗИЧЕСКИЕ СВОЙСТВА ПРИ РАБОЧИХ УСЛОВИЯХ")
    print(
        f"Плотность газа в р.у.: {conditions.phase[VAPOR].rhomass():.3f} кг/м³")
    liquid_density = ((1 - flow_based_water_cut(conditions)) *
                      conditions.phase[OIL].rhomass() +
                      flow_based_water_cut(conditions) *
                      conditions.phase[WATER].rhomass())
    print(f"Плотность жидкости (Н+В) при заданной обводненности: "
          f"{liquid_density:.1f} кг/м³")
