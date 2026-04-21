from dataclasses import dataclass
import numpy as np
from pytroleum.plant.tps.utils import (DEFAULT_PRESSURE,
                                       DEFAULT_TEMPERATURE,
                                       SECONDS_PER_DAY,
                                       KG_PER_TON,
                                       KELVIN_TO_CELSIUS)
from pytroleum.meter import volume_cover_elliptic_trunc, volume_section_horiz_ellipses


@dataclass
class OperationConditions:
    pressure_work: float        # Па - рабочее давление
    temperature_work: float     # К - рабочая температура
    flow_gas_norm: float        # м³/с - объемный расход газа при н.у.
    flow_liquid: float          # м³/с - объемный расход жидкости

    # NOTE _work постфиксы избыточны, т.к. они не добавляют контекста
    # NOTE и просто удлинняют идентификатор


@dataclass
class PhysicalProperties:
    gas_density_norm: float     # кг/м³ - плотность газа при н.у.
    oil_density: float          # кг/м³ - плотность нефти
    water_density: float        # кг/м³ - плотность воды
    water_cut: float            # Обводненность, д.ед.
    gas_factor: float           # Газовый фактор, м³/т
    oil_surface_tension: float  # Поверхностное натяжение нефти, Н/м
    viscosity_oil: float        # Вязкость нефти, Па*с
    viscosity_water: float      # Вязкость воды, Па*с

    # NOTE здесь можно использовать то, что мы делали в tdyna

    def liquid_density(self) -> float:
        """Плотность жидкости (Н+В) при заданной обводнённости, кг/м³.

        ρ_ж = ρ_в * w + ρ_н * (1 - w)

        где ρ_в, ρ_н — плотности воды и нефти, w — обводнённость (д.ед.).
        """
        return (self.water_density * self.water_cut +
                self.oil_density * (1 - self.water_cut))

    def gas_density_work(self, conditions: OperationConditions) -> float:
        """Плотность газа при рабочих условиях, кг/м³.

        ρг_ру = ρг_ну * (P_ру / P_ну) * (T_ну / T_ру)

        где ρ_ну — плотность газа при н.у., P_ру, T_ру — рабочие давление и температура,
        P_ну, T_ну — давление и температура при нормальных условиях.
        """
        return self.gas_density_norm * (conditions.pressure_work / DEFAULT_PRESSURE) * \
            (DEFAULT_TEMPERATURE / conditions.temperature_work)


@dataclass
class CoalescerPacking:
    coalescer_top_gap: float     # расстояние между пластинами в верхнем коалесцере, м
    coalescer_bottom_gap: float  # расстояние между пластинами в нижнем коалесцере, м
    angle: float = 45.0          # угол наклона пластин, градусы


@dataclass
class GeometryCyclone:
    width_inlet_cyclone: float   # м - ширина входа в циклон
    height_inlet_cyclone: float  # м - высота входа в циклон
    number_of_cyclones: float    # Количество циклонов

    # NOTE количество циклонов - не часть информации о геометрии циклона
    # NOTE в классе лучше описывать одну единицу оборудования, если расчёт требует
    # NOTE количества - его можно передавать явно как параметр функции по месту

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
    L_c: float                      # расстояние от решетки до сливной перегородки

    # NOTE L_c лучше переименовать во что-то более явное
    # NOTE length_to_baffle, например

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


@dataclass
class FlowRates:
    """Расходы флюидов, вычисленные из рабочих условий и физических свойств.

    Q_н = Q_ж * (1 - w),  Q_в = Q_ж * w  — объёмные расходы нефти и воды, м³/с.

    Q_г_ру = Q_г_ну * P_ну * T_ру / ((P_ру + P_ну) * T_ну)  — расход газа при р.у., м³/с.

    G_г = Q_г_ну * ρ_г_ну,  G_н = Q_н * ρ_н,  G_в = Q_в * ρ_в  — массовые расходы, кг/с.
    """
    conditions: OperationConditions
    properties: PhysicalProperties

    # NOTE расходы жидкостей - часть данных о рабочем режиме, их можно перенести
    # NOTE в OperationCondtions

    def __post_init__(self):
        self.flow_oil = (self.conditions.flow_liquid *
                         (1 - self.properties.water_cut))

        self.flow_water = self.conditions.flow_liquid * self.properties.water_cut

        self.flow_gas_work = ((self.conditions.flow_gas_norm * DEFAULT_PRESSURE *
                               self.conditions.temperature_work) /
                              ((self.conditions.pressure_work + DEFAULT_PRESSURE) *
                               DEFAULT_TEMPERATURE))

        self.mass_flow_gas = (self.conditions.flow_gas_norm *
                              self.properties.gas_density_norm)

        self.mass_flow_oil = self.flow_oil * self.properties.oil_density

        self.mass_flow_water = self.flow_water * self.properties.water_density

        self.mass_flow_liquid = self.mass_flow_oil + self.mass_flow_water

        self.mass_flow_total = (self.mass_flow_gas + self.mass_flow_oil +
                                self.mass_flow_water)

        self.mass_flow_oil_ton_per_day = self.mass_flow_oil * SECONDS_PER_DAY / KG_PER_TON

        self.flow_gas_from_gas_factor = (self.properties.gas_factor *
                                         self.mass_flow_oil_ton_per_day)

        self.flow_rate = (self.flow_gas_work, self.flow_oil, self.flow_water)
        self.velocity: list[float] = [0.0, 0.0, 0.0]

        # NOTE расходы по фазам в контейнеры, индексация по константам
        # NOTE
        # NOTE self.mass_flow_rate[VAPOR] <- для газовой фазы
        # NOTE self.mass_flow_rate[WATER] <- для воды
        # NOTE self.mass_flow_rate[OIL] <- для нефти
        # NOTE
        # NOTE меньше атрибутов в классе => удобнее использовать
        # NOTE
        # NOTE Так, например, можно уйти от хранения суммарного расхода
        # NOTE и просто пользоваться np.sum() где нужно (или даже просто sum())


# ============================================================
# Пример использования
# ============================================================

if __name__ == "__main__":
    from pytroleum.plant.tps.utils import (PA_TO_MPA,
                                           PERCENT,
                                           KG_S_TO_T_H,
                                           _major_header)

    conditions = OperationConditions(
        pressure_work=4e6,
        temperature_work=353,
        flow_gas_norm=300000 / SECONDS_PER_DAY,
        flow_liquid=500 / SECONDS_PER_DAY,
    )
    properties = PhysicalProperties(
        gas_density_norm=0.94,
        oil_density=933,
        water_density=966,
        water_cut=0.6,
        gas_factor=267.9,
        oil_surface_tension=0.02848,
        viscosity_oil=3.073e-3,
        viscosity_water=0.544e-3
    )
    flows = FlowRates(conditions=conditions, properties=properties)

    _major_header("УСЛОВИЯ РАБОТЫ")
    print(f"Давление при н.у.: {DEFAULT_PRESSURE / PA_TO_MPA:.1f} МПа")
    print(f"Температура при н.у.: {DEFAULT_TEMPERATURE} К")
    print(f"Рабочее давление: {conditions.pressure_work / PA_TO_MPA:.1f} МПа")
    print(f"Рабочая температура: {conditions.temperature_work} К "
          f"({conditions.temperature_work - KELVIN_TO_CELSIUS} °C)")
    print(f"Объемный расход газа при н.у.: "
          f"{conditions.flow_gas_norm * SECONDS_PER_DAY:,.0f} м³/сут".replace(",", " "))
    print(f"Объемный расход жидкости: "
          f"{conditions.flow_liquid * SECONDS_PER_DAY:.0f} м3/сут")

    _major_header("СВОЙСТВА ФЛЮИДА")
    print(f"Плотность газа при н.у.: {properties.gas_density_norm} кг/м3")
    print(f"Плотность нефти: {properties.oil_density} кг/м3")
    print(f"Плотность воды: {properties.water_density} кг/м3")
    print(f"Обводненность: {properties.water_cut * PERCENT:.0f}%")
    print(f"Газовый фактор: {properties.gas_factor} м3/т")

    _major_header("ОБЪЕМНЫЕ РАСХОДЫ")
    print(f"Объемный расход газа при р.у.: "
          f"{flows.flow_gas_work * SECONDS_PER_DAY:.0f} м³/сут")
    print(f"Объемный расход по нефти: "
          f"{flows.flow_oil * SECONDS_PER_DAY:.0f} м³/сут")
    print(f"Объемный расход по воде: "
          f"{flows.flow_water * SECONDS_PER_DAY:.0f} м³/сут")

    _major_header("МАССОВЫЕ РАСХОДЫ (кг/с)")
    print(f"Массовый расход газа: {flows.mass_flow_gas:.2f} кг/с")
    print(f"Массовый расход нефти: {flows.mass_flow_oil:.2f} кг/с")
    print(f"Массовый расход воды: {flows.mass_flow_water:.2f} кг/с")
    print(f"Массовый расход жидкости (Н+В): "
          f"{flows.mass_flow_liquid:.2f} кг/с")
    print(f"Массовый суммарный расход по продукту (Г+Н+В): "
          f"{flows.mass_flow_total:.2f} кг/с")

    _major_header("МАССОВЫЕ РАСХОДЫ (т/ч)")
    print(f"Массовый расход газа: "
          f"{flows.mass_flow_gas * KG_S_TO_T_H:.2f} т/ч")
    print(f"Массовый расход нефти: "
          f"{flows.mass_flow_oil * KG_S_TO_T_H:.2f} т/ч")
    print(f"Массовый расход воды: "
          f"{flows.mass_flow_water * KG_S_TO_T_H:.2f} т/ч")
    print(f"Массовый расход жидкости (Н+В): "
          f"{flows.mass_flow_liquid * KG_S_TO_T_H:.2f} т/ч")
    print(f"Массовый суммарный расход по продукту (Г+Н+В): "
          f"{flows.mass_flow_total * KG_S_TO_T_H:.2f} т/ч")

    _major_header("ФИЗИЧЕСКИЕ СВОЙСТВА ПРИ РАБОЧИХ УСЛОВИЯХ")
    print(f"Плотность газа в р.у.: "
          f"{properties.gas_density_work(conditions):.3f} кг/м³")
    print(f"Плотность жидкости (Н+В) при заданной обводненности: "
          f"{properties.liquid_density():.1f} кг/м³")
    print(f"Производительность по газу из условия газового фактора: "
          f"{flows.flow_gas_from_gas_factor:.1f} м³/сут")
