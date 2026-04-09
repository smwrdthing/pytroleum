from dataclasses import dataclass
from pytroleum.plant.tps.utils import (DEFAULT_PRESSURE,
                                       DEFAULT_TEMPERATURE,
                                       SECONDS_PER_DAY,
                                       KG_PER_TON,
                                       KELVIN_TO_CELSIUS)


@dataclass
class OperationConditions:
    pressure_work: float        # Па - рабочее давление
    temperature_work: float     # К - рабочая температура
    flow_gas_norm: float        # м³/с - объемный расход газа при н.у.
    flow_liquid: float          # м³/с - объемный расход жидкости


@dataclass
class PhysicalProperties:
    gas_density_norm: float     # кг/м³ - плотность газа при н.у.
    oil_density: float          # кг/м³ - плотность нефти
    water_density: float        # кг/м³ - плотность воды
    water_cut: float            # Обводненность
    gas_factor: float           # Газовый фактор, м³/т
    oil_surface_tension: float  # Поверхностное натяжение нефти, Н/м

    def liquid_density(self) -> float:
        """Плотность жидкости (Н+В) при заданной обводненности"""
        return (self.water_density * self.water_cut +
                self.oil_density * (1 - self.water_cut))

    def gas_density_work(self, conditions: OperationConditions) -> float:
        """Плотность газа в рабочих условиях"""
        return self.gas_density_norm * (conditions.pressure_work / DEFAULT_PRESSURE) * \
            (DEFAULT_TEMPERATURE / conditions.temperature_work)


@dataclass
class GeometryCyclone:
    width_inlet_cyclone: float   # м - ширина входа в циклон
    height_inlet_cyclone: float  # м - высота входа в циклон
    number_of_cyclones: float    # Количество циклонов


@dataclass
class SeparatorParameters:
    inner_diameter: float  # внутренний диаметр
    length_cylindrical_part: float  # длина цилиндрической части сепаратора
    fill_coeff: float  # коэффициент заполнения сепаратора
    fill_coeff_first_section: float  # коэффициент заполнения первой секции
    fill_coeff_after_wall: float  # коэффициент заполнения после перегородки
    volume_ell_head: float  # внутренний объем эллиптического днища
    length_first_section: float  # длина цилиндрической части первой секции
    length_section_after_wall: float  # длина секции после перегородки


@dataclass
class Coefficients:
    # Коэффициент учитывающий снижение площади сечения элементами насадки
    area_reduction_coefficient: float
    # Коэффициент сопротивления сетчатого отбойника
    mesh_resistance_coefficient: float
    # Коэффициент сопротивления для входного патрубка
    inlet_resistance_coefficient: float
    # Коэффициент сопротивления для выходного патрубка
    outlet_resistance_coefficient: float
    # Коэффициент неучтенных потерь
    losses_unaccounted: float


@dataclass
class FlowRates:
    conditions: OperationConditions
    properties: PhysicalProperties

    def flow_oil(self) -> float:
        """ Расход нефти, м³/с """
        return self.conditions.flow_liquid * (1 - self.properties.water_cut)

    def flow_water(self) -> float:
        """ Расход воды, м³/с """
        return self.conditions.flow_liquid * self.properties.water_cut

    def flow_gas_work(self) -> float:
        """Расход газа при рабочих условиях"""
        return (self.conditions.flow_gas_norm*DEFAULT_PRESSURE *
                self.conditions.temperature_work) / \
            ((self.conditions.pressure_work+DEFAULT_PRESSURE) *
             DEFAULT_TEMPERATURE)

    def mass_flow_gas(self) -> float:
        """Массовый расход газа, кг/с"""
        return self.conditions.flow_gas_norm * self.properties.gas_density_norm

    def mass_flow_oil(self) -> float:
        """Массовый расход нефти, кг/с"""
        return self.flow_oil() * self.properties.oil_density

    def mass_flow_water(self) -> float:
        """Массовый расход воды, кг/с"""
        return self.flow_water() * self.properties.water_density

    def mass_flow_liquid(self) -> float:
        """Массовый расход жидкости (нефть + вода) в кг/с"""
        return self.mass_flow_oil() + self.mass_flow_water()

    def mass_flow_total(self) -> float:
        """Суммарный массовый расход (Г+Н+В) в кг/с"""
        return self.mass_flow_gas() + self.mass_flow_oil() + self.mass_flow_water()

    def mass_flow_oil_ton_per_day(self) -> float:
        """Массовый расход нефти, т/сут"""
        return self.mass_flow_oil() * SECONDS_PER_DAY / KG_PER_TON

    def flow_gas_from_gas_factor(self) -> float:
        """Объемный расход по газу из условия газового фактора, м³/сут"""
        return self.properties.gas_factor * self.mass_flow_oil_ton_per_day()

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
        oil_surface_tension=0.02848
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
          f"{flows.flow_gas_work() * SECONDS_PER_DAY:.0f} м³/сут")
    print(f"Объемный расход по нефти: "
          f"{flows.flow_oil() * SECONDS_PER_DAY:.0f} м³/сут")
    print(f"Объемный расход по воде: "
          f"{flows.flow_water() * SECONDS_PER_DAY:.0f} м³/сут")

    _major_header("МАССОВЫЕ РАСХОДЫ (кг/с)")
    print(f"Массовый расход газа: {flows.mass_flow_gas():.2f} кг/с")
    print(f"Массовый расход нефти: {flows.mass_flow_oil():.2f} кг/с")
    print(f"Массовый расход воды: {flows.mass_flow_water():.2f} кг/с")
    print(f"Массовый расход жидкости (Н+В): "
          f"{flows.mass_flow_liquid():.2f} кг/с")
    print(f"Массовый суммарный расход по продукту (Г+Н+В): "
          f"{flows.mass_flow_total():.2f} кг/с")

    _major_header("МАССОВЫЕ РАСХОДЫ (т/ч)")
    print(f"Массовый расход газа: "
          f"{flows.mass_flow_gas() * KG_S_TO_T_H:.2f} т/ч")
    print(f"Массовый расход нефти: "
          f"{flows.mass_flow_oil() * KG_S_TO_T_H:.2f} т/ч")
    print(f"Массовый расход воды: "
          f"{flows.mass_flow_water() * KG_S_TO_T_H:.2f} т/ч")
    print(f"Массовый расход жидкости (Н+В): "
          f"{flows.mass_flow_liquid() * KG_S_TO_T_H:.2f} т/ч")
    print(f"Массовый суммарный расход по продукту (Г+Н+В): "
          f"{flows.mass_flow_total() * KG_S_TO_T_H:.2f} т/ч")

    _major_header("ФИЗИЧЕСКИЕ СВОЙСТВА ПРИ РАБОЧИХ УСЛОВИЯХ")
    print(f"Плотность газа в р.у.: "
          f"{properties.gas_density_work(conditions):.3f} кг/м³")
    print(f"Плотность жидкости (Н+В) при заданной обводненности: "
          f"{properties.liquid_density():.1f} кг/м³")
    print(f"Производительность по газу из условия газового фактора: "
          f"{flows.flow_gas_from_gas_factor():.1f} м³/сут")
