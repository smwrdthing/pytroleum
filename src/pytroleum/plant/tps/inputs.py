from dataclasses import dataclass
from pytroleum.plant.tps.report import _minor_divider, _major_divider

DEFAULT_PRESSURE = 0.1e6      # Па
DEFAULT_TEMPERATURE = 293     # К (20°C)
SECONDS_PER_DAY = 86400
KG_PER_TON = 1000
KG_S_TO_T_H = 3.6
PERCENT = 100
PA_TO_MPA = 1e6


@dataclass
class OperationConditions:
    pressure_work: float        # Па - рабочее давление
    temperature_work: float     # К - рабочая температура
    flow_gas_norm: float        # м³/с - объемный расход газа при н.у.
    flow_liquid: float          # м³/с - объемный расход жидкости


@dataclass
class PhysicalProperties:
    gas_density_norm: float     # кг/м³ - плотность газа
    oil_density: float          # кг/м³ - плотность нефти
    water_density: float        # кг/м³ - плотность воды
    water_cut: float            # Обводненность
    gas_factor: float           # Газовый фактор, м³/т

    def liquid_density(self) -> float:
        """Плотность жидкости (Н+В) при заданной обводненности"""
        return (self.water_density * self.water_cut +
                self.oil_density * (1 - self.water_cut))

    def gas_density_work(self, conditions: OperationConditions) -> float:
        """Плотность газа в рабочих условиях"""
        return self.gas_density_norm * (conditions.pressure_work / DEFAULT_PRESSURE) * \
            (DEFAULT_TEMPERATURE / conditions.temperature_work)


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


# ==================== ВЫВОД РЕЗУЛЬТАТОВ ====================
if __name__ == '__main__':
    con = OperationConditions(
        pressure_work=1e6,
        temperature_work=353,
        flow_gas_norm=300000 / SECONDS_PER_DAY,
        flow_liquid=500 / SECONDS_PER_DAY,
    )
    props = PhysicalProperties(
        gas_density_norm=0.94,
        oil_density=933,
        water_density=966,
        water_cut=0.6,
        gas_factor=267.9,
    )
    flows = FlowRates(conditions=con, properties=props)

    _major_divider()
    print("УСЛОВИЯ РАБОТЫ")
    _major_divider()

    print(f"Давление при н.у.: {DEFAULT_PRESSURE / PA_TO_MPA:.1f} МПа")
    print(f"Температура при н.у.: {DEFAULT_TEMPERATURE} К")
    print(f"Рабочее давление: {con.pressure_work / PA_TO_MPA:.1f} МПа")
    print(f"Рабочая температура: {con.temperature_work} К")
    print(f"Объемный расход газа при н.у.: "
          f"{con.flow_gas_norm * SECONDS_PER_DAY:.0f} м3/сут")
    print(f"Объемный расход жидкости: "
          f"{con.flow_liquid * SECONDS_PER_DAY:.0f} м3/сут")

    _major_divider()
    print("СВОЙСТВА ФЛЮИДА")
    _major_divider()

    print(f"Плотность газа при н.у.: {props.gas_density_norm} кг/м3")
    print(f"Плотность нефти: {props.oil_density} кг/м3")
    print(f"Плотность воды: {props.water_density} кг/м3")
    print(f"Обводненность: {props.water_cut * PERCENT:.0f}%")
    print(f"Газовый фактор: {props.gas_factor} м3/т")

    _major_divider()
    print("ОБЪЕМНЫЕ РАСХОДЫ")
    _major_divider()

    print(f"Объемный расход газа при р.у.: "
          f"{flows.flow_gas_work() * SECONDS_PER_DAY:.0f} м³/сут")
    print(f"Объемный расход по нефти: "
          f"{flows.flow_oil() * SECONDS_PER_DAY:.0f} м³/сут")
    print(f"Объемный расход по воде: "
          f"{flows.flow_water() * SECONDS_PER_DAY:.0f} м³/сут")

    _major_divider()
    print("МАССОВЫЕ РАСХОДЫ (кг/с)")
    _major_divider()

    print(f"Массовый расход газа: {flows.mass_flow_gas():.2f} кг/с")
    print(f"Массовый расход нефти: {flows.mass_flow_oil():.2f} кг/с")
    print(f"Массовый расход воды: {flows.mass_flow_water():.2f} кг/с")
    print(f"Массовый расход жидкости (Н+В): "
          f"{flows.mass_flow_liquid():.2f} кг/с")
    print(f"Массовый суммарный расход по продукту (Г+Н+В): "
          f"{flows.mass_flow_total():.2f} кг/с")

    _major_divider()
    print("МАССОВЫЕ РАСХОДЫ (т/ч)")
    _major_divider()

    print(
        f"Массовый расход газа: {flows.mass_flow_gas() * KG_S_TO_T_H:.2f} т/ч")
    print(
        f"Массовый расход нефти: {flows.mass_flow_oil() * KG_S_TO_T_H:.2f} т/ч")
    print(
        f"Массовый расход воды: {flows.mass_flow_water() * KG_S_TO_T_H:.2f} т/ч")
    print(f"Массовый расход жидкости (Н+В): "
          f"{flows.mass_flow_liquid() * KG_S_TO_T_H:.2f} т/ч")
    print(f"Массовый суммарный расход по продукту (Г+Н+В): "
          f"{flows.mass_flow_total() * KG_S_TO_T_H:.2f} т/ч")

    _major_divider()
    print("РАСЧЕТ ФИЗИЧЕСКИХ СВОЙСТВ (Г, Н, В) ПРИ РАБОЧИХ УСЛОВИЯХ")
    _major_divider()

    print(f"Плотность газа в р.у.: {props.gas_density_work(con):.3f} кг/м³")
    print(f"Плотность жидкости (Н+В) при заданной обводненности: "
          f"{props.liquid_density():.1f} кг/м³")
    print(f"Производительность по газу из условия газового фактора:"
          f"{flows.flow_gas_from_gas_factor():.1f} м³/сут")
