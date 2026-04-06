"""Отчет по трехфазному сепаратору"""

if __name__ == '__main__':
    from pytroleum.plant.tps.inputs import (
        OperationConditions, PhysicalProperties, FlowRates,
        DEFAULT_PRESSURE, DEFAULT_TEMPERATURE,
        SECONDS_PER_DAY, PA_TO_MPA, PERCENT, KG_S_TO_T_H,
    )
    from pytroleum.plant.tps.nozzle import (
        GasNozzle, OilNozzle, WaterNozzle, LiquidNozzle, LiquidGasNozzle,
        _TO_MM,
    )

    from pytroleum.plant.tps.utils import _major_header, _minor_header

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

    _major_header("УСЛОВИЯ РАБОТЫ")
    print(f"Давление при н.у.: {DEFAULT_PRESSURE / PA_TO_MPA:.1f} МПа")
    print(f"Температура при н.у.: {DEFAULT_TEMPERATURE} К")
    print(f"Рабочее давление: {con.pressure_work / PA_TO_MPA:.1f} МПа")
    print(f"Рабочая температура: {con.temperature_work} К")
    print(f"Объемный расход газа при н.у.: "
          f"{con.flow_gas_norm * SECONDS_PER_DAY:.0f} м3/сут")
    print(f"Объемный расход жидкости: "
          f"{con.flow_liquid * SECONDS_PER_DAY:.0f} м3/сут")

    _major_header("СВОЙСТВА ФЛЮИДА")
    print(f"Плотность газа при н.у.: {props.gas_density_norm} кг/м3")
    print(f"Плотность нефти: {props.oil_density} кг/м3")
    print(f"Плотность воды: {props.water_density} кг/м3")
    print(f"Обводненность: {props.water_cut * PERCENT:.0f}%")
    print(f"Газовый фактор: {props.gas_factor} м3/т")

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

    _major_header("ФИЗИЧЕСКИЕ СВОЙСТВА ПРИ РАБОЧИХ УСЛОВИЯХ")
    print(f"Плотность газа в р.у.: {props.gas_density_work(con):.3f} кг/м³")
    print(f"Плотность жидкости (Н+В) при заданной обводненности: "
          f"{props.liquid_density():.1f} кг/м³")
    print(f"Производительность по газу из условия газового фактора: "
          f"{flows.flow_gas_from_gas_factor():.1f} м³/сут")

    gas_nozzle = GasNozzle(flows=flows, recommended_speed=10.0)
    oil_nozzle = OilNozzle(flows=flows, recommended_speed=1.0)
    water_nozzle = WaterNozzle(flows=flows, recommended_speed=1.0)
    liquid_nozzle = LiquidNozzle(flows=flows, recommended_speed=1.0)
    liquid_gas_nozzle = LiquidGasNozzle(
        flows=flows, gas_speed=10.0, liquid_speed=1.0)

    _major_header("РАСЧЕТ ШТУЦЕРОВ")

    _minor_header(gas_nozzle.name)
    print(f"  Рекомендуемая скорость:  {gas_nozzle.recommended_speed:.2f} м/с")
    print(
        f"  Расчетный диаметр:       {gas_nozzle.calculate_diameter() * _TO_MM:.1f} мм")
    print(f"  Стандартный диаметр:     "
          f"{gas_nozzle.select_nominal_diameter() * _TO_MM:.0f} мм")
    print(f"  Площадь сечения штуцера: {gas_nozzle.nozzle_area():.4f} м²")
    print(f"  Фактическая скорость:    {gas_nozzle.actual_speed():.4f} м/с")

    print()
    _minor_header(oil_nozzle.name)
    print(f"  Рекомендуемая скорость:  {oil_nozzle.recommended_speed:.2f} м/с")
    print(
        f"  Расчетный диаметр:       {oil_nozzle.calculate_diameter() * _TO_MM:.1f} мм")
    print(f"  Стандартный диаметр:     "
          f"{oil_nozzle.select_nominal_diameter() * _TO_MM:.0f} мм")
    print(f"  Площадь сечения штуцера: {oil_nozzle.nozzle_area():.4f} м²")
    print(f"  Фактическая скорость:    {oil_nozzle.actual_speed():.4f} м/с")

    print()
    _minor_header(water_nozzle.name)
    print(
        f"  Рекомендуемая скорость:  {water_nozzle.recommended_speed:.2f} м/с")
    print(
        f"  Расчетный диаметр:       {water_nozzle.calculate_diameter() * _TO_MM:.1f} мм")
    print(f"  Стандартный диаметр:     "
          f"{water_nozzle.select_nominal_diameter() * _TO_MM:.0f} мм")
    print(f"  Площадь сечения штуцера: {water_nozzle.nozzle_area():.4f} м²")
    print(f"  Фактическая скорость:    {water_nozzle.actual_speed():.4f} м/с")

    print()
    _minor_header(liquid_nozzle.name)
    print(
        f"  Рекомендуемая скорость:  {liquid_nozzle.recommended_speed:.2f} м/с")
    print(f"  Расчетный диаметр:       "
          f"{liquid_nozzle.calculate_diameter() * _TO_MM:.1f} мм")
    print(f"  Стандартный диаметр:     "
          f"{liquid_nozzle.select_nominal_diameter() * _TO_MM:.0f} мм")
    print(f"  Площадь сечения штуцера: {liquid_nozzle.nozzle_area():.4f} м²")
    print(f"  Фактическая скорость:    {liquid_nozzle.actual_speed():.4f} м/с")

    print()
    _minor_header(liquid_gas_nozzle.name)
    print(
        f"  Рекомендуемая скорость газа:     {liquid_gas_nozzle.gas_speed:.2f} м/с")
    print(
        f"  Рекомендуемая скорость жидкости: {liquid_gas_nozzle.liquid_speed:.2f} м/с")
    print(f"  Расчетный диаметр:               "
          f"{liquid_gas_nozzle.calculate_diameter() * _TO_MM:.1f} мм")
    print(f"  Стандартный диаметр:             "
          f"{liquid_gas_nozzle.select_nominal_diameter() * _TO_MM:.0f} мм")
    print(
        f"  Площадь сечения штуцера:         {liquid_gas_nozzle.nozzle_area():.4f} м²")
    print(
        f"  Фактическая скорость:            {liquid_gas_nozzle.actual_speed():.4f} м/с")
