pressure_norm = 0.1e6      # Па (0.1 МПа)
temperature_norm = 293     # К (20°C)

# Рабочие условия
pressure_work = 1e6        # Па (1.0 МПа) - рабочее давление
temperature_work = 353     # К (80°C) - рабочая температура

# Плотности компонентов
gas_density_norm = 0.94       # кг/м³ - плотность газа
oil_density = 933          # кг/м³ - плотность нефти
water_density = 966        # кг/м³ - плотность воды

# Обводненность
water_cut = 0.6            # 60%

# Газовый фактор
gas_factor = 267.9         # м³/т

# Расходы (перевод из м³/сут в м³/с)
flow_gas_norm = 300000 / 86400   # м³/с - расход газа
flow_liquid = 500 / 86400        # м³/с - расход жидкости (нефть+вода)

# Расходы компонентов жидкости
flow_oil = flow_liquid * (1 - water_cut)     # м³/с - расход нефти
flow_water = flow_liquid * water_cut         # м³/с - расход воды

# Расход газа при рабочих условиях
flow_gas_work = (flow_gas_norm*pressure_norm*temperature_work) / \
    ((pressure_work+pressure_norm)*temperature_norm)

# ==================== МАССОВЫЕ РАСХОДЫ ====================

# Массовый расход каждого компонента (кг/с)
mass_flow_gas = flow_gas_norm * gas_density_norm      # газ, кг/с
mass_flow_oil = flow_oil * oil_density                # нефть, кг/с
mass_flow_water = flow_water * water_density          # вода, кг/с

# Массовый расход жидкости (нефть + вода) в кг/с
mass_flow_liquid = mass_flow_oil + mass_flow_water

# Суммарный массовый расход (Г+Н+В) в кг/с
mass_flow_total = mass_flow_gas + mass_flow_oil + mass_flow_water

# ==================== РАСЧЕТ ФИЗИЧЕСКИХ СВОЙСТВ ====================

# Плотность газа в рабочих условиях
gas_density_work = gas_density_norm * \
    (pressure_work / pressure_norm) * (temperature_norm / temperature_work)

# Плотность жидкости (Н+В) при заданной обводненности
liquid_density = water_density * water_cut + oil_density * (1 - water_cut)

# Производительность по газу из условия газового фактора
mass_flow_oil_ton_per_day = mass_flow_oil * 86400 / 1000  # т/сут
flow_gas_from_gas_factor = gas_factor * mass_flow_oil_ton_per_day  # м³/сут

# ==================== ВЫВОД РЕЗУЛЬТАТОВ ====================


print("=" * 60)
print("ОБЪЕМНЫЕ РАСХОДЫ")
print("=" * 60)
print(f"Объемный расход газа при н.у.: {flow_gas_norm * 86400:.0f} м³/сут")
print(f"Объемный расход газа при р.у.: {flow_gas_work * 86400:.0f} м³/сут")
print(f"Объемный расход по нефти: {flow_oil * 86400:.0f} м³/сут")
print(f"Объемный расход по воде: {flow_water * 86400:.0f} м³/сут")
print(f"Объемный расход жидкости: {flow_liquid * 86400:.0f} м³/сут")

print("\n" + "=" * 60)
print("МАССОВЫЕ РАСХОДЫ (кг/с)")
print("=" * 60)
print(f"Массовый расход газа: {mass_flow_gas:.2f} кг/с")
print(f"Массовый расход нефти: {mass_flow_oil:.2f} кг/с")
print(f"Массовый расход воды: {mass_flow_water:.2f} кг/с")
print(f"Массовый расход жидкости (Н+В): {mass_flow_liquid:.2f} кг/с")
print(
    f"Массовый суммарный расход по продукту (Г+Н+В): {mass_flow_total:.2f} кг/с")

print("\n" + "=" * 60)
print("МАССОВЫЕ РАСХОДЫ (т/ч)")
print("=" * 60)
print(f"Массовый расход газа: {mass_flow_gas * 3.6:.2f} т/ч")
print(f"Массовый расход нефти: {mass_flow_oil * 3.6:.2f} т/ч")
print(f"Массовый расход воды: {mass_flow_water * 3.6:.2f} т/ч")
print(f"Массовый расход жидкости (Н+В): {mass_flow_liquid * 3.6:.2f} т/ч")
print(
    f"Массовый суммарный расход по продукту (Г+Н+В): {mass_flow_total * 3.6:.2f} т/ч")

print("\n" + "=" * 60)
print("РАСЧЕТ ФИЗИЧЕСКИХ СВОЙСТВ (Г, Н, В) ПРИ РАБОЧИХ УСЛОВИЯХ")
print("=" * 60)
print(f"Плотность газа в р.у.: {gas_density_work:.3f} кг/м³")
print(
    f"Плотность жидкости (Н+В) при заданной обводненности: {liquid_density:.1f} кг/м³")
print(f"Производительность по газу из условия газового фактора:"
      f"{flow_gas_from_gas_factor:.1f} м³/сут")
