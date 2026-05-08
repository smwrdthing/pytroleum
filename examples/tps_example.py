import numpy as np
import matplotlib.pyplot as plt
from CoolProp import constants as CoolConst

from pytroleum.plant.tps.inputs import (
    OperationConditions, SeparatorDesign, CoalescerPacking, GeometryCyclone,
    STANDARD_STATE, STANDARD_PRESSURE, STANDARD_TEMPERATURE,
    flow_based_water_cut,
    VAPOR, OIL, WATER,
)
from pytroleum.plant.tps.separator import (
    Separator, compute_settling_velocity,
    FILL_COEFFS, FIRST_SECTION, SECOND_SECTION, TOTAL,
)
from pytroleum.plant.tps.devices import Coalescer, Cyclone, required_length_for
from pytroleum.plant.tps.nozzle import design_nozzle, design_two_phase_nozzle
from pytroleum.plant.tps.wire_mesh_demister import (
    design_demister, get_flow_stability_coefficient,
    calculate_critical_velocity,
    _PRESSURE_PA, _FLOW_STABILITY_COEFFICIENT, _STABILITY_COEFFICIENT_INTERPOLATOR,
)
from pytroleum.plant.tps.utils import (
    _major_header, _minor_header, _minor_divider,
    SECONDS_PER_DAY, SECONDS_PER_HOUR, SECONDS_PER_MINUTE,
    PA_TO_MPA, PERCENT, KG_S_TO_T_H,
    _TO_MM, _TO_MICRON,
    KELVIN_TO_CELSIUS,
    print_row as p,
)


# ============================================================
# Исходные данные
# ============================================================

# Рабочие условия
pressure = 2e6             # Па
temperature = 273.15 + 50  # К

# Расходы
vol_flow_gas_norm = 300_000 / SECONDS_PER_DAY  # м³/с при н.у.
vol_flow_oil = 200 / SECONDS_PER_DAY           # м³/с
vol_flow_water = 300 / SECONDS_PER_DAY         # м³/с

# Свойства нефти
oil_density = 933          # кг/м³
oil_viscosity = 3.073e-3   # Па·с
oil_surface_tension = 0.02848  # Н/м

# Диаметры капель
diameter_water_droplet = 100e-6  # м
diameter_oil_droplet = 50e-6     # м

# Геометрия сепаратора
design = SeparatorDesign(
    inner_diameter=2.0,
    length_cylindrical_part=9.5,
    length_semiaxis=0.618,
    length_first_section=8.2,
    length_second_section=1.3,
    length_to_baffle=4.7,
)

# Коалесцер
packing = CoalescerPacking(coalescer_top_gap=15e-3,
                           coalescer_bottom_gap=25e-3)

# Циклон
number_of_cyclones = 4
geometry = GeometryCyclone(inlet_width=47.5e-3,
                           inlet_height=75e-3)

# Штуцера
gas_speed = 10.0    # м/с
liquid_speed = 1.0  # м/с

# ============================================================
# Термодинамические условия
# ============================================================

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

vol_flow_liquid = conditions.vol_flow_rate[OIL] + \
    conditions.vol_flow_rate[WATER]

water_cut = flow_based_water_cut(conditions)

liquid_density = (conditions.phase[OIL].rhomass() * (1 - water_cut) +
                  conditions.phase[WATER].rhomass() * water_cut)

rho_oil = conditions.phase[OIL].rhomass()
rho_water = conditions.phase[WATER].rhomass()
mu_oil = conditions.phase[OIL].viscosity()
mu_water = conditions.phase[WATER].viscosity()

# ============================================================
# Сепаратор
# ============================================================

separator = Separator(design=design, conditions=conditions)
separator.compute_velocities()
rt = separator.residence_time()
capacities = separator.capacity()
areas = separator.compute_flow_areas()

velocity_water = abs(compute_settling_velocity(
    diameter_water_droplet, rho_oil, mu_oil, rho_water))
height_water = separator.settling_height(
    diameter_water_droplet, rho_oil, mu_oil, rho_water, OIL)

velocity_oil = compute_settling_velocity(
    diameter_oil_droplet, rho_water, mu_water, rho_oil)
height_oil = separator.settling_height(
    diameter_oil_droplet, rho_water, mu_water, rho_oil, WATER)

# ============================================================
# Коалесцер и циклон
# ============================================================

coalescer = Coalescer(packing=packing)
cyclone = Cyclone(conditions=conditions, geometry=geometry)

t_top = coalescer.droplet_settling_time(
    packing.coalescer_top_gap,
    diameter_water_droplet,
    rho_oil, mu_oil, rho_water,
)
t_bottom = coalescer.droplet_settling_time(
    packing.coalescer_bottom_gap,
    diameter_oil_droplet,
    rho_water, mu_water, rho_oil,
)

# ============================================================
# Штуцера
# ============================================================

gas_nozzle = design_nozzle(conditions.vol_flow_rate[VAPOR], gas_speed)
oil_nozzle = design_nozzle(conditions.vol_flow_rate[OIL], liquid_speed)
water_nozzle = design_nozzle(conditions.vol_flow_rate[WATER], liquid_speed)
liquid_nozzle = design_nozzle(vol_flow_liquid, liquid_speed)
liquid_gas_nozzle = design_two_phase_nozzle(conditions=conditions,
                                            gas_speed=gas_speed,
                                            liquid_speed=liquid_speed)

# ============================================================
# Каплеуловитель
# ============================================================

current_flow_stability_coefficient = get_flow_stability_coefficient(pressure)
wmd = design_demister(conditions, oil_surface_tension)

# ============================================================
# Вывод результатов
# ============================================================

_major_header("УСЛОВИЯ РАБОТЫ")
p("Давление при н.у.:",
  f"{STANDARD_PRESSURE / PA_TO_MPA:.3f}", "МПа")
p("Температура при н.у.:", f"{STANDARD_TEMPERATURE}", "К")
p("Рабочее давление:", f"{pressure / PA_TO_MPA:.1f}", "МПа")
p("Рабочая температура:",
  f"{temperature} К ({temperature - KELVIN_TO_CELSIUS:.0f} °C)")
p("Объемный расход газа при н.у.:",
  f"{vol_flow_gas_norm * SECONDS_PER_DAY:,.0f}".replace(",", " "), "м³/сут")
p("Объемный расход жидкости:",
  f"{vol_flow_liquid * SECONDS_PER_DAY:.0f}", "м³/сут")

_major_header("СВОЙСТВА ФЛЮИДА")
p("Плотность газа при н.у.:", f"{gas_density_norm:.3f}", "кг/м³")
p("Плотность нефти:", f"{rho_oil:.0f}", "кг/м³")
p("Плотность воды:", f"{rho_water:.0f}", "кг/м³")
p("Обводненность:", f"{water_cut * PERCENT:.0f}", "%")

_major_header("ОБЪЕМНЫЕ РАСХОДЫ")
p("Объемный расход газа при р.у.:",
  f"{conditions.vol_flow_rate[VAPOR] * SECONDS_PER_DAY:.0f}", "м³/сут")
p("Объемный расход по нефти:",
  f"{conditions.vol_flow_rate[OIL] * SECONDS_PER_DAY:.0f}", "м³/сут")
p("Объемный расход по воде:",
  f"{conditions.vol_flow_rate[WATER] * SECONDS_PER_DAY:.0f}", "м³/сут")

_major_header("МАССОВЫЕ РАСХОДЫ (кг/с)")
p("Массовый расход газа:", f"{mass_flow_gas:.2f}", "кг/с")
p("Массовый расход нефти:", f"{mass_flow_oil:.2f}", "кг/с")
p("Массовый расход воды:", f"{mass_flow_water:.2f}", "кг/с")
p("Массовый расход жидкости (Н+В):", f"{mass_flow_liquid:.2f}", "кг/с")
p("Массовый суммарный расход (Г+Н+В):", f"{mass_flow_total:.2f}", "кг/с")

_major_header("МАССОВЫЕ РАСХОДЫ (т/ч)")
p("Массовый расход газа:",
  f"{mass_flow_gas * KG_S_TO_T_H:.2f}", "т/ч")
p("Массовый расход нефти:",
  f"{mass_flow_oil * KG_S_TO_T_H:.2f}", "т/ч")
p("Массовый расход воды:",
  f"{mass_flow_water * KG_S_TO_T_H:.2f}", "т/ч")
p("Массовый расход жидкости (Н+В):",
  f"{mass_flow_liquid * KG_S_TO_T_H:.2f}", "т/ч")
p("Массовый суммарный расход (Г+Н+В):",
  f"{mass_flow_total * KG_S_TO_T_H:.2f}", "т/ч")

_major_header("ФИЗИЧЕСКИЕ СВОЙСТВА ПРИ РАБОЧИХ УСЛОВИЯХ")
p("Плотность газа в р.у.:",
  f"{conditions.phase[VAPOR].rhomass():.3f}", "кг/м³")
p("Плотность жидкости (Н+В):", f"{liquid_density:.1f}", "кг/м³")

_major_header("РАСЧЁТ ПРОПУСКНОЙ СПОСОБНОСТИ СЕПАРАТОРА ПО ЖИДКОСТИ")
p("Внутренний диаметр сепаратора:",
  f"{design.inner_diameter * _TO_MM:.0f}", "мм")
p("Длина цилиндрической части:",
  f"{design.length_cylindrical_part:.1f}", "м")
p("Длина полуоси эллиптического днища:", f"{design.length_semiaxis:.3f}", "м")
p("Объёмный расход жидкости:",
  f"{vol_flow_liquid * SECONDS_PER_DAY:.1f}", "м³/сут")
p("Коэффициент заполнения:",
  f"{FILL_COEFFS[TOTAL] * PERCENT:.1f}", "%")
p("Номинальный объём сепаратора:",
  f"{design.volume_separator:.3f}", "м³")
p("Время пребывания жидкости:",
  f"{sum(rt) / SECONDS_PER_MINUTE:.2f}", "мин")

_minor_header("ПЕРВАЯ СЕКЦИЯ (НЕФТЬ + ВОДА)")
p("Длина первой секции:", f"{design.length_first_section:.1f}", "м")
p("Коэффициент заполнения:",
  f"{FILL_COEFFS[FIRST_SECTION] * PERCENT:.1f}", "%")
p("Объём первой секции:", f"{design.volume[FIRST_SECTION]:.3f}", "м³")
p("Время пребывания (Н+В):",
  f"{rt[FIRST_SECTION] / SECONDS_PER_MINUTE:.3f}", "мин")
p("Пропускная способность:",
  f"{capacities[FIRST_SECTION] * SECONDS_PER_DAY:.2f}", "м³/сут")

_minor_header("СБОРНИК НЕФТИ (ПОСЛЕ ПЕРЕГОРОДКИ)")
p("Длина секции после перегородки:",
  f"{design.length_second_section:.1f}", "м")
p("Коэффициент заполнения:",
  f"{FILL_COEFFS[SECOND_SECTION] * PERCENT:.1f}", "%")
p("Объём секции после перегородки:",
  f"{design.volume[SECOND_SECTION]:.3f}", "м³")
p("Время пребывания:",
  f"{rt[SECOND_SECTION] / SECONDS_PER_MINUTE:.3f}", "мин")
p("Пропускная способность:",
  f"{capacities[SECOND_SECTION] * SECONDS_PER_DAY:.3f}", "м³/сут")

_minor_header("СКОРОСТИ ДВИЖЕНИЯ ФАЗ В СЕЧЕНИИ СЕПАРАТОРА")
p("Площадь сечения для прохода жидкости:",
  f"{areas[OIL] + areas[WATER]:.3f}", "м²")
p("Площадь сечения для прохода газа:", f"{areas[VAPOR]:.3f}", "м²")
p("Площадь сечения для прохода нефти:", f"{areas[OIL]:.3f}", "м²")
p("Площадь сечения для прохода воды:", f"{areas[WATER]:.3f}", "м²")
_minor_divider()
p("Скорость движения газа:", f"{separator.velocity[VAPOR]:.4f}", "м/с")
p("Скорость движения нефти:",
  f"{separator.velocity[OIL] * _TO_MM:.4f}", "мм/с")
p("Скорость движения воды:",
  f"{separator.velocity[WATER] * _TO_MM:.4f}", "мм/с")

_minor_header("ОСАЖДЕНИЕ КАПЕЛЬ ВОДЫ В СЛОЕ НЕФТИ")
p("Расстояние от распределительной решетки до сливной перегородки: ",
  f"{design.length_to_baffle:.1f}", "м")
p("Диаметр капли воды:",
  f"{diameter_water_droplet * _TO_MICRON:.0f}", "мкм")
p("Скорость осаждения капель воды:",
  f"{velocity_water * _TO_MM:.4f}", "мм/с")
p("Время прохождения нефтью:",
  f"{separator.transit_time(OIL):.2f}", "с")
p("Высота осаждения капель воды:", f"{height_water * _TO_MM:.2f}", "мм")

_minor_header("ВСПЛЫТИЕ КАПЕЛЬ НЕФТИ В СЛОЕ ВОДЫ")
p("Расстояние от распределительной решетки до сливной перегородки: ",
  f"{design.length_to_baffle:.1f}", "м")
p("Диаметр капли нефти:",
  f"{diameter_oil_droplet * _TO_MICRON:.0f}", "мкм")
p("Скорость подъёма капель нефти:", f"{velocity_oil * _TO_MM:.4f}", "мм/с")
p("Время прохождения водой:",
  f"{separator.transit_time(WATER):.2f}", "с")
p("Высота подъёма капель нефти:", f"{height_oil * _TO_MM:.2f}", "мм")

_major_header("РАСЧЁТ КОАЛЕСЦЕРА")

_minor_header("ВЕРХНИЙ КОАЛЕСЦЕР")
p("Угол наклона пластин:", f"{packing.angle:.0f}", "°")
p("Зазор между пластинами:",
  f"{packing.coalescer_top_gap * _TO_MM:.0f}", "мм")
p("Время осаждения капель воды в зазоре:",
  f"{t_top / SECONDS_PER_MINUTE:.2f}", "мин")
p("Длина канала:",
  f"{required_length_for(separator.velocity[OIL], t_top):.4f}", "м")

_minor_header("НИЖНИЙ КОАЛЕСЦЕР")
p("Угол наклона пластин:", f"{packing.angle:.0f}", "°")
p("Зазор между пластинами:",
  f"{packing.coalescer_bottom_gap * _TO_MM:.0f}", "мм")
p("Время всплытия капель нефти в зазоре:",
  f"{t_bottom / SECONDS_PER_MINUTE:.2f}", "мин")
p("Длина канала:",
  f"{required_length_for(separator.velocity[WATER], t_bottom):.4f}", "м")

_major_header(
    "РАСЧЁТ СКОРОСТИ ГАЗА В СЕПАРАЦИОННОМ ЭЛЕМЕНТЕ (СПИРАЛЬНЫЙ КАНАЛ)")

_minor_header("ГЕОМЕТРИЯ ЦИКЛОНА")
p("Ширина входа в циклон:",
  f"{geometry.inlet_width * _TO_MM:.1f}", "мм")
p("Высота входа в циклон:",
  f"{geometry.inlet_height * _TO_MM:.1f}", "мм")
p("Количество циклонов:", f"{number_of_cyclones}")
_minor_divider()
p("Расход газа при р.у.:",
  f"{conditions.vol_flow_rate[VAPOR] * SECONDS_PER_DAY:.1f}", "м³/сут")
p("Площадь сечения спирального канала:",
  f"{geometry.area_spiral_channel:.4f}", "м²")
p("Скорость газа в спиральном канале:",
  f"{cyclone.vapor_velocity(number_of_cyclones):.3f}", "м/с")

_major_header("РАСЧЕТ ШТУЦЕРОВ")

_minor_header("Штуцер газа")
p("Скорость:", f"{gas_speed:.2f}", "м/с")
p("Расчетный диаметр:", f"{gas_nozzle.diameter * _TO_MM:.1f}", "мм")
p("Стандартный диаметр:", f"{gas_nozzle.nominal_diameter * _TO_MM:.0f}", "мм")
p("Площадь сечения:", f"{gas_nozzle.nominal_area:.4f}", "м²")
p("Фактическая скорость:",
  f"{gas_nozzle.flow_velocity(conditions.vol_flow_rate[VAPOR]):.4f}", "м/с")

print()
_minor_header("Штуцер нефти")
p("Скорость:", f"{liquid_speed:.2f}", "м/с")
p("Расчетный диаметр:", f"{oil_nozzle.diameter * _TO_MM:.1f}", "мм")
p("Стандартный диаметр:", f"{oil_nozzle.nominal_diameter * _TO_MM:.0f}", "мм")
p("Площадь сечения:", f"{oil_nozzle.nominal_area:.4f}", "м²")
p("Фактическая скорость:",
  f"{oil_nozzle.flow_velocity(conditions.vol_flow_rate[OIL]):.4f}", "м/с")

print()
_minor_header("Штуцер воды")
p("Скорость:", f"{liquid_speed:.2f}", "м/с")
p("Расчетный диаметр:", f"{water_nozzle.diameter * _TO_MM:.1f}", "мм")
p("Стандартный диаметр:",
  f"{water_nozzle.nominal_diameter * _TO_MM:.0f}", "мм")
p("Площадь сечения:", f"{water_nozzle.nominal_area:.4f}", "м²")
p("Фактическая скорость:",
  f"{water_nozzle.flow_velocity(conditions.vol_flow_rate[WATER]):.4f}", "м/с")

print()
_minor_header("Штуцер жидкости")
p("Скорость:", f"{liquid_speed:.2f}", "м/с")
p("Расчетный диаметр:", f"{liquid_nozzle.diameter * _TO_MM:.1f}", "мм")
p("Стандартный диаметр:",
  f"{liquid_nozzle.nominal_diameter * _TO_MM:.0f}", "мм")
p("Площадь сечения:", f"{liquid_nozzle.nominal_area:.4f}", "м²")
p("Фактическая скорость:",
  f"{liquid_nozzle.flow_velocity(vol_flow_liquid):.4f}", "м/с")

print()
_minor_header("Штуцер ГЖС")
p("Скорость газа:", f"{gas_speed:.2f}", "м/с")
p("Скорость жидкости:", f"{liquid_speed:.2f}", "м/с")
p("Расчетный диаметр:", f"{liquid_gas_nozzle.diameter * _TO_MM:.1f}", "мм")
p("Стандартный диаметр:",
  f"{liquid_gas_nozzle.nominal_diameter * _TO_MM:.0f}", "мм")
p("Площадь сечения:", f"{liquid_gas_nozzle.nominal_area:.4f}", "м²")
p("Фактическая скорость:",
  f"{liquid_gas_nozzle.flow_velocity(conditions.vol_flow_rate[VAPOR]):.4f}", "м/с")

# ============================================================
# Сетчатый каплеуловитель — график
# ============================================================

plt.figure(figsize=(5.8, 5))
plt.plot(_PRESSURE_PA / PA_TO_MPA, _FLOW_STABILITY_COEFFICIENT,
         'o', markersize=4, label='Данные по графику')

pressure_smooth = np.linspace(_PRESSURE_PA.min(), _PRESSURE_PA.max(), 200)
flow_stability_coefficient_smooth = _STABILITY_COEFFICIENT_INTERPOLATOR(
    pressure_smooth)

plt.plot(pressure_smooth / PA_TO_MPA, flow_stability_coefficient_smooth, '--',
         alpha=0.7, label='Интерполяция')
plt.xlabel('Давление, МПа', fontsize=12)
plt.ylabel('Коэффициент устойчивости', fontsize=12)
plt.grid(True, alpha=0.3)
plt.ylim(0.4, 1.1)
plt.legend()

plt.plot(pressure / PA_TO_MPA, current_flow_stability_coefficient,
         'ro', markersize=8,
         label=(f'Рабочее давление: {pressure/PA_TO_MPA:.2f} МПа, '
                f'k={current_flow_stability_coefficient:.3f}'))
plt.legend()

# ax = plt.gca()
# ax.set_xlim((2, 13))
# ax.set_ylim((0.4, 1.1))

plt.tight_layout()
plt.show()

_major_header("РЕЗУЛЬТАТЫ РАСЧЁТА СЕТЧАТОГО КАПЛЕУЛОВИТЕЛЯ")
p("Рабочее давление:", f"{pressure / PA_TO_MPA:.1f}", "МПа")
p("Рабочая температура:",
  f"{temperature} К ({temperature - KELVIN_TO_CELSIUS:.0f} °C)")

_minor_divider()
p("Объемный расход газа при н.у.:",
  f"{vol_flow_gas_norm * SECONDS_PER_DAY:,.0f}".replace(",", " "), "м³/сут")
p("Объемный расход газа при р.у.:",
  f"{conditions.vol_flow_rate[VAPOR] * SECONDS_PER_HOUR:.4f}", "м³/ч")
p("Объемный расход жидкости:",
  f"{vol_flow_liquid * SECONDS_PER_DAY:.0f}", "м³/сут")
p("Обводнённость:", f"{water_cut * PERCENT:.0f}", "%")

_minor_divider()
p("Плотность газа в р.у.:",
  f"{conditions.phase[VAPOR].rhomass():.3f}", "кг/м³")
p("Плотность жидкости (Н+В):", f"{liquid_density:.2f}", "кг/м³")
p("Коэффициент k:",
  f"{current_flow_stability_coefficient:.3f}")
p("Критическая скорость:",
  f"{calculate_critical_velocity(conditions, oil_surface_tension):.3f}", "м/с")

_minor_divider()
p("Диаметр:", f"{wmd.diameter * _TO_MM:.1f}", "мм")
p("Принятый диаметр:",
  f"{wmd.nominal_diameter * _TO_MM:.0f}", "мм")
p("Действительная площадь сечения:", f"{wmd.nominal_area:.4f}", "м²")
p("Действительная скорость набегания:", f"{wmd.nominal_velocity:.3f}", "м/с")
p("Производительность:", f"{wmd.nominal_capacity:.4f}", "м³/с")
