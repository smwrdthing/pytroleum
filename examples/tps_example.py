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
from pytroleum.plant.tps.devices import Coalescer, Cyclone
from pytroleum.plant.tps.nozzle import (
    Nozzle, design_nozzle, design_two_phase_nozzle,
)
from pytroleum.plant.tps.wire_mesh_demister import (
    design_demister, get_flow_stability_coefficient,
    calculate_critical_velocity,
    _PRESSURE_PA, _FLOW_STABILITY_COEFFICIENT, _STABILITY_COEFFICIENT_INTERPOLATOR,
)
from pytroleum.plant.tps.utils import (
    _major_header, _minor_header, _minor_divider,
    SECONDS_PER_DAY, SECONDS_PER_HOUR, SECONDS_PER_MINUTE,
    PA_TO_MPA, PERCENT, KG_S_TO_T_H,
    _TO_MM, _TO_MICRON, _TO_M,
    KELVIN_TO_CELSIUS,
)

# ============================================================
# Исходные данные
# ============================================================

pressure = 2e6          # Па, абсолютное
temperature = 273.15 + 50  # К

vol_flow_gas_norm = 300_000 / SECONDS_PER_DAY  # м³/с при н.у.
vol_flow_oil = 200 / SECONDS_PER_DAY           # м³/с
vol_flow_water = 300 / SECONDS_PER_DAY         # м³/с

oil_density = 933        # кг/м³
oil_viscosity = 3.073e-3  # Па·с
oil_surface_tension = 0.02848  # Н/м

diameter_water_droplet = 100e-6  # м
diameter_oil_droplet = 50e-6     # м

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

_major_header("УСЛОВИЯ РАБОТЫ")
print(f"Давление при н.у.:    {STANDARD_PRESSURE / PA_TO_MPA:.3f} МПа")
print(f"Температура при н.у.: {STANDARD_TEMPERATURE} К")
print(f"Рабочее давление:     {pressure / PA_TO_MPA:.1f} МПа")
print(f"Рабочая температура:  {temperature} К "
      f"({temperature - KELVIN_TO_CELSIUS:.0f} °C)")
print(f"Объемный расход газа при н.у.: "
      f"{vol_flow_gas_norm * SECONDS_PER_DAY:,.0f} м³/сут".replace(",", " "))
vol_flow_liquid = conditions.vol_flow_rate[OIL] + conditions.vol_flow_rate[WATER]
print(f"Объемный расход жидкости: "
      f"{vol_flow_liquid * SECONDS_PER_DAY:.0f} м³/сут")

_major_header("СВОЙСТВА ФЛЮИДА")
print(f"Плотность газа при н.у.: {gas_density_norm:.3f} кг/м³")
print(f"Плотность нефти:         {conditions.phase[OIL].rhomass():.0f} кг/м³")
print(f"Плотность воды:          {conditions.phase[WATER].rhomass():.0f} кг/м³")
print(f"Обводненность:           {flow_based_water_cut(conditions) * PERCENT:.0f}%")

_major_header("ОБЪЕМНЫЕ РАСХОДЫ")
print(f"Объемный расход газа при р.у.: "
      f"{conditions.vol_flow_rate[VAPOR] * SECONDS_PER_DAY:.0f} м³/сут")
print(f"Объемный расход по нефти:      "
      f"{conditions.vol_flow_rate[OIL] * SECONDS_PER_DAY:.0f} м³/сут")
print(f"Объемный расход по воде:       "
      f"{conditions.vol_flow_rate[WATER] * SECONDS_PER_DAY:.0f} м³/сут")

_major_header("МАССОВЫЕ РАСХОДЫ (кг/с)")
print(f"Массовый расход газа:              {mass_flow_gas:.2f} кг/с")
print(f"Массовый расход нефти:             {mass_flow_oil:.2f} кг/с")
print(f"Массовый расход воды:              {mass_flow_water:.2f} кг/с")
print(f"Массовый расход жидкости (Н+В):    {mass_flow_liquid:.2f} кг/с")
print(f"Массовый суммарный расход (Г+Н+В): {mass_flow_total:.2f} кг/с")

_major_header("МАССОВЫЕ РАСХОДЫ (т/ч)")
print(f"Массовый расход газа:              {mass_flow_gas * KG_S_TO_T_H:.2f} т/ч")
print(f"Массовый расход нефти:             {mass_flow_oil * KG_S_TO_T_H:.2f} т/ч")
print(f"Массовый расход воды:              {mass_flow_water * KG_S_TO_T_H:.2f} т/ч")
print(f"Массовый расход жидкости (Н+В):    {mass_flow_liquid * KG_S_TO_T_H:.2f} т/ч")
print(f"Массовый суммарный расход (Г+Н+В): {mass_flow_total * KG_S_TO_T_H:.2f} т/ч")

_major_header("ФИЗИЧЕСКИЕ СВОЙСТВА ПРИ РАБОЧИХ УСЛОВИЯХ")
print(f"Плотность газа в р.у.: {conditions.phase[VAPOR].rhomass():.3f} кг/м³")
water_cut = flow_based_water_cut(conditions)
liquid_density = (conditions.phase[OIL].rhomass() * (1 - water_cut) +
                  conditions.phase[WATER].rhomass() * water_cut)
print(f"Плотность жидкости (Н+В) при заданной обводненности: "
      f"{liquid_density:.1f} кг/м³")

# ============================================================
# Сепаратор
# ============================================================

design = SeparatorDesign(
    inner_diameter=2.0,
    length_cylindrical_part=9.5,
    length_semiaxis=0.618,
    length_first_section=8.2,
    length_second_section=1.3,
    length_to_baffle=4.7,
)

separator = Separator(design=design, conditions=conditions)
separator.compute_velocities()
rt = separator.residence_time()
capacities = separator.capacity()

rho_oil = conditions.phase[OIL].rhomass()
rho_water = conditions.phase[WATER].rhomass()
mu_oil = conditions.phase[OIL].viscosity()
mu_water = conditions.phase[WATER].viscosity()

_major_header("РАСЧЁТ ПРОПУСКНОЙ СПОСОБНОСТИ СЕПАРАТОРА ПО ЖИДКОСТИ")
print(f"Внутренний диаметр сепаратора:      {design.inner_diameter * _TO_MM:.0f} мм")
print(f"Длина цилиндрической части:         {design.length_cylindrical_part:.1f} м")
print(f"Длина полуоси эллиптического днища: {design.length_semiaxis:.3f} м")
print(f"Объёмный расход жидкости: "
      f"{vol_flow_liquid * SECONDS_PER_DAY:.1f} м³/сут")
print(f"Коэффициент заполнения:             {FILL_COEFFS[TOTAL] * PERCENT:.1f} %")
print(f"Номинальный объём сепаратора:       {design.volume_separator:.3f} м³")
print(f"Время пребывания жидкости:          {rt[TOTAL] / SECONDS_PER_MINUTE:.2f} мин")

_minor_header("ПЕРВАЯ СЕКЦИЯ (НЕФТЬ + ВОДА)")
print(f"Длина первой секции:    {design.length_first_section:.1f} м")
print(f"Коэффициент заполнения: {FILL_COEFFS[FIRST_SECTION] * PERCENT:.1f} %")
print(f"Объём первой секции:    {design.volume[FIRST_SECTION]:.3f} м³")
print(f"Время пребывания (Н+В): {rt[FIRST_SECTION] / SECONDS_PER_MINUTE:.3f} мин")
print(f"Пропускная способность: "
      f"{capacities[FIRST_SECTION] * SECONDS_PER_DAY:.2f} м³/сут")

_minor_header("СБОРНИК НЕФТИ (ПОСЛЕ ПЕРЕГОРОДКИ)")
print(f"Длина секции после перегородки: {design.length_second_section:.1f} м")
print(f"Коэффициент заполнения:         {FILL_COEFFS[SECOND_SECTION] * PERCENT:.1f} %")
print(f"Объём секции после перегородки: {design.volume[SECOND_SECTION]:.3f} м³")
print(f"Время пребывания:       {rt[SECOND_SECTION] / SECONDS_PER_MINUTE:.3f} мин")
print(f"Пропускная способность: "
      f"{capacities[SECOND_SECTION] * SECONDS_PER_DAY:.3f} м³/сут")

_minor_header("СКОРОСТИ ДВИЖЕНИЯ ФАЗ В СЕЧЕНИИ СЕПАРАТОРА")
areas = separator.compute_flow_areas()
print(f"Площадь сечения для прохода жидкости: {areas[OIL] + areas[WATER]:.3f} м²")
print(f"Площадь сечения для прохода газа:     {areas[VAPOR]:.3f} м²")
print(f"Площадь сечения для прохода нефти:    {areas[OIL]:.3f} м²")
print(f"Площадь сечения для прохода воды:     {areas[WATER]:.3f} м²")
_minor_divider()
print(f"Скорость движения газа:  {separator.velocity[VAPOR]:.4f} м/с")
print(f"Скорость движения нефти: {separator.velocity[OIL] * _TO_MM:.4f} мм/с")
print(f"Скорость движения воды:  {separator.velocity[WATER] * _TO_MM:.4f} мм/с")

_minor_header("ОСАЖДЕНИЕ КАПЕЛЬ ВОДЫ В СЛОЕ НЕФТИ")
print(f"Расстояние решётка — перегородка: {design.length_to_baffle:.1f} м")
print(f"Диаметр капли воды: {diameter_water_droplet * _TO_MICRON:.0f} мкм")
velocity_water = abs(compute_settling_velocity(
    diameter_water_droplet, rho_oil, mu_oil, rho_water))
print(f"Скорость осаждения капель воды: {velocity_water * _TO_MM:.4f} мм/с")
print(f"Время прохождения нефтью: {separator.transit_time(OIL):.2f} с")
height_water = separator.settling_height(
    diameter_water_droplet, rho_oil, mu_oil, rho_water, OIL)
print(f"Высота осаждения капель воды: {height_water * _TO_MM:.2f} мм")

_minor_header("ВСПЛЫТИЕ КАПЕЛЬ НЕФТИ В СЛОЕ ВОДЫ")
print(f"Расстояние решётка — перегородка: {design.length_to_baffle:.1f} м")
print(f"Диаметр капли нефти: {diameter_oil_droplet * _TO_MICRON:.0f} мкм")
velocity_oil = compute_settling_velocity(
    diameter_oil_droplet, rho_water, mu_water, rho_oil)
print(f"Скорость подъёма капель нефти: {velocity_oil * _TO_MM:.4f} мм/с")
print(f"Время прохождения водой: {separator.transit_time(WATER):.2f} с")
height_oil = separator.settling_height(
    diameter_oil_droplet, rho_water, mu_water, rho_oil, WATER)
print(f"Высота подъёма капель нефти: {height_oil * _TO_MM:.2f} мм")

# ============================================================
# Коалесцер и циклон
# ============================================================

coalescer_packing = CoalescerPacking(coalescer_top_gap=15e-3,
                                     coalescer_bottom_gap=25e-3)
number_of_cyclones = 4
geometry_cyclone = GeometryCyclone(width_inlet_cyclone=47.5e-3,
                                   height_inlet_cyclone=75e-3)

coalescer = Coalescer(coalescer_packing=coalescer_packing, separator=separator)
cyclone = Cyclone(conditions=conditions, geometry_cyclone=geometry_cyclone)

_major_header("РАСЧЁТ КОАЛЕСЦЕРА")

_minor_header("ВЕРХНИЙ КОАЛЕСЦЕР")
print(f"Угол наклона пластин: {coalescer_packing.angle:.0f}°")
print(f"Зазор между пластинами: {coalescer_packing.coalescer_top_gap * _TO_MM:.0f} мм")
t_top = coalescer.droplet_settling_time(
    coalescer_packing.coalescer_top_gap,
    diameter_water_droplet,
    rho_oil, mu_oil, rho_water,
)
print(f"Время осаждения капель воды в зазоре: {t_top / SECONDS_PER_MINUTE:.2f} мин")
print(f"Длина канала: "
      f"{coalescer.required_length_for(separator.velocity[OIL], t_top):.4f} м")

_minor_header("НИЖНИЙ КОАЛЕСЦЕР")
print(f"Угол наклона пластин: {coalescer_packing.angle:.0f}°")
print(f"Зазор между пластинами: "
      f"{coalescer_packing.coalescer_bottom_gap * _TO_MM:.0f} мм")
t_bottom = coalescer.droplet_settling_time(
    coalescer_packing.coalescer_bottom_gap,
    diameter_oil_droplet,
    rho_water, mu_water, rho_oil,
)
print(f"Время всплытия капель нефти в зазоре: {t_bottom / SECONDS_PER_MINUTE:.2f} мин")
print(f"Длина канала: "
      f"{coalescer.required_length_for(separator.velocity[WATER], t_bottom):.4f} м")

_major_header("РАСЧЁТ СКОРОСТИ ГАЗА В СЕПАРАЦИОННОМ ЭЛЕМЕНТЕ (СПИРАЛЬНЫЙ КАНАЛ)")

_minor_header("ГЕОМЕТРИЯ ЦИКЛОНА")
print(f"Ширина входа в циклон: {geometry_cyclone.width_inlet_cyclone * _TO_MM:.1f} мм")
print(f"Высота входа в циклон: {geometry_cyclone.height_inlet_cyclone * _TO_MM:.1f} мм")
print(f"Количество циклонов:   {number_of_cyclones}")
_minor_divider()
print(f"Расход газа при р.у.:          "
      f"{conditions.vol_flow_rate[VAPOR] * SECONDS_PER_DAY:.1f} м³/сут")
print(f"Площадь сечения спирального канала: "
      f"{geometry_cyclone.area_spiral_channel:.4f} м²")
print(f"Скорость газа в спиральном канале:  "
      f"{cyclone.vapor_velocity(number_of_cyclones):.3f} м/с")

# ============================================================
# Штуцера
# ============================================================

gas_speed = 10.0
liquid_speed = 1.0

gas_nozzle = design_nozzle(conditions.vol_flow_rate[VAPOR], gas_speed)
oil_nozzle = design_nozzle(conditions.vol_flow_rate[OIL], liquid_speed)
water_nozzle = design_nozzle(conditions.vol_flow_rate[WATER], liquid_speed)
liquid_nozzle = design_nozzle(vol_flow_liquid, liquid_speed)
liquid_gas_nozzle = design_two_phase_nozzle(conditions=conditions,
                                            gas_speed=gas_speed,
                                            liquid_speed=liquid_speed)

_major_header("РАСЧЕТ ШТУЦЕРОВ")

_minor_header("Штуцер газа")
print(f"Скорость: {gas_speed:.2f} м/с")
print(f"Расчетный диаметр:  {gas_nozzle.diameter * _TO_MM:.1f} мм")
print(f"Стандартный диаметр: {gas_nozzle.nominal_diameter * _TO_MM:.0f} мм")
print(f"Площадь сечения: {gas_nozzle.nominal_area:.4f} м²")
print(f"Фактическая скорость: "
      f"{gas_nozzle.flow_velocity(conditions.vol_flow_rate[VAPOR]):.4f} м/с")

print()
_minor_header("Штуцер нефти")
print(f"Скорость: {liquid_speed:.2f} м/с")
print(f"Расчетный диаметр:   {oil_nozzle.diameter * _TO_MM:.1f} мм")
print(f"Стандартный диаметр: {oil_nozzle.nominal_diameter * _TO_MM:.0f} мм")
print(f"Площадь сечения: {oil_nozzle.nominal_area:.4f} м²")
print(f"Фактическая скорость: "
      f"{oil_nozzle.flow_velocity(conditions.vol_flow_rate[OIL]):.4f} м/с")

print()
_minor_header("Штуцер воды")
print(f"Скорость: {liquid_speed:.2f} м/с")
print(f"Расчетный диаметр:   {water_nozzle.diameter * _TO_MM:.1f} мм")
print(f"Стандартный диаметр: {water_nozzle.nominal_diameter * _TO_MM:.0f} мм")
print(f"Площадь сечения: {water_nozzle.nominal_area:.4f} м²")
print(f"Фактическая скорость: "
      f"{water_nozzle.flow_velocity(conditions.vol_flow_rate[WATER]):.4f} м/с")

print()
_minor_header("Штуцер жидкости")
print(f"Скорость: {liquid_speed:.2f} м/с")
print(f"Расчетный диаметр:   {liquid_nozzle.diameter * _TO_MM:.1f} мм")
print(f"Стандартный диаметр: {liquid_nozzle.nominal_diameter * _TO_MM:.0f} мм")
print(f"Площадь сечения: {liquid_nozzle.nominal_area:.4f} м²")
print(f"Фактическая скорость: "
      f"{liquid_nozzle.flow_velocity(vol_flow_liquid):.4f} м/с")

print()
_minor_header("Штуцер ГЖС")
print(f"Скорость газа:     {gas_speed:.2f} м/с")
print(f"Скорость жидкости: {liquid_speed:.2f} м/с")
print(f"Расчетный диаметр:   {liquid_gas_nozzle.diameter * _TO_MM:.1f} мм")
print(f"Стандартный диаметр: {liquid_gas_nozzle.nominal_diameter * _TO_MM:.0f} мм")
print(f"Площадь сечения: {liquid_gas_nozzle.nominal_area:.4f} м²")
print(f"Фактическая скорость: "
      f"{liquid_gas_nozzle.flow_velocity(conditions.vol_flow_rate[VAPOR]):.4f} м/с")

# ============================================================
# Сетчатый каплеуловитель
# ============================================================

plt.figure(figsize=(10, 6))
plt.plot(_PRESSURE_PA / PA_TO_MPA, _FLOW_STABILITY_COEFFICIENT,
         'o', markersize=4, label='Данные по графику')

pressure_smooth = np.linspace(_PRESSURE_PA.min(), _PRESSURE_PA.max(), 200)
flow_stability_coefficient_smooth = _STABILITY_COEFFICIENT_INTERPOLATOR(
    pressure_smooth)

plt.plot(pressure_smooth / PA_TO_MPA, flow_stability_coefficient_smooth, '--',
         alpha=0.7, label='Интерполяция')
plt.xlabel('Давление, МПа', fontsize=12)
plt.ylabel('Коэффициент устойчивости', fontsize=12)
plt.title('Коэффициент устойчивости режимов течения от давления')
plt.grid(True, alpha=0.3)
plt.ylim(0.4, 1.1)
plt.legend()

current_pressure = pressure
current_flow_stability_coefficient = get_flow_stability_coefficient(pressure)
plt.plot(current_pressure / PA_TO_MPA, current_flow_stability_coefficient,
         'ro', markersize=8,
         label=(f'Рабочее давление: {current_pressure/PA_TO_MPA:.2f} МПа, '
                f'k={current_flow_stability_coefficient:.3f}'))
plt.legend()

# ax = plt.gca()
# ax.set_xlim((2, 13))
# ax.set_ylim((0.4, 1.1))

plt.tight_layout()
plt.show()

_major_header("РЕЗУЛЬТАТЫ РАСЧЁТА СЕТЧАТОГО КАПЛЕУЛОВИТЕЛЯ")
print(f"Рабочее давление:    {pressure / PA_TO_MPA} МПа")
print(f"Рабочая температура: {temperature} К "
      f"({temperature - KELVIN_TO_CELSIUS} °C)")

_minor_divider()
print(f"Объемный расход газа при н.у.: "
      f"{vol_flow_gas_norm * SECONDS_PER_DAY:,.0f} м³/сут".replace(",", " "))
print(f"Объемный расход газа при р.у.: "
      f"{conditions.vol_flow_rate[VAPOR] * SECONDS_PER_HOUR:.4f} м³/ч")
print(f"Объемный расход жидкости:      "
      f"{vol_flow_liquid * SECONDS_PER_DAY:.0f} м³/сут")
print(f"Обводнённость:                 "
      f"{flow_based_water_cut(conditions) * PERCENT:.0f} %")

_minor_divider()
print(f"Плотность газа в р.у.:   {conditions.phase[VAPOR].rhomass():.3f} кг/м³")
print(f"Плотность жидкости (Н+В): {liquid_density:.2f} кг/м³")
print(f"Коэффициент k:            {current_flow_stability_coefficient:.3f}")
print(f"Критическая скорость:     "
      f"{calculate_critical_velocity(conditions, oil_surface_tension):.3f} м/с")

_minor_divider()
wmd = design_demister(conditions, oil_surface_tension)
print(f"Диаметр:                    {wmd.diameter * _TO_MM:.1f} мм")
print(f"Принятый диаметр:           {wmd.nominal_diameter * _TO_MM:.0f} мм")
print(f"Действительная площадь сечения:    {wmd.actual_area:.4f} м²")
print(f"Действительная скорость набегания: {wmd.actual_velocity:.3f} м/с")
print(f"Производительность:         {wmd.capacity:.4f} м³/с")
