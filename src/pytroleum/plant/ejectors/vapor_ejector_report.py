import matplotlib.pyplot as plt

from pytroleum.plant.ejectors.vapor_ejector import VaporEjector
from pytroleum.plant.ejectors.inputs import (ActiveMediumData,
                                             PassiveMediumData,
                                             CommonParams)

from pytroleum.plant.ejectors.utils import (_major_header, _minor_header,
                                            print_row as p,
                                            _minor_divider,
                                            PA_TO_MPA, KELVIN_TO_CELSIUS,
                                            KCAL_TO_J, KCAL_PER_KMOL_TO_J_PER_MOL,
                                            KGS_S_M2_TO_PA_S, KG_PER_KMOL_TO_KG_PER_MOL,
                                            M_TO_MM, M2_TO_MM2)

# ============================================================
# Исходные данные
# ============================================================

active = ActiveMediumData(
    mass_flow=36.25,
    temperature=248,
    inlet_pressure=7e6,
    enthalpy=1045.91 * KCAL_TO_J,
    entropy=1.73 * KCAL_TO_J,
    specific_volume=0.01,
    density=98.89,
    dynamic_viscosity=0.00000099 * KGS_S_M2_TO_PA_S,
    inlet_diameter=0.33,
    molecular_mass=18.70 * KG_PER_KMOL_TO_KG_PER_MOL,
    heat_capacity=16.23 * KCAL_PER_KMOL_TO_J_PER_MOL
)

passive = PassiveMediumData(
    mass_flow=6.52,
    temperature=289,
    inlet_pressure=2.5e6,
    enthalpy=927.89 * KCAL_TO_J,
    entropy=1.69 * KCAL_TO_J,
    specific_volume=0.04,
    density=26.18,
    dynamic_viscosity=0.00000116 * KGS_S_M2_TO_PA_S,
    inlet_diameter=0.11,
    molecular_mass=22.18 * KG_PER_KMOL_TO_KG_PER_MOL,
    heat_capacity=11.64 * KCAL_PER_KMOL_TO_J_PER_MOL
)

common = CommonParams(
    num_stages=1,
    outlet_pressure=4e6,
    outlet_diameter=0.325
)

ejector = VaporEjector(active, passive, common)
ejector.calculate_nozzle_params(
    entropy_lower_boundary=-0.34 * KCAL_TO_J,
    entropy_upper_boundary=2.39 * KCAL_TO_J,
    enthalpy_lower_boundary=-91.89 * KCAL_TO_J,
    phi=0.95,
)

ejector.calculate_mixture_parameters(
    entrainment_ratios=[0.1, 0.3, 0.5],
    pressure_recovery_coefficient=0.70,
    mach_number=0.90,
    enthalpies_cyl_exit=[980.00 * KCAL_TO_J,
                         990.00 * KCAL_TO_J,
                         985.00 * KCAL_TO_J],
    pressure_delta=0.78
)

ejector.calculate_geometry(
    nozzle_expansion_ratio=0.72,
    psi=2.03
)

_major_header("ИСХОДНЫЕ ДАННЫЕ")

_minor_header("АКТИВНАЯ СРЕДА (Эжектирующая)")
p("Массовый расход:", f"{active.mass_flow:.2f}", "кг/с")
p("Температура:",
  f"{active.temperature} К ({active.temperature - KELVIN_TO_CELSIUS:.0f} °C)")
p("Давление на входе:", f"{active.inlet_pressure / PA_TO_MPA:.2f}", "МПа")
p("Энтальпия:", f"{active.enthalpy / KCAL_TO_J:.2f}", "ккал/кг")
p("Энтропия:", f"{active.entropy / KCAL_TO_J:.2f}", "ккал/(кг·°С)")
p("Удельный объём:", f"{active.specific_volume:.3f}", "м³/кг")
p("Плотность:", f"{active.density:.3f}", "кг/м³")
p("Динамическая вязкость:",
  f"{active.dynamic_viscosity / KGS_S_M2_TO_PA_S:.8f}", "кгс·с/м²")
p("Диаметр трубопровода:", f"{active.inlet_diameter * M_TO_MM:.0f}", "мм")
p("Молекулярная масса:",
  f"{active.molecular_mass / KG_PER_KMOL_TO_KG_PER_MOL:.2f}", "кг/кмоль")
p("Теплоёмкость:",
  f"{active.heat_capacity / KCAL_PER_KMOL_TO_J_PER_MOL:.2f}", "ккал/(кмоль·°С)")

_minor_header("ПАССИВНАЯ СРЕДА (Эжектируемая)")
p("Массовый расход:", f"{passive.mass_flow:.2f}", "кг/с")
p("Температура:",
  f"{passive.temperature} К ({passive.temperature - KELVIN_TO_CELSIUS:.0f} °C)")
p("Давление на входе:", f"{passive.inlet_pressure / PA_TO_MPA:.2f}", "МПа")
p("Энтальпия:", f"{passive.enthalpy / KCAL_TO_J:.2f}", "ккал/кг")
p("Энтропия:", f"{passive.entropy / KCAL_TO_J:.2f}", "ккал/(кг·°С)")
p("Удельный объём:", f"{passive.specific_volume:.3f}", "м³/кг")
p("Плотность:", f"{passive.density:.3f}", "кг/м³")
p("Динамическая вязкость:",
  f"{passive.dynamic_viscosity / KGS_S_M2_TO_PA_S:.8f}", "кгс·с/м²")
p("Диаметр трубопровода:", f"{passive.inlet_diameter * M_TO_MM:.0f}", "мм")
p("Молекулярная масса:",
  f"{passive.molecular_mass / KG_PER_KMOL_TO_KG_PER_MOL:.2f}", "кг/кмоль")
p("Теплоёмкость:",
  f"{passive.heat_capacity / KCAL_PER_KMOL_TO_J_PER_MOL:.2f}", "ккал/(кмоль·°С)")

_minor_header("ОБЩИЕ ПАРАМЕТРЫ")
p("Количество ступеней:", f"{common.num_stages}", "шт.")
p("Давление на выходе:", f"{common.outlet_pressure / PA_TO_MPA:.3f}", "МПа")
p("Диаметр выходного трубопровода:",
  f"{common.outlet_diameter * M_TO_MM:.0f}", "мм")

_major_header("РЕЗУЛЬТАТЫ РАСЧЁТА СОПЛА")

p("Энтропия на нижней пограничной кривой s(a)':",
  f"{ejector.entropy_lower_boundary / KCAL_TO_J:.2f}", "ккал/(кг·°С)")
p("Энтропия на верхней пограничной кривой s(a)'':",
  f"{ejector.entropy_upper_boundary / KCAL_TO_J:.2f}", "ккал/(кг·°С)")
p("Энтальпия на нижней пограничной кривой i(a)':",
  f"{ejector.enthalpy_lower_boundary / KCAL_TO_J:.2f}", "ккал/кг")

p("Температура активной среды в сопле t1:",
  f"{ejector.temperature_nozzle_exit:.2f}", "К")
p("Давление газа в конце сопла p1:",
  f"{ejector.pressure_nozzle_exit / PA_TO_MPA:.2f}", "МПа")
p("Скрытая степень льдистости газа в выходном сечении сопла r(a):",
  f"{ejector.latent_heat_nozzle_exit / KCAL_TO_J:.2f}", "ккал/кг")
p("Степень льдистости x(a):",
  f"{ejector.ice_quality_nozzle_exit:.4f}")
p("Энтальпия расширившейся среды i1(a):",
  f"{ejector.enthalpy_nozzle_exit / KCAL_TO_J:.2f}", "ккал/кг")
p("Скорость истечения газа из сопла w1:",
  f"{ejector.velocity_nozzle_exit:.2f}", "м/с")

p("Потери тепла в сопле hc:",
  f"{ejector.heat_loss_nozzle / KCAL_TO_J:.2f}", "ккал/кг")
p("Энтальпия газа в конце сопла с учётом потерь i1:",
  f"{ejector.enthalpy_nozzle_exit_actual / KCAL_TO_J:.2f}", "ккал/кг")
p("Степень льдистости в конце действительного расширения x1:",
  f"{ejector.ice_quality_nozzle_actual:.4f}")

p("Давление в критическом сечении сопла pкр:",
  f"{ejector.pressure_critical / PA_TO_MPA:.4f}", "МПа")
p("Температура в критическом сечении сопла pкр:",
  f"{ejector.temperature_critical:.2f}", "K")

p("Удельный объём насыщенного пара при давлении p1 v'':",
  f"{ejector.vapor_specific_vol:.5f}", "м³/кг")
p("Удельный объём льдистого газа на выходе из сопла v1:",
  f"{ejector.specific_volume_nozzle_exit:.5f}", "м³/кг")

p("Динамический напор эжектирующей струи H_дин:",
  f"{ejector.dynamic_head_nozzle / PA_TO_MPA:.2f}", "МПа")

_major_header("РАСЧЁТ ОСНОВНОГО ГЕОМЕТРИЧЕСКОГО ПАРАМЕТРА СТУПЕНИ")

p("Коэффициент восстановления давления φ:",
  f"{ejector.pressure_recovery_coefficient:.2f}")
p("Число Маха M3:", f"{ejector.mach_number:.2f}")

_minor_header("РЕЗУЛЬТАТЫ ПО КОЭФФИЦИЕНТАМ ЭЖЕКЦИИ")
print("Зададимся коэффициентами эжекции:")
for i, q in enumerate(ejector.stage_entrainment_ratios, 1):
    p(f"q({i}):", f"{q:.1f}")

_minor_divider()
print("Рассчитываем показатель адиабаты смеси в конце цилиндрического участка (сечение III)")
print("каждого коэффициента эжекции:")
for i, k3 in enumerate(ejector.stage_adiabatic_indices, 1):
    p(f"k3({i}):", f"{k3:.3f}")

_minor_divider()
print("Расчёт давления смеси в конце цилиндрического участка:")
for i, p3 in enumerate(ejector.stage_pressures_cyl_exit, 1):
    p(f"p3({i}):", f"{p3 / PA_TO_MPA:.3f}", "МПа")

_minor_divider()
print("Парциальное давление активной среды в конце цилиндрического участка:")
for i, p3_active in enumerate(ejector.stage_partial_pressures_active, 1):
    p(f"p3_active({i}):", f"{p3_active / PA_TO_MPA:.3f}", "МПа")

_minor_divider()
print("Газовая постоянная смеси:")
for i, gas_constant_mixture in enumerate(ejector.stage_gas_constants_mixture, 1):
    p(f"R_см({i}):", f"{gas_constant_mixture:.2f}", "Дж/(кг·К)")

_minor_divider()
print("Заданные энтальпии смеси в конце цилиндрического участка i3:")
for i, i3 in enumerate(ejector.stage_enthalpies_cyl_exit, 1):
    p(f"i3({i}):", f"{i3 / KCAL_TO_J:.1f}", "ккал/кг")

_minor_divider()
print("Температура смеси в конце цилиндрического участка T3")
for i, (t3, q) in enumerate(
        zip(ejector.stage_temperatures_cyl_exit,
            ejector.stage_entrainment_ratios), 1):
    p(f"T3({i}):",
      f"{t3:.2f} К  ({t3 - KELVIN_TO_CELSIUS:.2f} °C)")

_minor_divider()
print("Местная скорость звука в конце цилиндрического участка a(3):")
for i, a3 in enumerate(ejector.stage_sound_velocities, 1):
    p(f"a(3){i}:", f"{a3:.2f}", "м/с")

_minor_divider()
print("Скорость смеси в конце цилиндрического участка w(3):")
for i, w3 in enumerate(ejector.stage_mixture_velocities, 1):
    p(f"w(3){i}:", f"{w3:.2f}", "м/с")

_minor_divider()
print("Удельный объём смеси в конце цилиндрического участка v(3):")
for i, v3 in enumerate(ejector.stage_specific_volumes, 1):
    p(f"v(3){i}:", f"{v3:.4f}", "м³/кг")

_minor_divider()
print("Основной геометрический параметр ступени m:")
for i, m in enumerate(ejector.stage_geometric_params, 1):
    p(f"m({i}):", f"{m:.2f}")

_minor_divider()
print("Расчёт давления смеси")
for i, p3 in enumerate(ejector.stage_mixture_pressures, 1):
    p(f"p(3){i}:", f"{p3 / PA_TO_MPA:.3f}", "МПа")

ejector.plot_mixture_pressure_vs_entrainment()
plt.show()

_major_header("ГЕОМЕТРИЧЕСКИЕ ПАРАМЕТРЫ СОПЛА")

p("Расчётная площадь выходного сечения F1*:",
  f"{ejector.nozzle_exit_area_theoretical * M2_TO_MM2:.2f}", "мм²")
p("Площадь критического сечения Fкр:",
  f"{ejector.nozzle_throat_area * M2_TO_MM2:.2f}", "мм²")
p("Диаметр критического сечения Dкр:",
  f"{ejector.nozzle_throat_diameter * M_TO_MM:.2f}", "мм")
p("Диаметр критического сечения с учётом погранслоя Dкр*:",
  f"{ejector.nozzle_throat_diameter_corrected * M_TO_MM:.2f}", "мм")

p("Степень расширения сопла ϑ:",
  f"{ejector.nozzle_exit_area / ejector.nozzle_exit_area_theoretical:.2f}")
p("Действительная площадь выходного сечения F1:",
  f"{ejector.nozzle_exit_area * M2_TO_MM2:.2f}", "мм²")
p("Действительный диаметр выходного сечения D1:",
  f"{ejector.nozzle_exit_diameter * M_TO_MM:.2f}", "мм")
p("Диаметр сопла D' равен диаметру трубопровода активной среды:",
  f"{ejector.nozzle_diameter * M_TO_MM:.2f}", "мм")

p("Диаметр камеры разряжения D2 (Dп):",
  f"{ejector.vacuum_chamber_diameter * M_TO_MM:.2f}", "мм")
p("Входной диаметр камеры смешения Dк':",
  f"{ejector.inlet_mixing_diameter * M_TO_MM:.2f}", "мм")
p("Диаметр камеры смешения Dк:",
  f"{ejector.mixing_diameter * M_TO_MM:.2f}", "мм")
p("Радиус кривизны камеры смешения Rк:",
  f"{ejector.curvature_radius * M_TO_MM:.2f}", "мм")
p("Длина камеры смешения Lк:",
  f"{ejector.mixing_length * M_TO_MM:.2f}", "мм")
