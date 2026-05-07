from pytroleum.plant.ejector.inputs import (ActiveMediumData,
                                            PassiveMediumData,
                                            CommonParams)
from pytroleum.plant.ejector.gas_ejector import calculate_gas_ejector
from pytroleum.plant.ejector.equations import (
    calculate_gas_constant,
    calculate_specific_heat_capacity,
    calculate_adiabatic_index,
)
from pytroleum.plant.ejector.utils import (_major_header, _minor_header,
                                           print_row as p,
                                           PA_TO_MPA, KELVIN_TO_CELSIUS,
                                           KCAL_TO_J, KCAL_PER_KMOL_TO_J_PER_MOL,
                                           KGS_S_M2_TO_PA_S, KG_PER_KMOL_TO_KG_PER_MOL,
                                           KG_PER_MOL_TO_G_PER_MOL, J_TO_KJ, M_TO_MM)

# ============================================================
# Исходные данные
# ============================================================

# Активная среда (эжектирующая)
active = ActiveMediumData(
    mass_flow=36.25,                                    # кг/с
    temperature=248,                                    # К
    inlet_pressure=7e6,                                 # Па
    enthalpy=1045.91 * KCAL_TO_J,                      # ккал/кг → Дж/кг
    entropy=1.73 * KCAL_TO_J,                 # ккал/(кг·°С) → Дж/(кг·К)
    specific_volume=0.01,                               # м³/кг
    density=98.89,                                      # кг/м³
    dynamic_viscosity=0.00000099 * KGS_S_M2_TO_PA_S,  # кгс·с/м² → Па·с
    inlet_diameter=0.33,                                # м
    molecular_mass=18.70 * KG_PER_KMOL_TO_KG_PER_MOL,  # кг/кмоль → кг/моль
    # ккал/(кмоль·°С) → Дж/(моль·К)
    heat_capacity=16.23 * KCAL_PER_KMOL_TO_J_PER_MOL
)

# Пассивная среда (эжектируемая)
passive = PassiveMediumData(
    mass_flow=6.52,                                      # кг/с
    temperature=289,                                     # К
    inlet_pressure=2.5e6,                                # Па
    enthalpy=927.89 * KCAL_TO_J,                        # ккал/кг → Дж/кг
    entropy=1.69 * KCAL_TO_J,                  # ккал/(кг·°С) → Дж/(кг·К)
    specific_volume=0.04,                                # м³/кг
    density=26.18,                                       # кг/м³
    dynamic_viscosity=0.00000116 * KGS_S_M2_TO_PA_S,   # кгс·с/м² → Па·с
    inlet_diameter=0.11,                                 # м
    molecular_mass=22.18 * KG_PER_KMOL_TO_KG_PER_MOL,  # кг/кмоль → кг/моль
    # ккал/(кмоль·°С) → Дж/(моль·К)
    heat_capacity=11.64 * KCAL_PER_KMOL_TO_J_PER_MOL
)

# Общие параметры
common = CommonParams(
    num_stages=1,          # количество ступеней
    outlet_pressure=4e6,   # Па
    outlet_diameter=0.325  # м
)

ejector = calculate_gas_ejector(
    active, passive, common,
    s=2,
    mixture_density=56.05,
    pressure_recovery_coefficient=0.8,
    psi=2.14,
    opening_angle=8,
    mixture_dynamic_viscosity=0.0000012 * KGS_S_M2_TO_PA_S)

# ============================================================
# Вывод результатов расчета эжектора
# ============================================================

_major_header("ИСХОДНЫЕ ДАННЫЕ")

_minor_header("АКТИВНАЯ СРЕДА (Эжектирующая)")
p("Массовый расход:", f"{active.mass_flow:.2f}", "кг/с")
p("Температура:",
  f"{active.temperature} К ({active.temperature - KELVIN_TO_CELSIUS:.0f} °C)")
p("Давление на входе:", f"{active.inlet_pressure/PA_TO_MPA:.3f}", "МПа")
p("Энтальпия:", f"{active.enthalpy/J_TO_KJ:.2f}", "кДж/кг")
p("Энтропия:", f"{active.entropy:.2f}", "Дж/(кг·К)")
p("Удельный объем:", f"{active.specific_volume:.3f}", "м³/кг")
p("Плотность:", f"{active.density:.3f}", "кг/м³")
p("Динамическая вязкость:", f"{active.dynamic_viscosity:.6f}", "Па·с")
p("Диаметр трубопровода:", f"{active.inlet_diameter * M_TO_MM:.0f}", "мм")
p("Молекулярная масса:",
  f"{active.molecular_mass * KG_PER_MOL_TO_G_PER_MOL:.1f}", "г/моль")
p("Теплоемкость:", f"{active.heat_capacity/J_TO_KJ:.3f}", "кДж/(моль·К)")

_minor_header("ПАССИВНАЯ СРЕДА (Эжектируемая)")
p("Массовый расход:", f"{passive.mass_flow:.2f}", "кг/с")
p("Температура:",
  f"{passive.temperature} К ({passive.temperature - KELVIN_TO_CELSIUS:.0f} °C)")
p("Давление на входе:", f"{passive.inlet_pressure/PA_TO_MPA:.3f}", "МПа")
p("Энтальпия:", f"{passive.enthalpy/J_TO_KJ:.2f}", "кДж/кг")
p("Энтропия:", f"{passive.entropy:.2f}", "Дж/(кг·К)")
p("Удельный объем:", f"{passive.specific_volume:.3f}", "м³/кг")
p("Плотность:", f"{passive.density:.3f}", "кг/м³")
p("Динамическая вязкость:", f"{passive.dynamic_viscosity:.6f}", "Па·с")
p("Диаметр трубопровода:", f"{passive.inlet_diameter * M_TO_MM:.0f}", "мм")
p("Молекулярная масса:",
  f"{passive.molecular_mass * KG_PER_MOL_TO_G_PER_MOL:.1f}", "г/моль")
p("Теплоемкость:", f"{passive.heat_capacity/J_TO_KJ:.3f}", "кДж/(моль·К)")

_minor_header("ОБЩИЕ ПАРАМЕТРЫ")
p("Количество ступеней:", f"{common.num_stages}", "шт.")
p("Давление на выходе:", f"{common.outlet_pressure/PA_TO_MPA:.3f}", "МПа")
p("Диаметр выходного трубопровода:",
  f"{common.outlet_diameter * M_TO_MM:.0f}", "мм")

_major_header("РЕЗУЛЬТАТЫ РАСЧЕТА ЭЖЕКТОРА")

R_active = calculate_gas_constant(active.molecular_mass)
R_passive = calculate_gas_constant(passive.molecular_mass)
Cp_active = calculate_specific_heat_capacity(
    active.heat_capacity, active.molecular_mass)
Cp_passive = calculate_specific_heat_capacity(
    passive.heat_capacity, passive.molecular_mass)
adiabatic_index = calculate_adiabatic_index(
    active.molecular_mass, active.heat_capacity,
    passive.molecular_mass, passive.heat_capacity,
    ejector.ejection_params.entrainment_ratio)

_minor_header("ОСНОВНЫЕ ПАРАМЕТРЫ ЭЖЕКЦИИ")
p("Степень сжатия:", f"{ejector.ejection_params.compression_ratio:.4f}")
p("Коэффициент эжекции:",
  f"{ejector.ejection_params.entrainment_ratio:.4f}")
p("m1 (участок струи до стенки):", f"{ejector.ejection_params.m1:.4f}")
p("m (основной геометрический параметр):",
  f"{ejector.ejection_params.m:.4f}")
p("n:", f"{ejector.ejection_params.n:.4f}")
p("Напор эжектора без диффузора:",
  f"{ejector.pressure.ejector_head_no_diff/PA_TO_MPA:.3f}", "МПа")

_minor_header("ГАЗОДИНАМИЧЕСКИЕ ПАРАМЕТРЫ")
p("Газовая постоянная активной среды R_a:", f"{R_active:.2f}", "Дж/(кг·К)")
p("Газовая постоянная пассивной среды R_n:",
  f"{R_passive:.2f}", "Дж/(кг·К)")
p("Теплоемкость активной среды Cp_a:", f"{Cp_active:.2f}", "Дж/(кг·К)")
p("Теплоемкость пассивной среды Cp_n:", f"{Cp_passive:.2f}", "Дж/(кг·К)")
p("Показатель адиабаты k:", f"{adiabatic_index:.4f}")
p("Критическое отношение давлений β:",
  f"{ejector.ejection_params.critical_pressure_ratio:.6f}")

_minor_header("ДАВЛЕНИЯ")
p("Динамический напор на выходе из сопла (I-I):",
  f"{ejector.pressure.dynamic_head_nozzle_exit/PA_TO_MPA:.3f}", "МПа")
p("Напор эжектора без диффузора:",
  f"{ejector.pressure.ejector_head_no_diff/PA_TO_MPA:.3f}", "МПа")
p("Критическое давление P_кр:",
  f"{ejector.pressure.critical/PA_TO_MPA:.3f}", "МПа")
p("Давление в конце цилиндрического участка (III-III):",
  f"{ejector.pressure.cyl_section_exit/PA_TO_MPA:.3f}", "МПа")
p("Давление за диффузором:",
  f"{ejector.pressure.ejector_outlet/PA_TO_MPA:.3f}", "МПа")

_minor_header("СКОРОСТИ")
p("Скорость активной среды в трубопроводе:",
  f"{ejector.velocity.active_inlet:.2f}", "м/с")
p("Скорость пассивной среды в трубопроводе:",
  f"{ejector.velocity.passive_inlet:.2f}", "м/с")
p("Скорость истечения газа из сопла (w1):",
  f"{ejector.velocity.nozzle_exit:.2f}", "м/с")
p("Скорость газа в конце смесительного участка (w3):",
  f"{ejector.velocity.cyl_section_exit:.2f}", "м/с")
p("Скорость на выходе из эжектора (w4):",
  f"{ejector.velocity.ejector_outlet:.2f}", "м/с")

_minor_header("ТЕМПЕРАТУРЫ")
p("Температура в критическом сечении сопла (t1):",
  f"{ejector.temperature.nozzle_exit:.2f} К "
  f"({ejector.temperature.nozzle_exit - KELVIN_TO_CELSIUS:.0f} °C)")
p("Температура в конце цилиндрического участка (t3):",
  f"{ejector.temperature.cyl_section_exit:.2f} К "
  f"({ejector.temperature.cyl_section_exit - KELVIN_TO_CELSIUS:.0f} °C)")
p("Температура на выходе из диффузора (t4):",
  f"{ejector.temperature.diffuser_exit:.2f} К "
  f"({ejector.temperature.diffuser_exit - KELVIN_TO_CELSIUS:.0f} °C)")

_minor_header("ГЕОМЕТРИЧЕСКИЕ РАЗМЕРЫ")
p("Площадь выходного сечения сопла F1:",
  f"{ejector.geometry.nozzle_exit_area:.4f}", "м²")
p("Диаметр выходного сечения сопла D1:",
  f"{ejector.geometry.nozzle_exit_diameter * M_TO_MM:.2f}", "мм")
p("Площадь узкой части сопла Fкр:",
  f"{ejector.geometry.nozzle_throat_area:.4f}", "м²")
p("Диаметр узкой части сопла Dкр:",
  f"{ejector.geometry.nozzle_throat_diameter * M_TO_MM:.2f}", "мм")
p("Площадь сечения смесительного участка F3:",
  f"{ejector.geometry.mixing_section_area:.4f}", "м²")
p("Диаметр смесительного участка D3:",
  f"{ejector.geometry.mixing_section_diameter * M_TO_MM:.2f}", "мм")
p("Площадь конечного сечения диффузора F4:",
  f"{ejector.geometry.diffuser_exit_area:.4f}", "м²")
p("Длина струи Lx'':",
  f"{ejector.geometry.jet_length * M_TO_MM:.2f}", "мм")
p("Длина смесительного участка Lсм:",
  f"{ejector.geometry.mixing_section_length * M_TO_MM:.2f}", "мм")
p("Расстояние от сопла до цилиндрического участка L1:",
  f"{ejector.geometry.nozzle_to_inlet_distance * M_TO_MM:.2f}", "мм")
p("Длина цилиндрического участка L2:",
  f"{ejector.geometry.cylinder_length * M_TO_MM:.2f}", "мм")
p("Длина диффузора L3:",
  f"{ejector.geometry.diffuser_length * M_TO_MM:.2f}", "мм")

_minor_header("НАГНЕТАТЕЛЬНЫЙ ТРУБОПРОВОД")
p("Число Рейнольдса Re:", f"{ejector.reynolds_number:.0f}")
