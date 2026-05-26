from pytroleum.plant.ejectors.equations import (calculate_gas_constant,
                                                calculate_specific_heat_capacity)

from pytroleum.plant.ejectors.gas_ejector import GasEjector
from pytroleum.plant.ejectors.inputs import (ActiveMediumData,
                                             PassiveMediumData,
                                             CommonParams)

from pytroleum.plant.ejectors.utils import (_major_header, _minor_header,
                                            print_row as p,
                                            PA_TO_MPA, KELVIN_TO_CELSIUS,
                                            KCAL_TO_J, KCAL_PER_KMOL_TO_J_PER_MOL,
                                            KGS_S_M2_TO_PA_S, KG_PER_KMOL_TO_KG_PER_MOL,
                                            KG_PER_MOL_TO_G_PER_MOL, J_TO_KJ, M_TO_MM,
                                            )

# ============================================================
# Исходные данные
# ============================================================

# Активная среда (эжектирующая)
active = ActiveMediumData(
    mass_flow=36.25,                                     # кг/с
    temperature=248,                                     # К
    inlet_pressure=7e6,                                  # Па
    enthalpy=1045.91 * KCAL_TO_J,                       # ккал/кг → Дж/кг
    # ккал/(кг·°С) → Дж/(кг·К)
    entropy=1.73 * KCAL_TO_J,
    specific_volume=0.01,                                # м³/кг
    density=98.89,                                       # кг/м³
    dynamic_viscosity=0.00000099 * KGS_S_M2_TO_PA_S,   # кгс·с/м² → Па·с
    inlet_diameter=0.33,                                 # м
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
    # ккал/(кг·°С) → Дж/(кг·К)
    entropy=1.69 * KCAL_TO_J,
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

gas_ejector = GasEjector(active, passive, common)
gas_ejector.calculate(
    s=2,
    mixture_density=56.05,
    pressure_recovery_coefficient=0.8,
    psi=2.14,
    opening_angle=8,
    mixture_dynamic_viscosity=0.0000012 * KGS_S_M2_TO_PA_S,
)

# ============================================================
# Вывод результатов расчёта эжектора
# ============================================================

_major_header("ИСХОДНЫЕ ДАННЫЕ")

_minor_header("АКТИВНАЯ СРЕДА (Эжектирующая)")
p("Массовый расход:", f"{active.mass_flow:.2f}", "кг/с")
p("Температура:",
  f"{active.temperature} К ({active.temperature - KELVIN_TO_CELSIUS:.0f} °C)")
p("Давление на входе:", f"{active.inlet_pressure / PA_TO_MPA:.2f}", "МПа")
p("Энтальпия:", f"{active.enthalpy / KCAL_TO_J:.2f}", "ккал/кг")
p("Энтропия:", f"{active.entropy / KCAL_TO_J:.2f}", "ккал/(кг·°С)")
p("Удельный объём:", f"{active.specific_volume:.2f}", "м³/кг")
p("Плотность:", f"{active.density:.2f}", "кг/м³")
p("Динамическая вязкость:",
  f"{active.dynamic_viscosity / KGS_S_M2_TO_PA_S:.8f}", "кгс·с/м²")
p("Диаметр трубопровода:", f"{active.inlet_diameter * M_TO_MM:.0f}", "мм")
p("Молекулярная масса:",
  f"{active.molecular_mass / KG_PER_KMOL_TO_KG_PER_MOL:.2f}", "кг/кмоль")
p("Теплоёмкость:",
  f"{active.heat_capacity / KCAL_PER_KMOL_TO_J_PER_MOL:.3f}", "ккал/(кмоль·°С)")

_minor_header("ПАССИВНАЯ СРЕДА (Эжектируемая)")
p("Массовый расход:", f"{passive.mass_flow:.2f}", "кг/с")
p("Температура:",
  f"{passive.temperature} К ({passive.temperature - KELVIN_TO_CELSIUS:.0f} °C)")
p("Давление на входе:", f"{passive.inlet_pressure / PA_TO_MPA:.2f}", "МПа")
p("Энтальпия:", f"{passive.enthalpy / KCAL_TO_J:.2f}", "ккал/кг")
p("Энтропия:", f"{passive.entropy / KCAL_TO_J:.2f}", "ккал/(кг·°С)")
p("Удельный объём:", f"{passive.specific_volume:.2f}", "м³/кг")
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
p("Давление на выходе:", f"{common.outlet_pressure / PA_TO_MPA:.2f}", "МПа")
p("Диаметр выходного трубопровода:",
  f"{common.outlet_diameter * M_TO_MM:.0f}", "мм")

_major_header("РЕЗУЛЬТАТЫ РАСЧЁТА ЭЖЕКТОРА")

R_active = calculate_gas_constant(active.molecular_mass)
R_passive = calculate_gas_constant(passive.molecular_mass)
Cp_active = calculate_specific_heat_capacity(
    active.heat_capacity, active.molecular_mass)
Cp_passive = calculate_specific_heat_capacity(
    passive.heat_capacity, passive.molecular_mass)

_minor_header("ОСНОВНЫЕ ПАРАМЕТРЫ ЭЖЕКЦИИ")
p("Степень сжатия:", f"{gas_ejector.compression_ratio:.4f}")
p("Коэффициент эжекции:", f"{gas_ejector.entrainment_ratio:.4f}")
p("m1 (участок струи до стенки):", f"{gas_ejector.m1:.4f}")
p("m (основной геометрический параметр):", f"{gas_ejector.m:.4f}")
p("n:", f"{gas_ejector.n:.4f}")
p("Напор эжектора без диффузора:",
  f"{gas_ejector.ejector_head_no_diff / PA_TO_MPA:.3f}", "МПа")

_minor_header("ГАЗОДИНАМИЧЕСКИЕ ПАРАМЕТРЫ")
p("Газовая постоянная активной среды R_a:", f"{R_active:.2f}", "Дж/(кг·К)")
p("Газовая постоянная пассивной среды R_n:", f"{R_passive:.2f}", "Дж/(кг·К)")
p("Удельная теплоёмкость активной среды Cp_a:",
  f"{Cp_active:.2f}", "Дж/(кг·К)")
p("Удельная теплоёмкость пассивной среды Cp_n:",
  f"{Cp_passive:.2f}", "Дж/(кг·К)")
p("Показатель адиабаты k:", f"{gas_ejector.adiabatic_index:.4f}")
p("Критическое отношение давлений β:",
  f"{gas_ejector.critical_pressure_ratio:.6f}")

_minor_header("ДАВЛЕНИЯ")
p("Динамический напор на выходе из сопла (I-I):",
  f"{gas_ejector.dynamic_head_nozzle_exit / PA_TO_MPA:.3f}", "МПа")
p("Напор эжектора без диффузора:",
  f"{gas_ejector.ejector_head_no_diff / PA_TO_MPA:.3f}", "МПа")
p("Критическое давление P_кр:",
  f"{gas_ejector.pressure_critical / PA_TO_MPA:.3f}", "МПа")
p("Давление в конце цилиндрического участка (III-III):",
  f"{gas_ejector.pressure_cyl_section_exit / PA_TO_MPA:.3f}", "МПа")
p("Давление за диффузором:",
  f"{gas_ejector.pressure_ejector_outlet / PA_TO_MPA:.3f}", "МПа")

_minor_header("СКОРОСТИ")
p("Скорость активной среды в трубопроводе:",
  f"{gas_ejector.velocity_active_inlet:.2f}", "м/с")
p("Скорость пассивной среды в трубопроводе:",
  f"{gas_ejector.velocity_passive_inlet:.2f}", "м/с")
p("Скорость истечения газа из сопла (w1):",
  f"{gas_ejector.velocity_nozzle_exit:.2f}", "м/с")
p("Скорость газа в конце смесительного участка (w3):",
  f"{gas_ejector.velocity_cyl_section_exit:.2f}", "м/с")
p("Скорость на выходе из эжектора (w4):",
  f"{gas_ejector.velocity_ejector_outlet:.2f}", "м/с")

_minor_header("ТЕМПЕРАТУРЫ")
p("Температура в критическом сечении сопла (t1):",
  f"{gas_ejector.temperature_nozzle_exit:.2f} К "
  f"({gas_ejector.temperature_nozzle_exit - KELVIN_TO_CELSIUS:.0f} °C)")
p("Температура в конце цилиндрического участка (t3):",
  f"{gas_ejector.temperature_cyl_section_exit:.2f} К "
  f"({gas_ejector.temperature_cyl_section_exit - KELVIN_TO_CELSIUS:.0f} °C)")
p("Температура на выходе из диффузора (t4):",
  f"{gas_ejector.temperature_diffuser_exit:.2f} К "
  f"({gas_ejector.temperature_diffuser_exit - KELVIN_TO_CELSIUS:.0f} °C)")

_minor_header("ГЕОМЕТРИЧЕСКИЕ РАЗМЕРЫ")
p("Площадь выходного сечения сопла F1:",
  f"{gas_ejector.nozzle_exit_area:.4f}", "м²")
p("Диаметр выходного сечения сопла D1:",
  f"{gas_ejector.nozzle_exit_diameter * M_TO_MM:.2f}", "мм")
p("Площадь узкой части сопла Fкр:",
  f"{gas_ejector.nozzle_throat_area:.4f}", "м²")
p("Диаметр узкой части сопла Dкр:",
  f"{gas_ejector.nozzle_throat_diameter * M_TO_MM:.2f}", "мм")
p("Площадь сечения смесительного участка F3:",
  f"{gas_ejector.mixing_section_area:.4f}", "м²")
p("Диаметр смесительного участка D3:",
  f"{gas_ejector.mixing_section_diameter * M_TO_MM:.2f}", "мм")
p("Площадь конечного сечения диффузора F4:",
  f"{gas_ejector.diffuser_exit_area:.4f}", "м²")
p("Длина струи Lx'':",
  f"{gas_ejector.jet_length * M_TO_MM:.2f}", "мм")
p("Длина смесительного участка Lсм:",
  f"{gas_ejector.mixing_section_length * M_TO_MM:.2f}", "мм")
p("Расстояние от сопла до цилиндрического участка L1:",
  f"{gas_ejector.nozzle_to_inlet_distance * M_TO_MM:.2f}", "мм")
p("Длина цилиндрического участка L2:",
  f"{gas_ejector.cylinder_length * M_TO_MM:.2f}", "мм")
p("Длина диффузора L3:",
  f"{gas_ejector.diffuser_length * M_TO_MM:.2f}", "мм")

_minor_header("НАГНЕТАТЕЛЬНЫЙ ТРУБОПРОВОД")
p("Число Рейнольдса Re:", f"{gas_ejector.reynolds_number:.0f}")

# NOTE код с конкретными расчётами лучше кидать либо в подпапку examples в ejectors
# NOTE либо в examples всей библиотеки
