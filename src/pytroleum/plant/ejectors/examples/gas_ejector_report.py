from CoolProp import constants as CoolConst

from pytroleum.plant.ejectors.equations import (calculate_gas_constant,
                                                calculate_specific_heat_capacity)
from pytroleum.plant.ejectors.gas_ejector import GasEjector
from pytroleum.plant.ejectors.inputs import (OperationConditions,
                                             Requirements,
                                             ACTIVE, PASSIVE)
from pytroleum.plant.ejectors.utils import (_major_header, _minor_header,
                                            print_row as p,
                                            PA_TO_MPA, KELVIN_TO_CELSIUS,
                                            KGS_S_M2_TO_PA_S,
                                            KG_PER_MOL_TO_G_PER_MOL,
                                            J_TO_KJ, M_TO_MM,
                                            )

# ============================================================
# Исходные данные
# ============================================================

conditions = OperationConditions()

conditions.update_state(
    (CoolConst.PT_INPUTS, 7e6, 248),
    index=ACTIVE, upd_containers=True)
conditions.mass_flow_rate[ACTIVE] = 36.25       # кг/с

conditions.update_state(
    (CoolConst.PT_INPUTS, 2.5e6, 289),
    index=PASSIVE, upd_containers=True)
conditions.mass_flow_rate[PASSIVE] = 6.52       # кг/с

req = Requirements(
    num_stages=1,
    outlet_pressure=4e6,                        # Па
    outlet_diameter=0.325,                      # м
    active_inlet_diameter=0.33,                 # м
    passive_inlet_diameter=0.11,                # м
)

gas_ejector = GasEjector(conditions, req)
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
p("Массовый расход:", f"{conditions.mass_flow_rate[ACTIVE]:.2f}", "кг/с")
p("Температура:",
  f"{conditions.temperature[ACTIVE]} К "
  f"({conditions.temperature[ACTIVE] - KELVIN_TO_CELSIUS:.0f} °C)")
p("Давление на входе:",
  f"{conditions.pressure[ACTIVE] / PA_TO_MPA:.2f}", "МПа")
p("Удельный объём:", f"{1 / conditions.phase[ACTIVE].rhomass():.4f}", "м³/кг")
p("Плотность:", f"{conditions.phase[ACTIVE].rhomass():.2f}", "кг/м³")
p("Динамическая вязкость:",
  f"{conditions.phase[ACTIVE].viscosity():.2e}", "Па·с")
p("Диаметр трубопровода:", f"{req.active_inlet_diameter * M_TO_MM:.0f}", "мм")
p("Молярная масса:",
  f"{conditions.phase[ACTIVE].molar_mass() * KG_PER_MOL_TO_G_PER_MOL:.2f}", "г/моль")
p("Теплоёмкость:", f"{conditions.phase[ACTIVE].cpmass():.2f}", "Дж/(кг·К)")

_minor_header("ПАССИВНАЯ СРЕДА (Эжектируемая)")
p("Массовый расход:", f"{conditions.mass_flow_rate[PASSIVE]:.2f}", "кг/с")
p("Температура:",
  f"{conditions.temperature[PASSIVE]} К "
  f"({conditions.temperature[PASSIVE] - KELVIN_TO_CELSIUS:.0f} °C)")
p("Давление на входе:",
  f"{conditions.pressure[PASSIVE] / PA_TO_MPA:.2f}", "МПа")
p("Удельный объём:", f"{1 / conditions.phase[PASSIVE].rhomass():.4f}", "м³/кг")
p("Плотность:", f"{conditions.phase[PASSIVE].rhomass():.2f}", "кг/м³")
p("Динамическая вязкость:",
  f"{conditions.phase[PASSIVE].viscosity():.2e}", "Па·с")
p("Диаметр трубопровода:", f"{req.passive_inlet_diameter * M_TO_MM:.0f}", "мм")
p("Молярная масса:",
  f"{conditions.phase[PASSIVE].molar_mass() * KG_PER_MOL_TO_G_PER_MOL:.2f}", "г/моль")
p("Теплоёмкость:", f"{conditions.phase[PASSIVE].cpmass():.2f}", "Дж/(кг·К)")

_minor_header("ОБЩИЕ ПАРАМЕТРЫ")
p("Количество ступеней:", f"{req.num_stages}", "шт.")
p("Давление на выходе:", f"{req.outlet_pressure / PA_TO_MPA:.2f}", "МПа")
p("Диаметр выходного трубопровода:",
  f"{req.outlet_diameter * M_TO_MM:.0f}", "мм")

_major_header("РЕЗУЛЬТАТЫ РАСЧЁТА ЭЖЕКТОРА")

R_active = calculate_gas_constant(conditions.phase[ACTIVE].molar_mass())
R_passive = calculate_gas_constant(conditions.phase[PASSIVE].molar_mass())
Cp_active = conditions.phase[ACTIVE].cpmass()
Cp_passive = conditions.phase[PASSIVE].cpmass()

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
