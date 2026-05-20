from pytroleum.plant.ejectors.vapor_ejector import VaporEjector
from pytroleum.plant.ejectors.inputs import (ActiveMediumData,
                                             PassiveMediumData,
                                             CommonParams)

from pytroleum.plant.ejectors.utils import (_major_header, _minor_header,
                                            print_row as p,
                                            PA_TO_MPA, KELVIN_TO_CELSIUS,
                                            KCAL_TO_J, KCAL_PER_KMOL_TO_J_PER_MOL,
                                            KGS_S_M2_TO_PA_S, KG_PER_KMOL_TO_KG_PER_MOL,
                                            KG_PER_MOL_TO_G_PER_MOL, J_TO_KJ, M_TO_MM)

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

# ============================================================
# Вывод результатов
# ============================================================

_major_header("ИСХОДНЫЕ ДАННЫЕ")

_minor_header("АКТИВНАЯ СРЕДА (Эжектирующая)")
p("Массовый расход:", f"{active.mass_flow:.2f}", "кг/с")
p("Температура:",
  f"{active.temperature} К ({active.temperature - KELVIN_TO_CELSIUS:.0f} °C)")
p("Давление на входе:", f"{active.inlet_pressure / PA_TO_MPA:.3f}", "МПа")
p("Энтальпия:", f"{active.enthalpy / KCAL_TO_J:.2f}", "ккал/кг")
p("Энтропия:", f"{active.entropy / KCAL_TO_J:.4f}", "ккал/(кг·°С)")
p("Удельный объём:", f"{active.specific_volume:.3f}", "м³/кг")
p("Плотность:", f"{active.density:.3f}", "кг/м³")
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
p("Давление на входе:", f"{passive.inlet_pressure / PA_TO_MPA:.3f}", "МПа")
p("Энтальпия:", f"{passive.enthalpy / KCAL_TO_J:.2f}", "ккал/кг")
p("Энтропия:", f"{passive.entropy / KCAL_TO_J:.4f}", "ккал/(кг·°С)")
p("Удельный объём:", f"{passive.specific_volume:.3f}", "м³/кг")
p("Плотность:", f"{passive.density:.3f}", "кг/м³")
p("Динамическая вязкость:",
  f"{passive.dynamic_viscosity / KGS_S_M2_TO_PA_S:.8f}", "кгс·с/м²")
p("Диаметр трубопровода:", f"{passive.inlet_diameter * M_TO_MM:.0f}", "мм")
p("Молекулярная масса:",
  f"{passive.molecular_mass / KG_PER_KMOL_TO_KG_PER_MOL:.2f}", "кг/кмоль")
p("Теплоёмкость:",
  f"{passive.heat_capacity / KCAL_PER_KMOL_TO_J_PER_MOL:.3f}", "ккал/(кмоль·°С)")

_minor_header("ОБЩИЕ ПАРАМЕТРЫ")
p("Количество ступеней:", f"{common.num_stages}", "шт.")
p("Давление на выходе:", f"{common.outlet_pressure / PA_TO_MPA:.3f}", "МПа")
p("Диаметр выходного трубопровода:",
  f"{common.outlet_diameter * M_TO_MM:.0f}", "мм")

_major_header("РЕЗУЛЬТАТЫ РАСЧЁТА СОПЛА")

p("Энтропия на нижней пограничной кривой s(a)':",
  f"{ejector.entropy_lower_boundary / J_TO_KJ:.4f}", "кДж/(кг·°С)")
p("Энтропия на верхней пограничной кривой s(a)'':",
  f"{ejector.entropy_upper_boundary / J_TO_KJ:.4f}", "кДж/(кг·°С)")
p("Энтальпия на нижней пограничной кривой i(a)':",
  f"{ejector.enthalpy_lower_boundary / J_TO_KJ:.4f}", "кДж/кг")

p("Температура активной среды в сопле t1:",
  f"{ejector.temperature_nozzle_exit:.2f}", "К")
p("Давление газа в конце сопла p1:",
  f"{ejector.pressure_nozzle_exit / PA_TO_MPA:.3f}", "МПа")
p("Скрытая степень льдистости газа  в выходном сечении сопла r(a):",
  f"{ejector.latent_heat_nozzle_exit / J_TO_KJ:.2f}", "кДж/кг")
p("Степень льдистости x(a):",
  f"{ejector.ice_quality_nozzle_exit:.4f}")
p("Энтальпия расширившейся среды i1(a):",
  f"{ejector.enthalpy_nozzle_exit / J_TO_KJ:.2f}", "кДж/кг")
p("Скорость истечения газа из сопла w1:",
  f"{ejector.velocity_nozzle_exit:.2f}", "м/с")
p("Потери тепла в сопле hc:",
  f"{ejector.heat_loss_nozzle / J_TO_KJ:.2f}", "кДж/кг")
p("Энтальпия газа в конце сопла с учётом потерь i1:",
  f"{ejector.enthalpy_nozzle_exit_actual / J_TO_KJ:.2f}", "кДж/кг")
p("Степень льдистости в конце действительного расширения x1:",
  f"{ejector.ice_quality_nozzle_actual:.4f}")
