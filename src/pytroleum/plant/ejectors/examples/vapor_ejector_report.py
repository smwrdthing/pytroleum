import matplotlib.pyplot as plt
from CoolProp import constants as CoolConst

from pytroleum.plant.ejectors.ejector import VaporEjector
from pytroleum.plant.ejectors.inputs import (OperationConditions,
                                             Requirements,
                                             ACTIVE, PASSIVE)
from pytroleum.plant.ejectors.utils import (_major_header, _minor_header,
                                            _minor_divider,
                                            print_row as p,
                                            PA_TO_MPA, KELVIN_TO_CELSIUS,
                                            KCAL_TO_J,
                                            M_TO_MM, M2_TO_MM2)

# ============================================================
# Исходные данные
# ============================================================

conditions = OperationConditions()

conditions.update_state(
    (CoolConst.PT_INPUTS, 7e6, 248),
    index=ACTIVE, upd_containers=True)
conditions.mass_flow_rate[ACTIVE] = 36.25           # кг/с

conditions.update_state(
    (CoolConst.PT_INPUTS, 2.5e6, 289),
    index=PASSIVE, upd_containers=True)
conditions.mass_flow_rate[PASSIVE] = 6.52           # кг/с

req = Requirements(
    num_stages=1,
    outlet_pressure=4e6,                            # Па
    outlet_diameter=0.325,                          # м
    active_inlet_diameter=0.325,                     # м
    passive_inlet_diameter=0.11,                    # м
)

# Термодинамические свойства активной среды
active_enthalpy = 1045.91 * KCAL_TO_J              # Дж/кг
active_entropy = 1.73 * KCAL_TO_J                  # Дж/(кг·К)

# Параметры расчёта сопла
entropy_lower_boundary = -0.34 * KCAL_TO_J         # Дж/(кг·К)
entropy_upper_boundary = 2.39 * KCAL_TO_J          # Дж/(кг·К)
enthalpy_lower_boundary = -91.89 * KCAL_TO_J       # Дж/кг
phi = 0.95

# Параметры расчёта смеси
entrainment_ratios = [0.1, 0.3, 0.5]
pressure_recovery_coefficient = 0.70
mach_number = 0.90
enthalpies_cyl_exit = [980.00 * KCAL_TO_J,
                       990.00 * KCAL_TO_J,
                       985.00 * KCAL_TO_J]         # Дж/кг
pressure_delta = 0.78

# Параметры расчёта геометрии
nozzle_expansion_ratio = 0.72
psi = 2.03

vapor_ejector = VaporEjector(conditions, req)
vapor_ejector.calculate_nozzle_params(
    active_enthalpy=active_enthalpy,
    active_entropy=active_entropy,
    entropy_lower_boundary=entropy_lower_boundary,
    entropy_upper_boundary=entropy_upper_boundary,
    enthalpy_lower_boundary=enthalpy_lower_boundary,
    phi=phi,
)
vapor_ejector.calculate_mixture_parameters(
    active_enthalpy=active_enthalpy,
    entrainment_ratios=entrainment_ratios,
    pressure_recovery_coefficient=pressure_recovery_coefficient,
    mach_number=mach_number,
    enthalpies_cyl_exit=enthalpies_cyl_exit,
    pressure_delta=pressure_delta,
)
vapor_ejector.calculate_geometry(
    nozzle_expansion_ratio=nozzle_expansion_ratio,
    psi=psi,
)

# ============================================================
# Вывод результатов расчёта эжектора
# ============================================================

_major_header("ИСХОДНЫЕ ДАННЫЕ")

_minor_header("АКТИВНАЯ СРЕДА (Эжектирующая)")
p("Давление на входе:",
  f"{conditions.pressure[ACTIVE] / PA_TO_MPA:.2f}", "МПа")
p("Температура:",
  f"{conditions.temperature[ACTIVE]} К "
  f"({conditions.temperature[ACTIVE] - KELVIN_TO_CELSIUS:.0f} °C)")
p("Массовый расход:", f"{conditions.mass_flow_rate[ACTIVE]:.2f}", "кг/с")
p("Энтальпия:", f"{active_enthalpy / KCAL_TO_J:.2f}", "ккал/кг")
p("Энтропия:", f"{active_entropy / KCAL_TO_J:.2f}", "ккал/(кг·°С)")
p("Диаметр трубопровода:", f"{req.active_inlet_diameter * M_TO_MM:.0f}", "мм")

_minor_header("ПАССИВНАЯ СРЕДА (Эжектируемая)")
p("Давление на входе:",
  f"{conditions.pressure[PASSIVE] / PA_TO_MPA:.2f}", "МПа")
p("Температура:",
  f"{conditions.temperature[PASSIVE]} К "
  f"({conditions.temperature[PASSIVE] - KELVIN_TO_CELSIUS:.0f} °C)")
p("Массовый расход:", f"{conditions.mass_flow_rate[PASSIVE]:.2f}", "кг/с")
p("Диаметр трубопровода:", f"{req.passive_inlet_diameter * M_TO_MM:.0f}", "мм")

_minor_header("ОБЩИЕ ПАРАМЕТРЫ")
p("Количество ступеней:", f"{req.num_stages}", "шт.")
p("Давление на выходе:", f"{req.outlet_pressure / PA_TO_MPA:.3f}", "МПа")
p("Диаметр выходного трубопровода:",
  f"{req.outlet_diameter * M_TO_MM:.0f}", "мм")

_minor_header("ПАРАМЕТРЫ РАСЧЁТА СОПЛА")
p("Энтропия на нижней пограничной кривой s':",
  f"{entropy_lower_boundary / KCAL_TO_J:.2f}", "ккал/(кг·°С)")
p("Энтропия на верхней пограничной кривой s'':",
  f"{entropy_upper_boundary / KCAL_TO_J:.2f}", "ккал/(кг·°С)")
p("Энтальпия на нижней пограничной кривой i':",
  f"{enthalpy_lower_boundary / KCAL_TO_J:.2f}", "ккал/кг")
p("Скоростной коэффициент сопла φ:", f"{phi:.2f}")

_minor_header("ПАРАМЕТРЫ РАСЧЁТА СМЕСИ")
p("Коэффициент восстановления давления:",
  f"{pressure_recovery_coefficient:.2f}")
p("Число Маха M3:", f"{mach_number:.2f}")
p("Коэффициент δ:", f"{pressure_delta:.2f}")

_minor_header("ПАРАМЕТРЫ РАСЧЁТА ГЕОМЕТРИИ")
p("Степень расширения сопла ϑ:", f"{nozzle_expansion_ratio:.2f}")
p("Коэффициент скорости сопла ψ:", f"{psi:.2f}")

# ============================================================

_major_header("РЕЗУЛЬТАТЫ РАСЧЁТА ЭЖЕКТОРА")

_minor_header("ОСНОВНЫЕ ПАРАМЕТРЫ ЭЖЕКЦИИ")
p("Коэффициент эжекции:", f"{vapor_ejector.entrainment_ratio:.4f}")
p("Степень сжатия:", f"{vapor_ejector.compression_ratio:.4f}")

_major_header("РЕЗУЛЬТАТЫ РАСЧЁТА СОПЛА")

p("Температура активной среды в сопле t1:",
  f"{vapor_ejector.temperature_nozzle_exit:.2f}", "К")
p("Давление газа в конце сопла p1:",
  f"{vapor_ejector.pressure_nozzle_exit / PA_TO_MPA:.2f}", "МПа")
p("Степень льдистости x(a):",
  f"{vapor_ejector.ice_quality_nozzle_exit:.4f}")
p("Скрытая теплота льдистого газа r(a):",
  f"{vapor_ejector.latent_heat_nozzle_exit / KCAL_TO_J:.2f}", "ккал/кг")
p("Энтальпия расширившейся среды i1(a):",
  f"{vapor_ejector.enthalpy_nozzle_exit / KCAL_TO_J:.2f}", "ккал/кг")
p("Скорость истечения газа из сопла w1:",
  f"{vapor_ejector.velocity_nozzle_exit:.2f}", "м/с")
p("Потери тепла в сопле hc:",
  f"{vapor_ejector.heat_loss_nozzle / KCAL_TO_J:.2f}", "ккал/кг")
p("Энтальпия газа в конце сопла с учётом потерь i1:",
  f"{vapor_ejector.enthalpy_nozzle_exit_actual / KCAL_TO_J:.2f}", "ккал/кг")
p("Степень льдистости в конце действительного расширения x1:",
  f"{vapor_ejector.ice_quality_nozzle_actual:.4f}")
p("Давление в критическом сечении сопла pкр:",
  f"{vapor_ejector.pressure_critical / PA_TO_MPA:.4f}", "МПа")
p("Температура в критическом сечении сопла tкр:",
  f"{vapor_ejector.temperature_critical:.2f}", "К")
p("Удельный объём насыщенного пара при давлении p1 v'':",
  f"{vapor_ejector.vapor_specific_vol:.5f}", "м³/кг")
p("Удельный объём льдистого газа на выходе из сопла v1:",
  f"{vapor_ejector.specific_volume_nozzle_exit:.5f}", "м³/кг")
p("Динамический напор эжектирующей струи H_дин:",
  f"{vapor_ejector.dynamic_head_nozzle / PA_TO_MPA:.2f}", "МПа")

_major_header("РАСЧЁТ ОСНОВНОГО ГЕОМЕТРИЧЕСКОГО ПАРАМЕТРА СТУПЕНИ")

_minor_divider()
print("Заданные коэффициенты эжекции:")
for i, q in enumerate(vapor_ejector.stage_entrainment_ratios, 1):
    p(f"  q({i}):", f"{q:.1f}")
_minor_divider()
print("Заданные энтальпии смеси в конце цилиндрического участка i3:")
for i, i3 in enumerate(vapor_ejector.stage_enthalpies_cyl_exit, 1):
    p(f"  i3({i}):", f"{i3 / KCAL_TO_J:.1f}", "ккал/кг")

_minor_divider()
print("Показатель адиабаты смеси в конце цилиндрического участка (сечение III):")
for i, k3 in enumerate(vapor_ejector.stage_adiabatic_indices, 1):
    p(f"  k3({i}):", f"{k3:.3f}")

_minor_divider()
print("Давление смеси в конце цилиндрического участка:")
for i, p3 in enumerate(vapor_ejector.stage_pressures_cyl_exit, 1):
    p(f"  p3({i}):", f"{p3 / PA_TO_MPA:.3f}", "МПа")

_minor_divider()
print("Парциальное давление активной среды в конце цилиндрического участка:")
for i, p3a in enumerate(vapor_ejector.stage_partial_pressures_active, 1):
    p(f"  p3_a({i}):", f"{p3a / PA_TO_MPA:.3f}", "МПа")

_minor_divider()
print("Газовая постоянная смеси:")
for i, R_mix in enumerate(vapor_ejector.stage_gas_constants_mixture, 1):
    p(f"  R_см({i}):", f"{R_mix:.2f}", "Дж/(кг·К)")

_minor_divider()
print("Температура смеси в конце цилиндрического участка T3:")
for i, t3 in enumerate(vapor_ejector.stage_temperatures_cyl_exit, 1):
    p(f"  T3({i}):", f"{t3:.2f} К  ({t3 - KELVIN_TO_CELSIUS:.2f} °C)")

_minor_divider()
print("Местная скорость звука в конце цилиндрического участка a(3):")
for i, a3 in enumerate(vapor_ejector.stage_sound_velocities, 1):
    p(f"  a(3)({i}):", f"{a3:.2f}", "м/с")

_minor_divider()
print("Скорость смеси в конце цилиндрического участка w(3):")
for i, w3 in enumerate(vapor_ejector.stage_mixture_velocities, 1):
    p(f"  w(3)({i}):", f"{w3:.2f}", "м/с")

_minor_divider()
print("Удельный объём смеси в конце цилиндрического участка v(3):")
for i, v3 in enumerate(vapor_ejector.stage_specific_volumes, 1):
    p(f"  v(3)({i}):", f"{v3:.4f}", "м³/кг")

_minor_divider()
print("Основной геометрический параметр ступени m:")
for i, m in enumerate(vapor_ejector.stage_geometric_params, 1):
    p(f"  m({i}):", f"{m:.2f}")

_minor_divider()
print("Давление смеси p(3):")
for i, p3 in enumerate(vapor_ejector.stage_mixture_pressures, 1):
    p(f"  p(3)({i}):", f"{p3 / PA_TO_MPA:.3f}", "МПа")

vapor_ejector.plot_mixture_pressure_vs_entrainment()
plt.show()

_major_header("ГЕОМЕТРИЧЕСКИЕ ПАРАМЕТРЫ СОПЛА")

p("Расчётная площадь выходного сечения F1*:",
  f"{vapor_ejector.nozzle_exit_area_theoretical * M2_TO_MM2:.2f}", "мм²")
p("Площадь критического сечения Fкр:",
  f"{vapor_ejector.nozzle_throat_area * M2_TO_MM2:.2f}", "мм²")
p("Диаметр критического сечения Dкр:",
  f"{vapor_ejector.nozzle_throat_diameter * M_TO_MM:.2f}", "мм")
p("Диаметр критического сечения с учётом погранслоя Dкр*:",
  f"{vapor_ejector.nozzle_throat_diameter_corrected * M_TO_MM:.2f}", "мм")
p("Действительная площадь выходного сечения F1:",
  f"{vapor_ejector.nozzle_exit_area * M2_TO_MM2:.2f}", "мм²")
p("Действительный диаметр выходного сечения D1:",
  f"{vapor_ejector.nozzle_exit_diameter * M_TO_MM:.2f}", "мм")
p("Диаметр сопла D' (= диаметр трубопровода активной среды):",
  f"{vapor_ejector.nozzle_diameter * M_TO_MM:.2f}", "мм")
p("Диаметр камеры разряжения D2 (= диаметр трубопровода пассивной среды):",
  f"{vapor_ejector.vacuum_chamber_diameter * M_TO_MM:.2f}", "мм")
p("Входной диаметр камеры смешения Dк':",
  f"{vapor_ejector.inlet_mixing_diameter * M_TO_MM:.2f}", "мм")
p("Диаметр камеры смешения Dк:",
  f"{vapor_ejector.mixing_diameter * M_TO_MM:.2f}", "мм")
p("Радиус кривизны камеры смешения Rк:",
  f"{vapor_ejector.curvature_radius * M_TO_MM:.2f}", "мм")
p("Длина камеры смешения Lк:",
  f"{vapor_ejector.mixing_length * M_TO_MM:.2f}", "мм")
