import matplotlib.pyplot as plt
from CoolProp import constants as CoolConst

from pytroleum.plant.ejectors.ejector import GasEjector
from pytroleum.plant.ejectors.inputs import (OperationConditions,
                                             Requirements,
                                             ACTIVE, PASSIVE)
from pytroleum.plant.ejectors.utils import (
    KGS_S_M2_TO_PA_S, PA_TO_MPA, KELVIN_TO_CELSIUS,
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
# Данные по сечениям
# ============================================================

sections = [0, 1, 2, 3]
labels = ['0', '1', '3', '4']

pressures = [
    conditions.pressure[ACTIVE] / PA_TO_MPA,
    gas_ejector.pressure_critical / PA_TO_MPA,
    gas_ejector.pressure_cyl_section_exit / PA_TO_MPA,
    gas_ejector.pressure_ejector_outlet / PA_TO_MPA,
]

temperatures = [
    conditions.temperature[ACTIVE] - KELVIN_TO_CELSIUS,
    gas_ejector.temperature_nozzle_exit - KELVIN_TO_CELSIUS,
    gas_ejector.temperature_cyl_section_exit - KELVIN_TO_CELSIUS,
    gas_ejector.temperature_diffuser_exit - KELVIN_TO_CELSIUS,
]

velocities = [
    gas_ejector.velocity_active_inlet,
    gas_ejector.velocity_nozzle_exit,
    gas_ejector.velocity_cyl_section_exit,
    gas_ejector.velocity_ejector_outlet,
]

# ============================================================
# Построение графиков
# ============================================================

fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(sections, pressures, marker='o')
ax1.set_title('Давление в эжекторе по сечениям', pad=10)
ax1.set_ylabel('Давление, МПа')
ax1.set_xlabel('Сечение')
ax1.set_xticks(sections)
ax1.set_xticklabels(labels)
ax1.grid(True, linestyle='--', alpha=0.5)
for s, val in zip(sections, pressures):
    ax1.annotate(f'{val:.2f}', (s, val),
                 textcoords='offset points', xytext=(0, 10), ha='center')
ax1.set_ylim(0, 8)

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.plot(sections, temperatures, marker='o')
ax2.set_title('Температура в эжекторе по сечениям', pad=10)
ax2.set_ylabel('Температура, °C')
ax2.set_xlabel('Сечение')
ax2.set_xticks(sections)
ax2.set_xticklabels(labels)
ax2.grid(True, linestyle='--', alpha=0.5)
for s, val in zip(sections, temperatures):
    ax2.annotate(f'{val:.1f}', (s, val),
                 textcoords='offset points', xytext=(0, 10), ha='center')

fig3, ax3 = plt.subplots(figsize=(8, 5))
ax3.plot(sections, velocities, marker='o')
ax3.set_title('Скорость в эжекторе по сечениям', pad=10)
ax3.set_ylabel('Скорость, м/с')
ax3.set_xlabel('Сечение')
ax3.set_xticks(sections)
ax3.set_xticklabels(labels)
ax3.grid(True, linestyle='--', alpha=0.5)
for s, val in zip(sections, velocities):
    ax3.annotate(f'{val:.1f}', (s, val),
                 textcoords='offset points', xytext=(0, 20), ha='center')
ax3.set_ylim(0, 450)

plt.show()
