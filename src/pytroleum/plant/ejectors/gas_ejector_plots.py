import matplotlib.pyplot as plt

from pytroleum.plant.ejectors.gas_ejector import GasEjector

from pytroleum.plant.ejectors.inputs import (ActiveMediumData,
                                             PassiveMediumData,
                                             CommonParams)

from pytroleum.plant.ejectors.utils import (
    KCAL_TO_J, KCAL_PER_KMOL_TO_J_PER_MOL,
    KGS_S_M2_TO_PA_S, KG_PER_KMOL_TO_KG_PER_MOL,
    PA_TO_MPA, KELVIN_TO_CELSIUS,
)

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

ejector = GasEjector(active, passive, common)
ejector.calculate(
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
    active.inlet_pressure / PA_TO_MPA,
    ejector.pressure_critical / PA_TO_MPA,
    ejector.pressure_cyl_section_exit / PA_TO_MPA,
    ejector.pressure_ejector_outlet / PA_TO_MPA,
]

temperatures = [
    active.temperature - KELVIN_TO_CELSIUS,
    ejector.temperature_nozzle_exit - KELVIN_TO_CELSIUS,
    ejector.temperature_cyl_section_exit - KELVIN_TO_CELSIUS,
    ejector.temperature_diffuser_exit - KELVIN_TO_CELSIUS,
]

velocities = [
    ejector.velocity_active_inlet,
    ejector.velocity_nozzle_exit,
    ejector.velocity_cyl_section_exit,
    ejector.velocity_ejector_outlet,
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
ax2.set_ylim(-50, 0)

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
