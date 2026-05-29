import matplotlib.pyplot as plt
from CoolProp import constants as CoolConst

from pytroleum.plant.ejectors.ejector import VaporEjector
from pytroleum.plant.ejectors.inputs import (OperationConditions,
                                             Requirements,
                                             ACTIVE, PASSIVE)
from pytroleum.plant.ejectors.utils import KCAL_TO_J, PA_TO_MPA

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
    active_inlet_diameter=0.325,                    # м
    passive_inlet_diameter=0.11,                    # м
)

active_enthalpy = 1045.91 * KCAL_TO_J              # Дж/кг
active_entropy = 1.73 * KCAL_TO_J                  # Дж/(кг·К)

entropy_lower_boundary = -0.34 * KCAL_TO_J         # Дж/(кг·К)
entropy_upper_boundary = 2.39 * KCAL_TO_J          # Дж/(кг·К)
enthalpy_lower_boundary = -91.89 * KCAL_TO_J       # Дж/кг
phi = 0.95

entrainment_ratios = [0.1, 0.3, 0.5]
pressure_recovery_coefficient = 0.70
mach_number = 0.90
enthalpies_cyl_exit = [980.00 * KCAL_TO_J,
                       990.00 * KCAL_TO_J,
                       985.00 * KCAL_TO_J]         # Дж/кг
pressure_delta = 0.78

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
# Построение графиков
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(vapor_ejector.stage_entrainment_ratios,
        [p / PA_TO_MPA for p in vapor_ejector.stage_mixture_pressures],
        marker='o')
ax.set_xlabel('Коэффициент эжекции q')
ax.set_ylabel('Давление смеси p(3), МПа')
ax.grid(True, linestyle='--', alpha=0.5)
for entrainment_ratio, pressure in zip(vapor_ejector.stage_entrainment_ratios,
                                       vapor_ejector.stage_mixture_pressures):
    ax.annotate(f'{pressure / PA_TO_MPA:.2f}',
                xy=(entrainment_ratio, pressure / PA_TO_MPA),
                textcoords='offset points', xytext=(0, 10),
                ha='center')
plt.tight_layout()
plt.show()
