"""
Hydrocyclone inverse problem for a range of target cut sizes.

For each given d50' value the hydrocyclone diameter Dc is found such that
the required separation is achieved. Results are printed as a table and
displayed as a series of plots.
"""

import numpy as np
import matplotlib.pyplot as plt

from pytroleum.plant.solid_cyclone.inputs import (
    PhysicalProperties,
    OperationConditions,
    SizeDistribution,
)
from pytroleum.plant.solid_cyclone.models import RietemaHydrocyclone
from pytroleum.plant.solid_cyclone.geometry import (
    CycloneDesign,
    HydrocycloneDiameters,
    HydrocycloneLengths,
    RIETEMA_DIAMETER_PROPORTIONS,
    RIETEMA_LENGTH_PROPORTIONS,
    RIETEMA_CONE_ANGLE,
)
from pytroleum.plant.solid_cyclone.efficiency import (
    calculate_reduced_grade_efficiency,
    calculate_reduced_total_efficiency,
    calculate_total_efficiency,
)
from pytroleum.plant.solid_cyclone.inverse import find_Dc_by_cut_size
from pytroleum.plant.solid_cyclone.utils import _minor_divider, _major_divider

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CUT_SIZE_TARGETS = np.array([1, 2, 3, 5, 7, 8, 10])   # µm

MODEL_NAME = 'Rietema'

properties = PhysicalProperties(solid_density=1500)

conditions = OperationConditions(
    feed_volumetric_concentration=0.00033,
    mode='Q',
    feed_volumetric_flow_rate=12.0 / (1000 * 60),   # L/min → m³/s
)

size_dist = SizeDistribution(
    # type: ignore[call-overload]
    particle_diameters=np.linspace(1e-6, 200e-6, 500),
    k=10.9918e-6,
    n=0.9187,
)

cut_size_targets_m = CUT_SIZE_TARGETS * 1e-6

print("Inverse problem: finding hydrocyclone diameter for target cut size")
print(f"Model: {MODEL_NAME}")
print(f"Target d50': {CUT_SIZE_TARGETS} µm")
print(f"Flow rate: Q = {conditions.feed_volumetric_flow_rate*6e4:.1f} L/min")

rietema_hydrocyclone = RietemaHydrocyclone(
    '', CycloneDesign(10e-3, RIETEMA_DIAMETER_PROPORTIONS,
                      RIETEMA_LENGTH_PROPORTIONS, RIETEMA_CONE_ANGLE),
    conditions)

hydrocyclones = []
for d50_target in cut_size_targets_m:
    find_Dc_by_cut_size(
        cut_size_target=d50_target,
        hydrocyclone=rietema_hydrocyclone,
        properties=properties,
    )
    hc = RietemaHydrocyclone('', rietema_hydrocyclone.design, conditions)
    hc.calculate_from_flow_rate(properties)
    hydrocyclones.append(hc)


def _efficiencies(hc):
    reduced_grade = calculate_reduced_grade_efficiency(
        size_dist.particle_diameters, hc.reduced_cut_size, 'plitt', hc.m, hc.alpha)

    reduced_total = calculate_reduced_total_efficiency(
        size_dist.particle_diameters, reduced_grade, size_dist.k, size_dist.n)

    total = calculate_total_efficiency(reduced_total, hc.water_flow_ratio)

    return reduced_total, total


print()
_major_divider()
header = (
    f"{'d50_target, µm':>16} {'d50_found, µm':>15} {'Dc, mm':>8} "
    f"{'ΔP, kPa':>10} {'Rw':>8} {'E_T, %':>8} {'E_T\', %':>8}"
)
print(header)
_minor_divider()
for target, hc in zip(cut_size_targets_m, hydrocyclones):
    et_reduced, et_total = _efficiencies(hc)
    print(
        f"{target*1e6:>16.1f} "
        f"{hc.reduced_cut_size*1e6:>15.2f} "
        f"{hc.design.diameters[HydrocycloneDiameters.C]*1e3:>8.2f} "
        f"{hc.conditions.pressure_drop/1e3:>10.2f} "
        f"{hc.water_flow_ratio:>8.4f} "
        f"{et_total*100:>8.1f} "
        f"{et_reduced*100:>8.1f}"
    )

# ΔP
_, ax = plt.subplots(figsize=(7, 5))
ax.plot(CUT_SIZE_TARGETS, [hc.conditions.pressure_drop / 1e3 for hc in hydrocyclones],
        color='steelblue', linewidth=1.8, marker='o', markersize=6)
ax.set_xlabel("$d_{50}'$, µm", fontsize=10)
ax.set_ylabel('ΔP, kPa', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Rw
_, ax = plt.subplots(figsize=(7, 5))
ax.plot(CUT_SIZE_TARGETS, [hc.water_flow_ratio for hc in hydrocyclones],
        color='steelblue', linewidth=1.8, marker='o', markersize=6)
ax.set_xlabel("$d_{50}'$, µm", fontsize=10)
ax.set_ylabel('$R_w$', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

efficiencies = [_efficiencies(hc) for hc in hydrocyclones]

# E_T
_, ax = plt.subplots(figsize=(7, 5))
ax.plot(CUT_SIZE_TARGETS, [et[1] * 100 for et in efficiencies],
        color='steelblue', linewidth=1.8, marker='o', markersize=6)
ax.set_xlabel("$d_{50}'$, µm", fontsize=10)
ax.set_ylabel('$E_T$, %', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# E_T'
_, ax = plt.subplots(figsize=(7, 5))
ax.plot(CUT_SIZE_TARGETS, [et[0] * 100 for et in efficiencies],
        color='steelblue', linewidth=1.8, marker='o', markersize=6)
ax.set_xlabel("$d_{50}'$, µm", fontsize=10)
ax.set_ylabel("$E_T'$, %", fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Diameters
_, ax = plt.subplots(figsize=(7, 5))
ax.plot(CUT_SIZE_TARGETS,
        [hc.design.diameters[HydrocycloneDiameters.C] * 1e3 for hc in hydrocyclones],
        color='slategray', linewidth=1.8, label='$D_c$')
ax.plot(CUT_SIZE_TARGETS,
        [hc.design.diameters[HydrocycloneDiameters.I] * 1e3 for hc in hydrocyclones],
        color='cornflowerblue', linewidth=1.8, label='$D_i$')
ax.plot(CUT_SIZE_TARGETS,
        [hc.design.diameters[HydrocycloneDiameters.O] * 1e3 for hc in hydrocyclones],
        color='salmon', linewidth=1.8, label='$D_o$')
ax.plot(CUT_SIZE_TARGETS,
        [hc.design.diameters[HydrocycloneDiameters.U] * 1e3 for hc in hydrocyclones],
        color='mediumseagreen', linewidth=1.8, label='$D_u$')
ax.set_xlabel("$d_{50}'$, µm", fontsize=10)
ax.set_ylabel('Diameter, mm', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Lengths
_, ax = plt.subplots(figsize=(7, 5))
ax.plot(CUT_SIZE_TARGETS,
        [hc.design.lengths[HydrocycloneLengths.T] * 1e3 for hc in hydrocyclones],
        color='slategray', linewidth=1.8, label='$L$')
ax.plot(CUT_SIZE_TARGETS,
        [hc.design.lengths[HydrocycloneLengths.C] * 1e3 for hc in hydrocyclones],
        color='cornflowerblue', linewidth=1.8, label='$L_c$')
ax.plot(CUT_SIZE_TARGETS,
        [hc.design.lengths[HydrocycloneLengths.V] * 1e3 for hc in hydrocyclones],
        color='salmon', linewidth=1.8, label='$l$')
ax.set_xlabel("$d_{50}'$, µm", fontsize=10)
ax.set_ylabel('Length, mm', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
