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
from pytroleum.plant.solid_cyclone.models import (
    RietemaHydrocyclone,
    BaseHydrocyclone,
)
from pytroleum.plant.solid_cyclone.geometry import (
    RIETEMA_DIAMETER_PROPORTIONS,
    RIETEMA_LENGTH_PROPORTIONS,
    RIETEMA_CONE_ANGLE,
)
from pytroleum.plant.solid_cyclone.inverse import find_Dc_by_cut_size

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CUT_SIZE_TARGETS = np.array([1, 2, 3, 5, 7, 8, 10])   # µm

HYDROCYCLONE_CLS: type[BaseHydrocyclone] = RietemaHydrocyclone

MODEL_NAME = 'Rietema'

properties = PhysicalProperties(solid_density=1500)

conditions = OperationConditions(
    feed_volumetric_concentration=0.00033,
    mode='Q',
    feed_volumetric_flow_rate=12.0 / (1000 * 60),   # L/min → m³/s
)

size_dist = SizeDistribution(
    particle_diameters=np.linspace(1e-6, 200e-6, 500),
    k=10.9918e-6,
    n=0.9187,
)

cut_size_targets_m = CUT_SIZE_TARGETS * 1e-6

print("Inverse problem: finding hydrocyclone diameter for target cut size")
print(f"Model: {MODEL_NAME}")
print(f"Target d50': {CUT_SIZE_TARGETS} µm")
print(f"Flow rate: Q = {conditions.feed_volumetric_flow_rate*6e4:.1f} L/min")

results = []
for d50_target in cut_size_targets_m:
    res = find_Dc_by_cut_size(
        cut_size_target=d50_target,
        conditions=conditions,
        diameter_proportions=RIETEMA_DIAMETER_PROPORTIONS,
        length_proportions=RIETEMA_LENGTH_PROPORTIONS,
        cone_angle=RIETEMA_CONE_ANGLE,
        hydrocyclone_cls=HYDROCYCLONE_CLS,
        properties=properties,
        size_dist=size_dist,
    )
    results.append(res)

x_um = CUT_SIZE_TARGETS

print(f"\n{'='*75}")
header = (
    f"{'d50_target, µm':>16} {'d50_found, µm':>15} {'Dc, mm':>8} "
    f"{'ΔP, kPa':>10} {'Rw':>8} {'E_T, %':>8} {'E_T\', %':>8}"
)
print(header)
print('-' * 75)
for target, r in zip(cut_size_targets_m, results):
    print(
        f"{target*1e6:>16.1f} "
        f"{r['reduced_cut_size']*1e6:>15.2f} "
        f"{r['Dc']*1e3:>8.2f} "
        f"{r['pressure_drop']/1e3:>10.2f} "
        f"{r['water_flow_ratio']:>8.4f} "
        f"{r['total_efficiency']*100:>8.1f} "
        f"{r['reduced_total_efficiency']*100:>8.1f}"
    )

# ΔP
_, ax = plt.subplots(figsize=(7, 5))
ax.plot(x_um, [r['pressure_drop'] / 1e3 for r in results],
        color='steelblue', linewidth=1.8, marker='o', markersize=6)
ax.set_xlabel("$d_{50}'$, µm", fontsize=10)
ax.set_ylabel('ΔP, kPa', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Rw
_, ax = plt.subplots(figsize=(7, 5))
ax.plot(x_um, [r['water_flow_ratio'] for r in results],
        color='steelblue', linewidth=1.8, marker='o', markersize=6)
ax.set_xlabel("$d_{50}'$, µm", fontsize=10)
ax.set_ylabel('$R_w$', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# E_T
_, ax = plt.subplots(figsize=(7, 5))
ax.plot(x_um, [r['total_efficiency'] * 100 for r in results],
        color='steelblue', linewidth=1.8, marker='o', markersize=6)
ax.set_xlabel("$d_{50}'$, µm", fontsize=10)
ax.set_ylabel('$E_T$, %', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# E_T'
_, ax = plt.subplots(figsize=(7, 5))
ax.plot(x_um, [r['reduced_total_efficiency'] * 100 for r in results],
        color='steelblue', linewidth=1.8, marker='o', markersize=6)
ax.set_xlabel("$d_{50}'$, µm", fontsize=10)
ax.set_ylabel("$E_T'$, %", fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Diameters
_, ax = plt.subplots(figsize=(7, 5))
ax.plot(x_um, [r['Dc'] * 1e3 for r in results],
        color='slategray', linewidth=1.8, label='$D_c$')
ax.plot(x_um, [r['Di'] * 1e3 for r in results],
        color='cornflowerblue', linewidth=1.8, label='$D_i$')
ax.plot(x_um, [r['Do'] * 1e3 for r in results],
        color='salmon', linewidth=1.8, label='$D_o$')
ax.plot(x_um, [r['Du'] * 1e3 for r in results],
        color='mediumseagreen', linewidth=1.8, label='$D_u$')
ax.set_xlabel("$d_{50}'$, µm", fontsize=10)
ax.set_ylabel('Diameter, mm', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Lengths
_, ax = plt.subplots(figsize=(7, 5))
ax.plot(x_um, [r['L'] * 1e3 for r in results],
        color='slategray', linewidth=1.8, label='$L$')
ax.plot(x_um, [r['Lc'] * 1e3 for r in results],
        color='cornflowerblue', linewidth=1.8, label='$L_c$')
ax.plot(x_um, [r['vortex_finder_length'] * 1e3 for r in results],
        color='salmon', linewidth=1.8, label='$l$')
ax.set_xlabel("$d_{50}'$, µm", fontsize=10)
ax.set_ylabel('Length, mm', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
