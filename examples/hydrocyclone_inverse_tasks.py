"""
Hydrocyclone inverse problem: two tasks.

Task 1 — find body diameter Dc for a target reduced cut size d50'.
Task 2 — find body diameter Dc for a target total efficiency E_T.
"""

import numpy as np

from pytroleum.plant.solid_cyclone.inputs import (
    PhysicalProperties,
    OperationConditions,
    SizeDistribution,
)
from pytroleum.plant.solid_cyclone.geometry import (
    CycloneDesign,
    RIETEMA_DIAMETER_PROPORTIONS,
    RIETEMA_LENGTH_PROPORTIONS,
    RIETEMA_CONE_ANGLE,
)
from pytroleum.plant.solid_cyclone.models import RietemaHydrocyclone
from pytroleum.plant.solid_cyclone.efficiency import (
    calculate_reduced_grade_efficiency,
    calculate_reduced_total_efficiency,
    calculate_total_efficiency,
)
from pytroleum.plant.solid_cyclone.inverse import (
    find_Dc_by_cut_size,
    find_Dc_by_efficiency,
    TOL_RELATIVE,
)
from pytroleum.plant.solid_cyclone.utils import _minor_divider, _major_divider

# ---------------------------------------------------------------------------
# Shared inputs
# ---------------------------------------------------------------------------

properties = PhysicalProperties(solid_density=1500)

conditions = OperationConditions(
    feed_volumetric_concentration=0.00033,
    mode='Q',
    feed_volumetric_flow_rate=12.0 / (1000 * 60),
)
size_dist = SizeDistribution(
    # type: ignore[call-overload]
    particle_diameters=np.linspace(1e-6, 200e-6, 500),
    k=10.9918e-6,
    n=0.9187,
)

rietema_hydrocyclone = RietemaHydrocyclone(
    '', CycloneDesign(10e-3, RIETEMA_DIAMETER_PROPORTIONS,
                      RIETEMA_LENGTH_PROPORTIONS, RIETEMA_CONE_ANGLE),
    conditions, properties)

# ---------------------------------------------------------------------------
# Task 1: find Dc for target d50'
# ---------------------------------------------------------------------------

cut_size_target = 5e-6
print("TASK 1: FIND Dc FOR TARGET REDUCED CUT SIZE d50'")
_minor_divider()
find_Dc_by_cut_size(
    cut_size_target=cut_size_target,
    hydrocyclone=rietema_hydrocyclone,
)

rietema_hydrocyclone.design.summary()

et1_reduced_grade = calculate_reduced_grade_efficiency(
    size_dist.particle_diameters, rietema_hydrocyclone.reduced_cut_size,
    'plitt', rietema_hydrocyclone.m, rietema_hydrocyclone.alpha)

et1_reduced = calculate_reduced_total_efficiency(
    size_dist.particle_diameters, et1_reduced_grade, size_dist.k, size_dist.n)

et1_total = calculate_total_efficiency(
    et1_reduced, rietema_hydrocyclone.water_flow_ratio)

print(
    f"Volumetric flow rate  Q = "
    f"{rietema_hydrocyclone.conditions.feed_volumetric_flow_rate*6e4:.3f} L/min")
print(
    f"Pressure drop ΔP = "
    f"{rietema_hydrocyclone.conditions.pressure_drop/1e3:.2f} kPa")
print(f"Water flow ratio Rw = {rietema_hydrocyclone.water_flow_ratio:.4f}")
print(f"Reduced cut size d50'= {rietema_hydrocyclone.reduced_cut_size*1e6:.2f} µm"
      f" (target: {cut_size_target*1e6:.2f} µm)")
print(f"Reduced total efficiency E_T'= {et1_reduced*100:.1f} %")
print(f"Total efficiency E_T = {et1_total*100:.2f} %")
reduced_cut_size_1 = rietema_hydrocyclone.reduced_cut_size

print()
_major_divider()
print()

# ---------------------------------------------------------------------------
# Task 2: find Dc for target E_T
# ---------------------------------------------------------------------------

efficiency_target = 0.9
print("TASK 2: FIND Dc FOR TARGET TOTAL EFFICIENCY E_T")
_minor_divider()
find_Dc_by_efficiency(
    efficiency_target=efficiency_target,
    hydrocyclone=rietema_hydrocyclone,
    size_dist=size_dist,
)
rietema_hydrocyclone.design.summary()

et2_reduced_grade = calculate_reduced_grade_efficiency(
    size_dist.particle_diameters, rietema_hydrocyclone.reduced_cut_size,
    'plitt', rietema_hydrocyclone.m, rietema_hydrocyclone.alpha)

et2_reduced = calculate_reduced_total_efficiency(
    size_dist.particle_diameters, et2_reduced_grade, size_dist.k, size_dist.n)

et2_total = calculate_total_efficiency(
    et2_reduced, rietema_hydrocyclone.water_flow_ratio)

print(
    f"Volumetric flow rate Q = "
    f"{rietema_hydrocyclone.conditions.feed_volumetric_flow_rate*6e4:.3f} L/min")
print(
    f"Pressure drop ΔP = "
    f"{rietema_hydrocyclone.conditions.pressure_drop/1e3:.2f} kPa")
print(f"Water flow ratio Rw = {rietema_hydrocyclone.water_flow_ratio:.4f}")
print(
    f"Reduced cut size d50' = "
    f"{rietema_hydrocyclone.reduced_cut_size*1e6:.2f} µm")
print(f"Reduced total efficiency E_T'= {et2_reduced*100:.1f} %")
print(f"Total efficiency E_T  = {et2_total*100:.2f} %"
      f"  (target: {efficiency_target*100:.2f} %)")

print()
_major_divider()
print()

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

print("VERIFICATION OF TASK 1 (d50')")
_minor_divider()
rel_err1 = abs(reduced_cut_size_1 - cut_size_target) / cut_size_target
print(f"d50' (found)  = {reduced_cut_size_1*1e6:.4f} µm")
print(f"d50' (target) = {cut_size_target*1e6:.4f} µm")
print(f"Relative error: {rel_err1:.2e}")
if rel_err1 <= TOL_RELATIVE:
    print("Task 1 converged")
else:
    print("Task 1 did NOT converge — error exceeds tolerance")

print()
_major_divider()
print()

print("VERIFICATION OF TASK 2 (E_T)")
_minor_divider()
rel_err2 = abs(et2_total - efficiency_target) / efficiency_target
print(f"E_T (found)  = {et2_total*100:.2f} %")
print(f"E_T (target) = {efficiency_target*100:.2f} %")
print(f"Relative error: {rel_err2:.2e}")
if rel_err2 <= TOL_RELATIVE:
    print("Task 2 converged")
else:
    print("Task 2 did NOT converge — error exceeds tolerance")

print()
_major_divider()
print()
