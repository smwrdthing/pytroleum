import numpy as np
import CoolProp
from CoolProp import AbstractState
from CoolProp.CoolProp import PropsSI

from jet_huang import (
    Design,
    Loc,
    OperationConditions,
    Phase,
    _LAST_LOC,
    _LAST_PHASE,
    solve_dimensions,
)

FLUID = "R141b"

T_gen = 95.0
T_evap = 8.0

P_gen = PropsSI("P", "T", T_gen + 273.15, "Q", 1.0, FLUID)
P_evap = PropsSI("P", "T", T_evap + 273.15, "Q", 1.0, FLUID)

print(f"Primary   inlet: T = {T_gen} C,  P = {P_gen/1e5:.4f} bar")
print(f"Secondary inlet: T = {T_evap} C,  P = {P_evap/1e5:.4f} bar")

SHAPE = (_LAST_PHASE, _LAST_LOC)

design = Design(
    diameter=np.full(SHAPE, np.nan),
    area=np.full(SHAPE, np.nan),
    length=np.full(SHAPE, np.nan),
)

design.diameter[Phase.JET, Loc.THROAT] = 2.82e-3
design.area[Phase.JET, Loc.THROAT] = np.pi / 4 * \
    design.diameter[Phase.JET, Loc.THROAT] ** 2

design.diameter[Phase.JET, Loc.EXIT_NOZZLE] = 4.0e-3
design.area[Phase.JET, Loc.EXIT_NOZZLE] = np.pi / 4 * \
    design.diameter[Phase.JET, Loc.EXIT_NOZZLE] ** 2

design.diameter[Phase.MIX, Loc.AFTERMIX] = 19.0e-3
design.area[Phase.MIX, Loc.AFTERMIX] = np.pi / 4 * \
    design.diameter[Phase.MIX, Loc.AFTERMIX] ** 2

_eos = AbstractState("HEOS", FLUID)
_eos.specify_phase(CoolProp.iphase_gas)

req = OperationConditions(
    phase=_eos,
    mass_flow_rate=np.zeros(2),
    pressure=np.full(SHAPE, np.nan),
    temperature=np.full(SHAPE, np.nan),
)

req.pressure[Phase.JET, Loc.INLET] = P_gen
req.pressure[Phase.CARRY, Loc.INLET] = P_evap
req.temperature[Phase.JET, Loc.INLET] = T_gen + 273.15
req.temperature[Phase.CARRY, Loc.INLET] = T_evap + 273.15

solve_dimensions(req, design)

print("\n" + "=" * 60)
print("OPERATING CONDITIONS")
print("=" * 60)
req.report()

print("\n" + "=" * 60)
print("GEOMETRY")
print("=" * 60)
design.report()

omega = req.mass_flow_rate[Phase.CARRY] / req.mass_flow_rate[Phase.JET]
print("\n" + "=" * 60)
print(f"Entrainment ratio  omega = ms/mp = {omega:.4f}")
print(f"Compression ratio  Pc/Pe         = "
      f"{req.pressure[Phase.MIX, Loc.DRAIN] / P_evap:.4f}")
print("=" * 60)
