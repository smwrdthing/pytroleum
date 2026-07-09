import numpy as np
import CoolProp
from CoolProp import AbstractState
from CoolProp.CoolProp import PropsSI

from jet_huang import (
    Design,
    Loc,
    Phase,
    Requirements,
    solve_dimensions,
)

FLUID = "R141b"

T_gen = 95.0
T_evap = 8.0
Tc_star = 31.3

P_gen = PropsSI("P", "T", T_gen + 273.15, "Q", 1.0, FLUID)
P_evap = PropsSI("P", "T", T_evap + 273.15, "Q", 1.0, FLUID)
Pc_star = PropsSI("P", "T", Tc_star + 273.15, "Q", 1.0, FLUID)

SHAPE = (Phase.SIZE, Loc.SIZE)

design = Design(
    diameter=np.full(SHAPE, np.nan),
    area=np.full(SHAPE, np.nan),
    length=np.full(SHAPE, np.nan),
)

# Table 1, nozzle E
design.diameter[Phase.P, Loc.TH] = 2.82e-3
design.area[Phase.P, Loc.TH] = np.pi / 4 * \
    design.diameter[Phase.P, Loc.TH] ** 2

design.diameter[Phase.P, Loc.EX] = 5.10e-3
design.area[Phase.P, Loc.EX] = np.pi / 4 * \
    design.diameter[Phase.P, Loc.EX] ** 2

_eos = AbstractState("HEOS", FLUID)
_eos.specify_phase(CoolProp.iphase_gas)

req = Requirements(
    phase=_eos,
    pressure=np.full(SHAPE, np.nan),
    temperature=np.full(SHAPE, np.nan),
)

req.pressure[Phase.P, Loc.IN] = P_gen
req.pressure[Phase.S, Loc.IN] = P_evap
req.temperature[Phase.P, Loc.IN] = T_gen + 273.15
req.temperature[Phase.S, Loc.IN] = T_evap + 273.15

conditions = solve_dimensions(req, design, Pc_star)

req.report()
design.report()
conditions.report()

# NOTE отношение площадей и коэффициент эжекции расходятся с данными в статье,
# NOTE нужно перепроверить реализацию алгоритма, найти ошибку

# NOTE такие скрипты кладём потом в examples, в пакете не оставляем
