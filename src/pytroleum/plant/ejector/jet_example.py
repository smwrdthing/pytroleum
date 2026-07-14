import numpy as np
from CoolProp import AbstractState
from CoolProp.CoolProp import PropsSI

from jet import (
    Loc,
    Phase,
    Requirements,
    design,
)

FLUID = "R141b"

T_gen = 95.0
T_evap = 8.0
Tc_star = 31.3

P_gen = PropsSI("P", "T", T_gen + 273.15, "Q", 1.0, FLUID)
P_evap = PropsSI("P", "T", T_evap + 273.15, "Q", 1.0, FLUID)
Pc_star = PropsSI("P", "T", Tc_star + 273.15, "Q", 1.0, FLUID)

SHAPE = (Phase.SIZE, Loc.SIZE)

eos = AbstractState("HEOS", FLUID)

req = Requirements(
    phase=eos,
    Pc_star=Pc_star,
    diameter_throat_nozzle=2.82e-3,  # Table 1, nozzle E
    diameter_exit_nozzle=5.10e-3,
    pressure=np.full(SHAPE, np.nan),
    temperature=np.full(SHAPE, np.nan),
)

req.pressure[Phase.P, Loc.IN] = P_gen
req.pressure[Phase.S, Loc.IN] = P_evap
req.temperature[Phase.P, Loc.IN] = T_gen + 273.15
req.temperature[Phase.S, Loc.IN] = T_evap + 273.15

design, operation_conditions = design(req)

req.report()
design.report()
operation_conditions.report()

# NOTE отношение площадей и коэффициент эжекции расходятся с данными в статье,
# NOTE нужно перепроверить реализацию алгоритма, найти ошибку

# NOTE такие скрипты кладём потом в examples, в пакете не оставляем
