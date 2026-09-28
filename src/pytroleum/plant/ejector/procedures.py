import numpy as np
from scipy.constants import R as UNIVERSAL_GAS_CONST
from scipy.optimize import fsolve
from typing import TYPE_CHECKING, Iterable
if TYPE_CHECKING:
    from pytroleum.tdyna.CoolStub import AbstractState
else:
    from CoolProp import AbstractState
import CoolProp.constants as CoolConst


from pytroleum.plant.ejector.locator import Stream, Place, ANY, ALL
from pytroleum.plant.ejector.design import Design
from pytroleum.plant.ejector.operation import Conditions
from pytroleum.plant.ejector import laws
import pytroleum.plant.ejector.interface as ifc


MACH_GUESSE = 1


def HuangChang1D(design: ifc.Design, requirements: ifc.Requirements
                 ) -> tuple[ifc.Design, ifc.Conditions]:
    """This function implements design procedure after Huang et al.

    For details refer to:
    "A 1-D analysis of ejector performance"
    International Journal of Refrigeration          B. J. Huang
    vol 22                                          J. M. Chang
    May 4, 1999                                  V. A. Petrenko
    """

    # Procedure is presented with scheme in the paper, function executes all steps from
    # said scheme, whenever useful steps are referred to in comments for development and
    # maintenance convenience

    conditions = Conditions()
    conditions.adopt(requirements)

    conditions.adopt_state_from(Place.INLET)

    # Step 1
    conditions.flow_through(design.area[Place.THROAT], whose=Stream.PRIMPARY)

    # Step 2
    conditions.nozzle_mach_for(design)
    conditions.nozzle_pressure()

    # Step 3
    conditions.shock_pressure()

    # initial guesse (obviously wrong)
    design.area[Place.CONST] = design.area[Place.NOZZLE]
    while True:

        # Step 4

        break

        while True:

            break

    return design, conditions
