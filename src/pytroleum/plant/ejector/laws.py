import numpy as np
from scipy.constants import R as UNIVERSAL_GAS_CONST
from scipy.optimize import fsolve
from typing import TYPE_CHECKING, Iterable
if TYPE_CHECKING:
    from pytroleum.tdyna.CoolStub import AbstractState
else:
    from CoolProp import AbstractState


ISENTROPIC_EFFICIECNY = 0.8
MACH_GUESSE = 1


def mass_flow_rate(area: float, state: AbstractState,
                   efficiency=ISENTROPIC_EFFICIECNY):
    """Function for mass flow rate computation"""

    hcr = state.cpmass()/state.cvmass()
    hcr_inc = hcr+1
    hcr_dec = hcr-1

    R = UNIVERSAL_GAS_CONST/state.molar_mass()
    p = state.p()
    T = state.T()

    G = p*area/np.sqrt(T) * np.sqrt(
        hcr/R * (2/hcr_inc)**(hcr_inc/hcr_dec)
    )*np.sqrt(efficiency)

    return G


def isentropic_temperature_ratio(heat_capacity_ratio, mach_number):
    return 1 + (heat_capacity_ratio - 1)/2 * mach_number**2


def isentropic_pressure_ratio(heat_capacity_ratio, mach_number):

    hcr = heat_capacity_ratio
    hcr_dec = heat_capacity_ratio-1

    ratio = isentropic_temperature_ratio(hcr, mach_number)**(hcr/hcr_dec)

    return ratio


def isentropic_area_ratio(heat_capacity_ratio, mach_number):

    hcr_inc = heat_capacity_ratio + 1
    hcr_dec = heat_capacity_ratio - 1

    realtion = isentropic_temperature_ratio(heat_capacity_ratio, mach_number)

    ratio = np.sqrt(
        mach_number**-2 * (2 / hcr_inc * realtion)**(hcr_inc/hcr_dec)
    )

    return ratio

# Auxiliary functions


def _area_ratio_error(mach_number, area_ratio, heat_capacity_ratio):

    error = area_ratio - \
        isentropic_area_ratio(heat_capacity_ratio, mach_number)

    return error


def _mach_for(area_ratio, heat_capacity_ratio, mach_guesse=MACH_GUESSE):

    # This one probably belongs to Conditions class - if this is a common approach
    # to parameters determination in the ejector

    mach = fsolve(
        _area_ratio_error, mach_guesse,
        args=(area_ratio, heat_capacity_ratio))[-1]

    return mach
