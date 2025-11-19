# Interfaces for Equation of State

from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from CoolStub import AbstractState
else:
    from CoolProp import AbstractState
import CoolProp
import CoolProp.constants as CPconst


# Realised we don't need subclass for hydrocarbon mixture interface definition at all.
# in this case we're good to go with factory function. Though code for AbstractState subclassing
# is retained in case of future needs
#
# How to subclass AbstractState appropriately:
# class CertainState(AbstractState):

#     def __new__(cls, backend: str = 'HEOS', fluids: str):
#         # Must define __new__ instead of __init__ due to some cython shenanigans.

#         instance = super().__new__(cls, backend, '&'.join(
#             cls.__generic_hc_names))

#         return instance

#     def __init__(self, backend: str, fluids: str):
#         # call signature here, should repeat one in __new__ to silence pylance
#         pass

# Maybe a default generic composition?
GENERIC_HYDROCARBON_MIXTURE = [
    # Hydrocarbons
    'Methane',
    'Ethane',
    'Propane',
    'IsoButane',
    'n-Butane',
    'Isopentane',
    'n-Pentane',

    # Contaminants/Others
    'Nitrogen',
    'CarbonMonoxide',
    'CarbonDioxide',
    'HydrogenSulfide'
]

CP = __import__('CoolProp.CoolProp')


def hydrocarb_factory(composition: dict[str, float], backend: str = 'HEOS') -> AbstractState:

    # No assertions are imposed here

    hcm_name = '&'.join(list(composition.keys()))
    hcm_frac = list(composition.values())

    hcmix = AbstractState(backend, hcm_name)
    hcmix.set_mole_fractions(hcm_frac)

    return hcmix


if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt

    # Let's see if this makes sense

    natural_gas_composition = {
        'Methane': 0.6,
        'Ethane': 0.2,
        'Propane': 0.1,

        'Nitrogen': 0.05,
        'CO2': 0.05,
        # 'CO': 0.04,
        # 'H2S' : 0.01
    }

    # Check composition sanity
    assert (abs(np.sum(list(natural_gas_composition.values()))-1) < 1e-2)

    # NOTE on molar fractions : CoolProp does absolutely nothing to ensure sum(mole_fractins) = 1

    hcm_eos = hydrocarb_factory(natural_gas_composition)

    print('Hydrocarbon composition')
    print(80*'=')
    i = 1
    for flname, flx in zip(hcm_eos.fluid_names(), hcm_eos.get_mole_fractions()):
        print(f'{i} : {flname} :: {flx}')
        i += 1
    print(80*'=')

    # Not mandatory, might be useful for robustness
    # CP.set_config_double(CP.PHASE_ENVELOPE_STARTING_PRESSURE_PA, 1e4)

    # Trying some calculations
    hcm_eos.build_phase_envelope("")
    PE = hcm_eos.get_phase_envelope_data()

    fig, ax = plt.subplots()
    ax.set_title('Isopleth')
    ax.set_xlabel(r'T $\degree$C')
    ax.set_ylabel('Pressure, atm')
    plt.plot(np.array(PE.T)-273.15, np.array(PE.p)/1e5)
    ax.grid(True)

    # What if we change backend?
    hcm_eos_PR = hydrocarb_factory(natural_gas_composition, 'PR')
    hcm_eos_PR.build_phase_envelope("")
    PE = hcm_eos_PR.get_phase_envelope_data()
    ax.plot(np.array(PE.T)-273.15, np.array(PE.p)/1e5)

    hcm_eos_SRK = hydrocarb_factory(natural_gas_composition, 'SRK')
    hcm_eos_SRK.build_phase_envelope("")
    PE = hcm_eos_SRK.get_phase_envelope_data()
    ax.plot(np.array(PE.T)-273.15, np.array(PE.p)/1e5)

    ax.legend(['HEOS', 'PR', 'SRK'])

    # Different backends provide different isopleths

    # Compute some specific points

    # HEOS interface refuses to perform computations in regions where Q < 1
    # for some reasons, maybe biary interaction parameters are off? IDK

    P = np.array([10, 20, 30, 40, 50, 60])*1e5
    T = np.ones_like(P)*(-15+273.15)

    # Trying PT-flash with const T
    print('\nPT FLASH :: const T :: k^')
    print(80*'=')
    Q = []
    for pressure, temperature in zip(P, T):
        hcm_eos_PR.update(CPconst.PT_INPUTS, pressure, temperature)
        Q.append(hcm_eos_PR.Q())

        print(80*'-')
        print(
            f'P = {pressure/1e5} bar; T = {temperature-273.15} C; Q = {Q[-1]*100} %'
        )
        ax.plot(temperature-273.15, pressure/1e5, 'k^')
    # Vapor quality is off??
    print(80*'='+'\n')

    # Trying PQ-flash with variable PT
    print('\nPQ FLASH :: var P :: r>')
    print(80*'=')
    Q = np.linspace(0, 1, len(P))Ля
    T = []
    for pressure, quality in zip(P, Q):
        hcm_eos_PR.update(CPconst.PQ_INPUTS, pressure, quality)
        T.append(hcm_eos_PR.T())

        print(80*'-')
        print(
            f'P = {pressure/1e5} bar; T = {T[-1]-273.15:.2f} C; Q = {quality*100:.2f} %'
        )

        ax.plot(T[-1]-273.15, pressure/1e5, 'r>')
    print(80*'='+'\n')

    # Trying PQ flash with const P
    print('\nPQ FLASH :: const P :: b>')
    print(80*'=')
    P = np.ones_like(Q)*40e5
    # Q from flash-calculations above
    T = []
    for pressure, quality in zip(P, Q):
        hcm_eos_PR.update(CPconst.PQ_INPUTS, pressure, quality)
        T.append(hcm_eos_PR.T())

        print(80*'-')
        print(
            f'P = {pressure/1e5} bar; T = {T[-1]-273.15:.2f} C; Q = {quality*100:.2f} %'
        )

        ax.plot(T[-1]-273.15, pressure/1e5, 'b>')
