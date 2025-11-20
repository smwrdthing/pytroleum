# Interfaces for Equation of State

from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from CoolStub import AbstractState
else:
    from CoolProp import AbstractState
import CoolProp
import CoolProp.constants as CoolConst

# Realised we don't need subclass for hydrocarbon mixture interface definition at all.
# in this case we're good to go with factory function. Though code for AbstractState
# subclassing is retained in case of future needs
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

# Constants
MOLE_FRACTION_SUM_TOL = 1e-3
GENERIC_HYDROCARBS = {
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
}


def factory_eos(composition: dict[str, float], backend: str = 'HEOS',
                with_state: None | Iterable = None) -> AbstractState:

    # should work both for mixtures and pures
    names = '&'.join(list(composition.keys()))

    # Initially I did not want to specify mole fractions immediately, but decided to do so
    mole_fractions = list(composition.values())

    # CoolProp does not enforce sum(mole_fractions) == 1, so we must do it
    if np.abs(np.sum(mole_fractions) - 1) > MOLE_FRACTION_SUM_TOL:
        msg = "Sum of mole fractions is not unity, provide valid composition"
        raise ValueError(msg)

    # If composition check is successfull, create interface and set mole fractions
    eos = AbstractState(backend, names)
    eos.set_mole_fractions(mole_fractions)

    # If some state is given jump to it
    if with_state is not None:
        # Iterable of three is expected as with_state, first element - key of input pair,
        # others — corresponding values. CoolProp should handle invalid with_state.
        eos.update(*with_state)

    return eos


def factory_hydrocarbs(composition: dict[str, float], backend: str = 'PR',
                       with_state: None | Iterable = None) -> AbstractState:

    # Note how for general factory default CoolProp backend is Helmholtz EOS and for
    # hydrocarbons it is Peng-Robinson EOS.

    # Here we want to obtain an interface of hydrocarbon gases. Some assertions with
    # regard to provided compostion should be done.
    names = list(composition.keys())
    if len(GENERIC_HYDROCARBS.intersection(names)) == 0:
        msg = ("No components associated with typical natural gas")
        raise ValueError(msg)

    # If names are valid we create an interface with call to a common eos factory
    hcmix = factory_eos(composition, backend, with_state)

    return hcmix


if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt

    # Let's see if this makes sense

    natural_gas_composition = {
        'Methane': 0.6,
        'Ethane': 0.1,
        'Propane': 0.1,

        'Nitrogen': 0.1,
        'CO2': 0.05,
        'CO': 0.04,
        'H2S': 0.01
    }

    # What if we change backend?
    hcm_eos_PR = factory_hydrocarbs(natural_gas_composition)
    hcm_eos_PR.build_phase_envelope("")
    PE = hcm_eos_PR.get_phase_envelope_data()

    fig, ax = plt.subplots()
    ax.set_title('Isopleth')
    ax.set_xlabel(r'T $\degree$C')
    ax.set_ylabel('Pressure, atm')
    ax.plot(np.array(PE.T)-273.15, np.array(PE.p)/1e5)
    ax.grid(True)

    # Trying PQ flash with const P
    print('\nPQ FLASH :: const P')
    print(50*'=')
    Q = np.arange(0, 1+0.1, 0.1)
    P = np.ones_like(Q)*40e5
    # Q from flash-calculations above
    T = []
    for pressure, quality in zip(P, Q):
        hcm_eos_PR.update(CoolConst.PQ_INPUTS, pressure, quality)
        T.append(hcm_eos_PR.T())

        print(50*'-')
        print(
            f'P = {pressure/1e5} bar; T = {T[-1]-273.15:.2f} C; Q = {quality*100:.2f} %'
        )

        ax.plot(T[-1]-273.15, pressure/1e5, 'b>')
    plt.show()
