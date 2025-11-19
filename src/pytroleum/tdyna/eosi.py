from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from CoolStub import AbstractState
else:
    from CoolProp import AbstractState

# TODO: figure out __cinit__() issue, create if __name__ == '__main__'
# for quick tests


class Hydrocarbons(AbstractState):

    __generic_hc_names: list[str] = [
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

    def __new__(cls, backend: str = 'HEOS'):
        # Must define __new__ instead of __init__ due to some cython shenanigans.

        instance = super().__new__(cls, backend, '&'.join(  # type: ignore
            cls.__generic_hc_names))

        return instance

    def __init__(self, backend: str):
        # call signature here, should repeat one in __new__ to silence pylance
        pass


if __name__ == "__main__":

    hcmix = Hydrocarbons('HEOS')
    airmix = AbstractState('HEOS', 'Air.mix')
