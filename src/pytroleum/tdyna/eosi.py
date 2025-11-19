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
        'HydrogenSulfide',

    ]

    def __init__(self, backend: str = 'HEOS', molar_compostion: Iterable[float] | None = None):

        # This one only requires backend name. Fluid - generic mixture of hydrocarbons with
        # specified composition.

        super().__init__(backend, '&'.join(self.__generic_hc_names))  # something like that

        if molar_compostion is not None:
            self.set_mole_fractions(molar_compostion)


if __name__ == "__main__":

    hcmix = Hydrocarbons()
