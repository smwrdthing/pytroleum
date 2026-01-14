# Interfaces for Equation of State
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, overload
import numpy as np
from numpy import float64
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pytroleum.tdyna.CoolStub import AbstractState  # type: ignore
else:
    from CoolProp.CoolProp import AbstractState
import CoolProp.CoolProp as CP
import CoolProp.constants as CoolConst

# Large TODO : common facade for different thermodynamic libraries
# Specifically neqsim is very promising, definitely try to incorporate,
# algorithms from thermo might be useful too

MOLE_FRACTION_SUM_TOL = 1e-3
CRUDE_OIL = "CrudeOil", "CRUDEOIL"
_MESSAGE_UNSUPPORTED_DERIVATIVE = "Specified partial derivative is not supported"
_MESSAGE_UNSUPPORTED_INPUT_PAIR = "Provided input pair is not supported"


class AbstractStateImitator(ABC):

    # NOTE :
    # Main purpose is to represent petroleum fluids with CoolProp-like interface,
    # Should support PUmass and DmassT inputs for dynamic modelling
    #
    # Should read some papers on petroleum fluids modelling

    def __init__(self, backend: str | None, fluid: str | None) -> None:

        self._backend = backend
        self._fluid = fluid

        self._molar_mass: float
        self._mole_fractions: Iterable[float]

        self._heat_capacity_isochoric: float
        self._heat_capacity_isobaric: float
        self._dynamic_viscosity: float
        self._thermal_conductivity: float
        self._pressure: float
        self._temperature: float
        self._density: float
        self._mass_specific_energy: float
        self._molar_mass: float

    def molar_mass(self):
        """Returns molar mass of fluid."""
        return self._molar_mass

    def cvmass(self):
        """Returns isochoric heat capacity of fluid."""
        return self._heat_capacity_isochoric

    def cpmass(self):
        """Returns isochoric heat capacity of fluid."""
        return self._heat_capacity_isobaric

    def p(self):
        """Returns pressure."""
        return self._pressure

    def T(self):
        """Returns temperature."""
        return self._temperature

    def rhomass(self):
        """Returns density."""
        return self._density

    def umass(self):
        """Returns mass-based specific internal energy."""
        return self._mass_specific_energy

    def viscosity(self):
        return self._dynamic_viscosity

    def conductivity(self):
        return self._thermal_conductivity

    def set_mole_fractions(self, mole_fractions: Iterable[float]):
        """Sets new molar compositin of fluid."""
        self._mole_fractions = mole_fractions

    def get_mole_fractions(self) -> Iterable[float]:
        """Returns molar composition of fluid."""
        return self._mole_fractions

    # All imitators should contain implemented partial derivatives for
    # energy and density along with procedures for parameters refreshment with
    # PT inputs.

    def _valid_input_pair(self, pair_key):
        """Checks validity of provided input pair for state specification."""
        return pair_key == CoolConst.PT_INPUTS

    def _valid_partial_derivative(
            self, of_parameter_key: int,
            with_respect_to_key: int, holding_const_key: int) -> bool:
        """Checks validity of requested partial derivative."""

        # NOTE : logic should be checked later

        are_valid_keys = (
            (of_parameter_key == CoolConst.iUmass or
             of_parameter_key == CoolConst.iDmass) and
            with_respect_to_key == CoolConst.iT and
            holding_const_key == CoolConst.iP)

        return are_valid_keys

    @abstractmethod
    def first_partial_deriv(
            self, of_parameter_key: int,
            with_respect_to_key: int,
            holding_const_key: int
    ) -> None:
        """Returns requested partial derivative.

        Parameters
        ----------
        of_parameter_key
            integer key of parameter for which derivatvie is calculated.
        with_respect_to_key
            integer key of parameter with repect to which derivative is calculated.
        holding_const_key
            integer key of parameter to hold constand for derivative calculations.
        """

        # Important NOTE :
        # We are developing this primarly for dynamic modelling now, so we need
        # only a limited subset of partial derivatives. This might look a bit ugly,
        # but it grants universal approach for all phases in sdyna.
        #
        # We need partial derivative of internal energy with respect to temperature for
        # constant pressure and partial derivative of density with respect to temperature
        # for constant pressure, so we only implement them, validation is abstracted out
        # in separate method.

        # We could put key validation here and call parent class method from subclasses,
        # but I think calling validator in subclasses is more explicit

        return

    def _compute_density(self):
        """Computes density for current state."""
        return

    def _compute_viscosity(self):
        """Computes viscosity for current state."""
        return

    def _compute_heat_capacity_isochoric(self):
        """Computes heat capacity for current state."""
        return

    def _compute_mass_specific_energy(self):
        """Computes mass specific internal energy for current state."""
        return

    def _run_computations(self):
        self._compute_density()
        self._compute_viscosity()
        self._compute_heat_capacity_isochoric()
        self._compute_mass_specific_energy()

    @abstractmethod
    def update(self, input_pair_key: int,
               first_keyed_parameter: float,
               second_keyed_parameter: float) -> None:
        """Compute parameters of state specified with provided input pair.

        Parameters
        ----------
        input_pair_key
            integer key to specify which values are provided as imput pair.
        first_keyed_parameter
            value of first parameter corresponding to input pair.
        second_keyed_parameter
            value if second parameter corresponding to input pair.
        """
        return


class CrudeOilHardcoded(AbstractStateImitator):

    # NOTE :
    # This is to model crude oil with constant parameters

    _BACKEND: str = "CrudeOilHardcoded"
    _DENSITY: float = 830
    _DYNAMIC_VISCOSITY: float = 6e-3
    _HEAT_CAPACITY: float = 2300
    _THERMAL_CONDUCTIVITY: float = 0.6

    def __init__(self) -> None:
        super().__init__(self._BACKEND, CRUDE_OIL[0])
        self._density = self._DENSITY
        self._dynamic_viscosity = self._DYNAMIC_VISCOSITY
        self._heat_capacity_isochoric = self._HEAT_CAPACITY
        self._heat_capacity_isobaric = self._HEAT_CAPACITY
        self._thermal_conductivity = self._THERMAL_CONDUCTIVITY

    def default(self) -> None:
        """Sets default harcoded parameters for oil."""
        self._density = self._DENSITY
        self._dynamic_viscosity = self._DYNAMIC_VISCOSITY
        self._heat_capacity_isochoric = self._HEAT_CAPACITY
        self._heat_capacity_isobaric = self._HEAT_CAPACITY

    def change(self, new_density: float | None = None,
               new_visocsity: float | None = None,
               new_heat_capacity: float | None = None) -> None:
        """Sets new hardcoded parameters for oil."""
        if new_density is not None:
            self._density = new_density
        if new_visocsity is not None:
            self._dynamic_viscosity = new_visocsity
        if new_heat_capacity is not None:
            self._heat_capacity_isochoric = new_heat_capacity
            self._heat_capacity_isobaric = new_heat_capacity

    def _compute_mass_specific_energy(self):
        self._mass_specific_energy = self._heat_capacity_isochoric*self._temperature

    def first_partial_deriv(
            self, of_parameter_key: int,
            with_respect_to_key: int,
            holding_const_key: int) -> float | None:

        supported_derivative = self._valid_partial_derivative(
            of_parameter_key, with_respect_to_key, holding_const_key)

        if supported_derivative:
            if of_parameter_key == CoolConst.iUmass:
                return self._heat_capacity_isochoric
            if of_parameter_key == CoolConst.iDmass:
                return 0
        else:
            raise KeyError(_MESSAGE_UNSUPPORTED_DERIVATIVE)

    def update(self, input_pair_key: int, first_keyed_parameter: float,
               second_keyed_parameter: float):
        # shouldn't do anything crazy for this particular class, all parameters should
        # remain const
        if self._valid_input_pair(input_pair_key):
            self._pressure = first_keyed_parameter
            self._temperature = second_keyed_parameter
        self._compute_mass_specific_energy()


class CrudeOilReferenced(AbstractStateImitator):

    # NOTE :
    # This is to model some crude oil properties with simple models when reference
    # values are provided

    _BACKEND: str = "CrudeOilReferenced"

    # Maybe move this to dataclass for reference parameters?
    _DEFAULT_PRESSURE = 101_330
    _DEFAULT_TEMPERATURE = 15+273.15

    def __init__(self, reference: CrudeOilRefernceData) -> None:
        super().__init__(self._BACKEND, CRUDE_OIL[0])
        self.reference = reference
        self._prepare_density_model()
        self._prepare_viscosity_model()
        self._prepare_conductivity_model()

        # Set default state and compute parameters
        self._pressure = self._DEFAULT_PRESSURE
        self._temperature = self._DEFAULT_TEMPERATURE
        self._compute_density()
        self._compute_viscosity()

        self._compute_specific_gravity()
        self._compute_api_gravity()

        self._compute_heat_capacity_isochoric()
        self._compute_mass_specific_energy()

    def _prepare_density_model(self):
        """Prepares linear model for density, computes isobaric expansivity coefficient
        from given reference for further density computations."""
        self._expansivity_isobaric = -1/self.reference.mean_density*(
            self.reference.density[1]-self.reference.density[0])/(
                self.reference.temperature[1]-self.reference.temperature[0])

    def _prepare_viscosity_model(self):
        """Prepares exponential model for viscosity, computes exponent coefficient and
        linear coefficient from given reference for further viscosity computations."""
        T1, T2 = self.reference.temperature
        mu1, mu2 = self.reference.viscosity
        self._viscosity_model_power = T1*T2/(T1-T2)*np.log(mu2/mu1)
        self._viscosity_model_coeff = mu1*np.exp(
            -self._viscosity_model_power/T1)

    def _prepare_conductivity_model(self):

        lam1, lam2 = self.reference.thermal_conductivity
        T1, T2 = self.reference.temperature

        self._thermal_conductivity_slope = (lam2-lam1)/(T2-T1)

    # NOTE :
    # methods for specific and api gravity computations are only called once during
    # initialization with appropriate default temperature for correltions to work

    def _compute_specific_gravity(self, temperature=15+273.15):
        """Computes oil's specific gravity for specified temperature (default is 15C)."""
        self._specific_gravity = density_to_specific_gravity(
            self._density, temperature)

    def _compute_api_gravity(self):
        """Computes oil's API from current value of specific gravity."""
        self._api_gravity = specific_to_api_gravity(self._specific_gravity)

    def _compute_density(self):
        """Computes density of oil with linear model."""
        self._density = (
            self.reference.density[0] -
            self._expansivity_isobaric * self.reference.mean_density *
            (self._temperature-self.reference.temperature[0]))

    def _compute_viscosity(self):
        """Computes viscosity of oil with exponential model."""
        self._dynamic_viscosity = self._viscosity_model_coeff*np.exp(
            self._viscosity_model_power/self._temperature)

    def _compute_heat_capacity_isochoric(self) -> None:
        """Computes isochoric heat capacity with specific-gravity based correlation."""
        # Correlations with hardcoded coefficients for now, refine later
        temperature_celcius = self._temperature - 273.15
        self._heat_capacity_isochoric = (
            (2*temperature_celcius-1429)*self._specific_gravity +
            2.67*temperature_celcius + 3049)

    def _compute_mass_specific_energy(self) -> None:
        """Computes mass specific internal energy."""
        self._mass_specific_energy = self._heat_capacity_isochoric*self._temperature

    def _compute_thermal_conductivity(self) -> None:

        self._thermal_conductivity = (
            self._thermal_conductivity_slope*(
                self._temperature-self.reference.temperature[0]) +
            self.reference.thermal_conductivity[0])

    def first_partial_deriv(
            self, of_parameter_key: int,
            with_respect_to_key: int,
            holding_const_key: int) -> float | None:

        supported_derivative = self._valid_partial_derivative(
            of_parameter_key, with_respect_to_key, holding_const_key)

        if supported_derivative:
            if of_parameter_key == CoolConst.iUmass:
                # expansivity influence manisfests only for large pressures
                return self._heat_capacity_isochoric
            if of_parameter_key == CoolConst.iDmass:
                return -self._expansivity_isobaric*self._density
        else:
            raise KeyError(_MESSAGE_UNSUPPORTED_DERIVATIVE)

    def update(self, input_pair_key: int,
               first_keyed_parameter: float,
               second_keyed_parameter: float) -> None:

        # It is more explicit we perform validation here
        if self._valid_input_pair(input_pair_key):
            self._pressure = first_keyed_parameter
            self._temperature = second_keyed_parameter
        else:
            raise KeyError(_MESSAGE_UNSUPPORTED_INPUT_PAIR)
        # When temperature and pressure are updated we can update values
        # for heat capacity, mass-soecific energy, density and viscosity
        self._compute_heat_capacity_isochoric()
        self._compute_mass_specific_energy()
        self._compute_density()
        self._compute_viscosity()


@dataclass
class CrudeOilRefernceData:

    # NOTE :
    # This is needed to hold reference data describing crude oil

    temperature: tuple[float, float]
    density: tuple[float, float]
    viscosity: tuple[float, float]
    thermal_conductivity: tuple[float, float]

    mean_density: float | float64 = field(init=False)

    def __post_init__(self):
        self.mean_density = 0.5*(self.density[0]+self.density[1])


def factory_eos(
        composition: dict[str, float], backend: str = 'HEOS',
        with_state: None | Iterable = None) -> AbstractState:
    """Factory function to create interface for equation of state.

    Parameters
    ----------
    composition
        dictionary in the form of {fluid_name: molar_fraction}.
    backend, optional
        CoolProp backend to use, by default 'HEOS'.
    with_state, optional
        tuple with keyed state, of provided interface for equation of state jumps to
        specified state, otherwise no state is imposed, by default None.

    Returns
    -------
        object that provides interface to equation of state.

    Raises
    ------
    ValueError
        if molar fractions do not sum up to unity.
    """

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


def factory_crude_oil():
    pass


def density_to_specific_gravity(density: float, temperature: float = 15+273.15) -> float:
    return density/CP.PropsSI("DMASS", "T", temperature, "P", 1e5, "water")


def density_to_api_gravity(density: float, temperature: float = 15+273.15) -> float:
    specific = density_to_specific_gravity(density, temperature)
    api = specific_to_api_gravity(specific)
    return api


@overload
def api_to_specific_gravity(api_gravity: float | float64) -> float | float64:
    ...


@overload
def api_to_specific_gravity(api_gravity: NDArray[float64]) -> NDArray[float64]:
    ...


def api_to_specific_gravity(api_gravity):
    # Important NOTE : oil's specific gravity is relative to water
    specific_gravity = 141.5/(api_gravity+131.5)
    return specific_gravity


@overload
def specific_to_api_gravity(specific_gravity: float | float64) -> float | float64:
    ...


@overload
def specific_to_api_gravity(specific_gravity: NDArray[float64]) -> NDArray[float64]:
    ...


def specific_to_api_gravity(specific_gravity):
    api_gravity = 141.5/specific_gravity-131.5
    return api_gravity


if __name__ == "__main__":

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

    hcm_eos_PR = factory_eos(natural_gas_composition, "PR")
    hcm_eos_PR.build_phase_envelope("")
    PE = hcm_eos_PR.get_phase_envelope_data()

    fig, ax = plt.subplots()
    ax.set_title('Isopleth')
    ax.set_xlabel(r'T $\degree$C')
    ax.set_ylabel('Pressure, bar')
    ax.plot(np.array(PE.T)-273.15, np.array(PE.p)/1e5)
    ax.grid(True)

    # Configuring starting pressure
    # default is 100 Pa?
    CP.set_config_double(CP.PHASE_ENVELOPE_STARTING_PRESSURE_PA, 1e4)

    # Trying PQ flash with const P
    print('\nPQ FLASH :: const P')
    print(50*'=')
    Q = np.arange(0, 1+0.1, 0.1)
    P = np.ones_like(Q)*10e5
    # Q from flash-calculations above
    T = []
    for pressure, quality in zip(P, Q):
        hcm_eos_PR.update(CoolConst.PQ_INPUTS, pressure,
                          quality)  # type: ignore
        T.append(hcm_eos_PR.T())

        print(50*'-')
        print(
            f'P = {pressure/1e5} bar; T = {T[-1]-273.15:.2f} C; Q = {quality*100:.2f} %'
        )

        ax.plot(T[-1]-273.15, pressure/1e5, 'r.:')

    # Trying flash calcs for multiple pressure values, building Q(T) for const P
    stepQ = 0.1
    Q = np.arange(0, 1+stepQ, stepQ)
    P = np.array([5, 10, 20, 30, 40, 50, 60, 65])*1e5
    T = []
    for pressure in P:
        t = []
        for quality in Q:
            hcm_eos_PR.update(
                CoolConst.PQ_INPUTS, pressure, quality
            )
            t.append(hcm_eos_PR.T())
        T.append(t)
    T = np.asarray(T)

    QQ, PP = np.meshgrid(Q, P)

    fig, ax = plt.subplots()
    ax.set_title('Vapor quality vs temperature (const set of P)')
    ax.set_xlabel(r'Temperature [$\degree$C]')
    ax.set_ylabel('Q [mol/mol, %]')
    ax.grid(True)
    ct = ax.contour(T-273.15, QQ*100, PP/1e5, levels=P[1:-1]/1e5, colors='k')
    ct.clabel(fmt=r'%.0f bar')

    # I want to check binaries too
    print('\nBinary interaction coeffs :: kij')
    print(50*'=')
    fluids = hcm_eos_PR.fluid_names()
    for i, base in enumerate(fluids):
        print(50*'-')
        for j, check in enumerate(fluids[i+1:]):
            bip = hcm_eos_PR.get_binary_interaction_double(i, j, 'kij')
            print(f'{base.lower()} & {check.lower()} :: {bip:.2e}')

    plt.show()

    # NOTE
    # Peculiar issue : this section runs normally in Jupyter intercative window for the
    # first time, but re-runnin it crushes kernel. Those kind of issues were reported to
    # coolprop users group when computing phase envelope; apparently there are issues with
    # computations for interfaces used for phase envelope generation, it is easy to
    # circimvent, but still annoying
    #
    # Additional NOTE
    # Terminal execution runs normally, issue might be on jupyter's side?
