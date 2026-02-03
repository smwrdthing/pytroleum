import pytest
from numpy.testing import assert_almost_equal
import CoolProp.CoolProp as CP
import CoolProp.constants as CoolConst
import numpy as np
import matplotlib.pyplot as plt

from pytroleum.tdyna.eos import (
    CrudeOilHardcoded,
    CrudeOilReferenced,
    CrudeOilRefernceData,
    factory_eos,
    density_to_specific_gravity,
    specific_to_api_gravity,
    api_to_specific_gravity,
    density_to_api_gravity
)

"""
Module for testing equations of state (EOS) for oil

Two models are checked:
    1.CrudeOilHardcoded - simplified model with constant parameters
    2.CrudeOilReferenced - model with temperature-dependent parameters
"""


class TestCrudeOilHardcoded:
    """Тесты для класса CrudeOilHardcoded"""

    def test_initialization(self):
        """Testing initialization with default parameters."""
        oil = CrudeOilHardcoded()

        assert oil._backend == "CrudeOilHardcoded"
        assert oil._fluid == "CrudeOil"
        assert oil._density == 830
        assert oil._dynamic_viscosity == 6e-3
        assert oil._heat_capacity_isochoric == 2300

    def test_default_method(self):
        """Test of the default() method"""
        oil = CrudeOilHardcoded()

        oil._density = 800
        oil._dynamic_viscosity = 5e-3
        oil._heat_capacity_isochoric = 2000
        oil._heat_capacity_isobaric = 2000

        oil.default()

        assert oil._density == 830
        assert oil._dynamic_viscosity == 6e-3
        assert oil._heat_capacity_isochoric == 2300
        assert oil._heat_capacity_isobaric == 2300

    def test_change_method(self):
        """Test of the change() method"""
        oil = CrudeOilHardcoded()

        oil.change(new_density=850, new_visocsity=7e-3, new_heat_capacity=2400)

        assert oil._density == 850
        assert oil._dynamic_viscosity == 7e-3
        assert oil._heat_capacity_isochoric == 2400
        assert oil._heat_capacity_isobaric == 2400

    def test_first_partial_deriv(self):
        """Test calculation of partial derivatives"""
        oil = CrudeOilHardcoded()

        # First derivative dU/dT at constant P
        result = oil.first_partial_deriv(
            CoolConst.iUmass,
            CoolConst.iT,
            CoolConst.iP
        )
        assert result == 2300  # heat_capacity_isochoric

        # First derivative dρ/dT at constant P
        result = oil.first_partial_deriv(
            CoolConst.iDmass,
            CoolConst.iT,
            CoolConst.iP
        )
        assert result == 0  # for constant density

        # Unsupported derivative should raise an exception
        with pytest.raises(KeyError):
            oil.first_partial_deriv(
                CoolConst.iP,
                CoolConst.iT,
                CoolConst.iUmass
            )


class TestCrudeOilReferenced:
    """Tests for the CrudeOilReferenced class"""

    @pytest.fixture
    def reference_data(self):
        """Fixture with test data"""
        return CrudeOilRefernceData(
            temperature=(288.15, 333.15),
            density=(850, 830),
            viscosity=(0.008, 0.004),
            thermal_conductivity=(0.12, 0.10)
        )

    @pytest.fixture
    def oil(self, reference_data):
        """Fixture with an initialized object"""
        return CrudeOilReferenced(reference_data)

    def test_prepare_density_model(self, oil):
        """Test preparation of density model"""
        expected_mean_density = 0.5 * (850 + 830)
        expected = -1/expected_mean_density * (830 - 850) / (333.15 - 288.15)
        assert_almost_equal(oil._expansivity_isobaric, expected)

    def test_prepare_viscosity_model(self, oil):
        """Test preparation of viscosity model"""
        T1, T2 = 288.15, 333.15
        mu1, mu2 = 0.008, 0.004

        expected_power = T1*T2/(T1-T2)*np.log(mu2/mu1)
        expected_coeff = mu1*np.exp(-expected_power/T1)

        assert_almost_equal(oil._viscosity_model_power, expected_power)
        assert_almost_equal(oil._viscosity_model_coeff, expected_coeff)

    def test_compute_specific_gravity(self, oil):
        # Water density at 15°C
        water_density = CP.PropsSI("DMASS", "T", 288.15, "P", 1e5, "water")
        oil._compute_specific_gravity()
        expected = oil._density / water_density
        assert_almost_equal(oil._specific_gravity, expected, decimal=4)

    def test_compute_api_gravity(self, oil):
        """Test API calculation"""
        oil._compute_specific_gravity()
        oil._compute_api_gravity()
        expected = 141.5 / oil._specific_gravity - 131.5
        assert_almost_equal(oil._api_gravity, expected)

    @pytest.mark.parametrize("temp_K, expected_rho", [
        (288.15, 850),  # 15°C
        (333.15, 830),  # 60°C
    ])
    def test_compute_density(self, oil, temp_K, expected_rho):
        oil._temperature = temp_K
        oil._compute_density()
        assert_almost_equal(oil._density, expected_rho, decimal=1)

    @pytest.mark.parametrize("temp_K, expected_viscosity", [
        (288.15, 0.008),  # 15°C
        (333.15, 0.004),  # 60°C
    ])
    def test_compute_viscosity(self, oil, temp_K, expected_viscosity):
        """Test viscosity calculation """
        oil._temperature = temp_K
        oil._compute_viscosity()
        assert_almost_equal(oil._dynamic_viscosity,
                            expected_viscosity, decimal=6)

    def test_compute_heat_capacity_isochoric(self, oil):
        """Test calculation of heat capacity at constant volume"""
        oil._temperature = 320  # 46.85°C
        oil._compute_specific_gravity()
        oil._compute_heat_capacity_isochoric()
        T_celsius = 320 - 273.15
        expected = (2*T_celsius-1429)*oil._specific_gravity + \
            2.67*T_celsius + 3049
        assert_almost_equal(oil._heat_capacity_isochoric, expected, decimal=1)

    def test_compute_mass_specific_energy(self, oil):
        """Test calculation of specific energy"""
        oil._temperature = 320
        oil._compute_heat_capacity_isochoric()
        oil._compute_mass_specific_energy()

        expected = oil._heat_capacity_isochoric * 320
        assert_almost_equal(oil._mass_specific_energy, expected)

    def test_first_partial_deriv(self, oil):
        """Test calculation of partial derivatives"""
        oil.update(CoolConst.PT_INPUTS, 1e5, 300)

        # dU/dT at constant P
        dU_dT = oil.first_partial_deriv(
            CoolConst.iUmass,
            CoolConst.iT,
            CoolConst.iP
        )
        # Should be equal to heat capacity
        assert_almost_equal(dU_dT, oil._heat_capacity_isochoric)

        # dρ/dT at constant P
        drho_dT = oil.first_partial_deriv(
            CoolConst.iDmass,
            CoolConst.iT,
            CoolConst.iP
        )
        # Should be equal to -alpha * rho
        expected = -oil._expansivity_isobaric * oil._density
        assert_almost_equal(drho_dT, expected)

        # Unsupported derivative
        with pytest.raises(KeyError):
            oil.first_partial_deriv(
                CoolConst.iP,
                CoolConst.iT,
                CoolConst.iUmass
            )

    def test_update(self, oil):
        """Test of the update() method"""
        oil.update(CoolConst.PT_INPUTS, 2e5, 350)

        assert oil._pressure == 2e5
        assert oil._temperature == 350

        # Check that parameters were recalculated
        assert oil._density is not None
        assert oil._dynamic_viscosity is not None
        assert oil._mass_specific_energy is not None

        # Check exception for unsupported input pair
        with pytest.raises(KeyError):
            oil.update(CoolConst.PQ_INPUTS, 1e5, 0.5)


class TestCrudeOilRefernceData:
    """Tests for dataclass CrudeOilRefernceData"""

    def test_initialization(self):
        """Test dataclass initialization"""
        data = CrudeOilRefernceData(
            temperature=(288.15, 333.15),
            density=(850, 830),
            viscosity=(0.008, 0.004),
            thermal_conductivity=(0.12, 0.10)
        )

        assert data.temperature == (288.15, 333.15)
        assert data.density == (850, 830)
        assert data.viscosity == (0.008, 0.004)
        assert data.thermal_conductivity == (0.12, 0.10)
        expected_mean = 0.5 * (850 + 830)
        assert_almost_equal(data.mean_density, expected_mean)

    def test_post_init(self):
        """Test post-initialization"""
        data = CrudeOilRefernceData(
            temperature=(288.15, 333.15),
            density=(850, 830),
            viscosity=(0.008, 0.004),
            thermal_conductivity=(0.12, 0.10)
        )

        expected = 0.5 * (850 + 830)
        assert_almost_equal(data.mean_density, expected)


class TestFactoryFunctions:

    def test_factory_eos(self):
        """Test function for creating EOS"""
        composition = {'Methane': 0.7, 'Ethane': 0.3}

        # Create EOS
        eos = factory_eos(composition, backend='HEOS')

        # Check creation
        assert eos is not None
        assert 'Methane' in eos.fluid_names()
        assert 'Ethane' in eos.fluid_names()

        # Check composition
        fractions = eos.get_mole_fractions()
        assert_almost_equal(sum(fractions), 1.0)

        # Check exception for invalid composition
        with pytest.raises(ValueError):
            factory_eos({'Methane': 0.7, 'Ethane': 0.4})  # Sum > 1

        # Check creation with initial state
        eos_with_state = factory_eos(
            composition,
            with_state=(CoolConst.PT_INPUTS, 1e5, 300)
        )
        assert eos_with_state is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
