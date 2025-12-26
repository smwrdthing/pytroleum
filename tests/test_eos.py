import pytest
from numpy.testing import assert_almost_equal
import CoolProp.CoolProp as CP
import CoolProp.constants as CoolConst
import numpy as np

# Импорт тестируемых классов и функций
from pytroleum.tdyna.eos import (
    CrudeOilHardcoded,
    CrudeOilReferenced,
    CrudeOilRefernceData,
    factory_eos,
    factory_natgas,
    density_to_specific_gravity,
    specific_to_api_gravity,
    api_to_specific_gravity,
    MOLE_FRACTION_SUM_TOL
)


class TestCrudeOilHardcoded:
    """Тесты для класса CrudeOilHardcoded"""

    def test_initialization(self):
        """Тест инициализации с параметрами по умолчанию"""
        oil = CrudeOilHardcoded()

        assert oil._backend == "CrudeOilHardcoded"
        assert oil._fluid == "CrudeOil"
        assert oil._density == 830
        assert oil._dynamic_viscosity == 6e-3
        assert oil._heat_capacity_isochoric == 2300

    def test_default_method(self):
        """Тест метода default()"""
        oil = CrudeOilHardcoded()

        # Изменяем параметры
        oil._density = 800
        oil._dynamic_viscosity = 5e-3
        oil._heat_capacity_isochoric = 2000

        # Возвращаем к дефолтным значениям
        oil.default()

        assert oil._density == 830
        assert oil._dynamic_viscosity == 6e-3
        assert oil._heat_capacity_isochoric == 2300

    def test_change_method(self):
        """Тест метода change()"""
        oil = CrudeOilHardcoded()

        oil.change(new_density=850, new_visocsity=7e-3, new_heat_capacity=2400)

        assert oil._density == 850
        assert oil._dynamic_viscosity == 7e-3
        assert oil._heat_capacity_isochoric == 2400
        assert oil._heat_capacity_isobaric == 2400

    def test_update_method(self):
        """Тест метода update()"""
        oil = CrudeOilHardcoded()

        # Проверяем, что температура и давление устанавливаются
        oil.update(CoolConst.PT_INPUTS, 1e5, 300)

        assert oil._pressure == 1e5
        assert oil._temperature == 300

        # Проверяем вычисление энергии
        expected_energy = 2300 * 300  # heat_capacity * temperature
        assert oil._mass_specific_energy == expected_energy

    def test_first_partial_deriv(self):
        """Тест вычисления частных производных"""
        oil = CrudeOilHardcoded()

        # Правильная производная dU/dT при постоянном P
        result = oil.first_partial_deriv(
            CoolConst.iUmass,
            CoolConst.iT,
            CoolConst.iP
        )
        assert result == 2300  # heat_capacity_isochoric

        # Правильная производная dρ/dT при постоянном P
        result = oil.first_partial_deriv(
            CoolConst.iDmass,
            CoolConst.iT,
            CoolConst.iP
        )
        assert result == 0  # для постоянной плотности

        # Неподдерживаемая производная должна вызывать исключение
        with pytest.raises(KeyError):
            oil.first_partial_deriv(
                CoolConst.iP,
                CoolConst.iT,
                CoolConst.iUmass
            )

    def test_mass_specific_energy_calculation(self):
        """Тест вычисления удельной энергии"""
        oil = CrudeOilHardcoded()
        oil.update(CoolConst.PT_INPUTS, 1e5, 350)

        expected = 2300 * 350
        assert oil._mass_specific_energy == expected


class TestCrudeOilReferenced:
    """Тесты для класса CrudeOilReferenced"""

    @pytest.fixture
    def reference_data(self):
        """Фикстура с тестовыми данными"""
        return CrudeOilRefernceData(
            temperature=(288.15, 333.15),  # 15°C и 60°C
            density=(850, 830),  # кг/м³
            viscosity=(0.008, 0.004),  # Па·с
            mean_density=840
        )

    @pytest.fixture
    def oil(self, reference_data):
        """Фикстура с инициализированным объектом"""
        return CrudeOilReferenced(reference_data)

    def test_initialization(self, oil, reference_data):
        """Тест инициализации"""
        assert oil._backend == "CrudeOilReferenced"
        assert oil._fluid == "CrudeOil"
        assert oil.reference == reference_data

    def test_density_calculation(self, oil):
        """Тест вычисления плотности"""
        # Проверяем плотность при температуре по умолчанию
        expected_density = 850  # начальная плотность при 15°C
        assert_almost_equal(oil._density, expected_density, decimal=1)

        # Меняем температуру и проверяем новую плотность
        oil.update(CoolConst.PT_INPUTS, 1e5, 333.15)  # 60°C
        expected_at_60 = 830  # должна быть заданная плотность при 60°C
        assert_almost_equal(oil._density, expected_at_60, decimal=1)

    def test_first_partial_deriv(self, oil):
        """Тест частных производных"""
        # dU/dT при постоянном P
        result = oil.first_partial_deriv(
            CoolConst.iUmass,
            CoolConst.iT,
            CoolConst.iP
        )
        assert result == oil._heat_capacity_isochoric

        # dρ/dT при постоянном P
        result = oil.first_partial_deriv(
            CoolConst.iDmass,
            CoolConst.iT,
            CoolConst.iP
        )
        expected = -oil._expansivity_isobaric * oil._density
        assert_almost_equal(result, expected, decimal=6)

    def test_update_with_invalid_input(self, oil):
        """Тест update() с неподдерживаемым входным режимом"""
        with pytest.raises(KeyError):
            oil.update(CoolConst.PQ_INPUTS, 1e5, 0.5)

    def test_compute_heat_capacity(self, oil):
        """Тест вычисления теплоемкости"""
        # При температуре 15°C
        oil.update(CoolConst.PT_INPUTS, 1e5, 288.15)
        cp_15 = oil._heat_capacity_isochoric

        # При температуре 60°C
        oil.update(CoolConst.PT_INPUTS, 1e5, 333.15)
        cp_60 = oil._heat_capacity_isochoric

        # Проверяем, что теплоемкость изменилась
        assert cp_15 != cp_60


class TestCrudeOilRefernceData:
    """Тесты для dataclass CrudeOilRefernceData"""

    def test_initialization(self):
        """Тест инициализации dataclass"""
        data = CrudeOilRefernceData(
            temperature=(288.15, 333.15),
            density=(850, 830),
            viscosity=(0.008, 0.004),
            mean_density=840
        )

        assert data.temperature == (288.15, 333.15)
        assert data.density == (850, 830)
        assert data.viscosity == (0.008, 0.004)
        assert data.mean_density == 840


class TestFactoryFunctions:
    """Тесты для фабричных функций"""

    def test_factory_eos_valid_composition(self):
        """Тест фабрики EOS с валидным составом"""
        composition = {
            'Methane': 0.7,
            'Ethane': 0.2,
            'Propane': 0.1
        }

        eos = factory_eos(composition, backend='PR')

        # Проверяем, что объект создан
        assert eos is not None
        # Проверяем, что мольные доли установлены
        assert len(eos.get_mole_fractions()) == 3

    def test_factory_eos_invalid_composition(self):
        """Тест фабрики EOS с невалидным составом"""
        composition = {
            'Methane': 0.5,  # Сумма не равна 1
            'Ethane': 0.2
        }

        with pytest.raises(ValueError, match="Sum of mole fractions is not unity"):
            factory_eos(composition)

    def test_factory_eos_with_state(self):
        """Тест фабрики EOS с начальным состоянием"""
        composition = {'Water': 1.0}
        state = (CoolConst.PT_INPUTS, 101325, 300)  # 1 атм, 300K

        eos = factory_eos(composition, with_state=state)

        assert eos.T() == 300
        assert eos.p() == 101325

    def test_factory_natgas_valid(self):
        """Тест фабрики природного газа с валидным составом"""
        composition = {
            'Methane': 0.9,
            'Ethane': 0.1
        }

        eos = factory_natgas(composition)

        assert eos is not None
        fluid_names = [name.upper() for name in eos.fluid_names()]
        assert 'METHANE' in fluid_names

    def test_factory_natgas_invalid(self):
        """Тест фабрики природного газа с невалидным составом"""
        # Состав без углеводородов
        composition = {
            'Water': 0.5,
            'Air': 0.5
        }

        with pytest.raises(ValueError, match="No components associated with typical "
                           "natural gas"):
            factory_natgas(composition)


class TestUtilityFunctions:
    """Тесты вспомогательных функций"""

    def test_density_to_specific_gravity(self):
        """Тест преобразования плотности в удельный вес"""
        # Плотность воды при 15°C ~ 999 кг/м³
        density_water = 999
        specific_gravity = density_to_specific_gravity(density_water)

        # Удельный вес воды должен быть примерно 1
        assert_almost_equal(specific_gravity, 1.0, decimal=2)

    def test_specific_to_api_gravity(self):
        """Тест преобразования удельного веса в API"""
        # Удельный вес = 1.0 (вода)
        api = specific_to_api_gravity(1.0)
        # API воды = 10
        assert_almost_equal(api, 10.0, decimal=1)

        # Удельный вес = 0.8 (легкая нефть)
        api = specific_to_api_gravity(0.8)
        # API ~ 45.4
        expected = 141.5/0.8 - 131.5
        assert_almost_equal(api, expected, decimal=2)

    def test_api_to_specific_gravity(self):
        """Тест преобразования API в удельный вес"""
        # API = 10 (вода)
        specific = api_to_specific_gravity(10.0)
        assert_almost_equal(specific, 1.0, decimal=2)

        # Обратное преобразование
        api = 45.0
        specific = api_to_specific_gravity(api)
        api_back = specific_to_api_gravity(specific)
        assert_almost_equal(api, api_back, decimal=6)

    def test_array_conversions(self):
        """Тест преобразований для массивов"""
        # Массив значений API
        api_values = np.array([10.0, 20.0, 30.0, 40.0])

        # Преобразование в удельный вес
        specific_values = api_to_specific_gravity(api_values)

        # Обратное преобразование
        api_back = specific_to_api_gravity(specific_values)

        # Проверяем, что преобразования обратимы
        assert_almost_equal(api_values, api_back, decimal=6)

    def test_constants(self):
        """Тест констант"""
        # Проверяем допустимую погрешность для мольных долей
        assert MOLE_FRACTION_SUM_TOL == 1e-3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
