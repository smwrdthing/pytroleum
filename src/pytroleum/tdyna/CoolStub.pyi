# Stub for CoolProp API, needed for autocompletion and convenient class extension
from CoolProp.CoolProp import PyPhaseEnvelopeData, PySpinodalData
from numpy import ndarray
from typing import Any, Iterable


class AbstractState:

    # Accessibles from AbstractState

    def __init__(self, backend: str, fluids: str):
        ...

    @property
    def Bvirial(self) -> float:
        ...

    @property
    def Cvirial(self) -> float:
        ...

    @property
    def PIP(self) -> float:
        ...

    @property
    def Prandtl(self) -> float:
        ...

    @property
    def Q(self) -> float:
        ...

    @property
    def T(self) -> float:
        ...

    @property
    def T_critical(self) -> float:
        ...

    @property
    def T_reducing(self) -> float:
        ...

    @property
    def Tmax(self) -> float:
        ...

    @property
    def Tmin(self) -> float:
        ...

    @property
    def Ttriple(self) -> float:
        ...

    @property
    def acentric_factor(self) -> float:
        ...

    @property
    def all_critical_points(self) -> float:
        ...

    @property
    def alpha0(self) -> float:
        ...

    @property
    def alphar(self) -> float:
        ...

    @property
    def apply_simple_mixing_rule(self) -> float:
        ...

    @property
    def backend_name(self) -> float:
        ...

    def build_phase_envelope(self) -> float:
        ...

    def build_spinodal(self) -> float:
        ...

    def change_EOS(self, component: int, new_EOS: str) -> float:
        ...

    @property
    def chemical_potential(self) -> float:
        ...

    @property
    def compressibility_factor(self) -> float:
        ...

    @property
    def conductivity(self) -> float:
        ...

    @property
    def conductivity_contributions(self) -> float:
        ...

    @property
    def conformal_state(self) -> float:
        ...

    @property
    def cp0mass(self) -> float:
        ...

    @property
    def cp0molar(self) -> float:
        ...

    @property
    def cpmass(self) -> float:
        ...

    @property
    def cpmolar(self) -> float:
        ...

    @property
    def criticality_contour_values(self) -> float:
        ...

    @property
    def cvmass(self) -> float:
        ...

    @property
    def cvmolar(self) -> float:
        ...

    @property
    def d2alpha0_dDelta2(self) -> float:
        ...

    @property
    def d2alpha0_dDelta_dTau(self) -> float:
        ...

    @property
    def d2alpha0_dTau2(self) -> float:
        ...

    @property
    def d2alphar_dDelta2(self) -> float:
        ...

    @property
    def d2alphar_dDelta_dTau(self) -> float:
        ...

    @property
    def d2alphar_dTau2(self) -> float:
        ...

    @property
    def d3alpha0_dDelta2_dTau(self) -> float:
        ...

    @property
    def d3alpha0_dDelta3(self) -> float:
        ...

    @property
    def d3alpha0_dDelta_dTau2(self) -> float:
        ...

    @property
    def d3alpha0_dTau3(self) -> float:
        ...

    @property
    def d3alphar_dDelta2_dTau(self) -> float:
        ...

    @property
    def d3alphar_dDelta3(self) -> float:
        ...

    @property
    def d3alphar_dDelta_dTau2(self) -> float:
        ...

    @property
    def d3alphar_dTau3(self) -> float:
        ...

    @property
    def d4alphar_dDelta2_dTau2(self) -> float:
        ...

    @property
    def d4alphar_dDelta3_dTau(self) -> float:
        ...

    @property
    def d4alphar_dDelta4(self) -> float:
        ...

    @property
    def d4alphar_dDelta_dTau3(self) -> float:
        ...

    @property
    def d4alphar_dTau4(self) -> float:
        ...

    @property
    def dalpha0_dDelta(self) -> float:
        ...

    @property
    def dalpha0_dTau(self) -> float:
        ...

    @property
    def dalphar_dDelta(self) -> float:
        ...

    @property
    def dalphar_dTau(self) -> float:
        ...

    @property
    def delta(self) -> float:
        ...

    @property
    def first_partial_deriv(self) -> float:
        ...

    @property
    def first_saturation_deriv(self) -> float:
        ...

    @property
    def first_two_phase_deriv(self) -> float:
        ...

    @property
    def first_two_phase_deriv_splined(self) -> float:
        ...

    @property
    def fluid_names(self) -> float:
        ...

    @property
    def fluid_param_string(self) -> float:
        ...

    def fugacity(self) -> float:
        ...

    def fugacity_coefficient(self) -> float:
        ...

    @property
    def fundamental_derivative_of_gas_dynamics(self) -> float:
        ...

    @property
    def gas_constant(self) -> float:
        ...

    def get_binary_interaction_double(self, component1: str | int, component2: str | int) -> float:
        ...

    def get_binary_interaction_string(self, component1: str | int, component2: str | int) -> str:
        ...

    def get_fluid_constant(self, component_index: int, const_key: int) -> float:
        ...

    def get_fluid_parameter_double(self, component: int, param: str) -> float:
        ...

    def get_mass_fractions(self) -> list[float]:
        ...

    def get_mole_fractions(self) -> list[float]:
        ...

    def get_phase_envelope_data(self) -> PyPhaseEnvelopeData:
        ...

    def get_spinodal_data(self) -> PySpinodalData:
        ...

    @property
    def gibbsmass(self) -> float:
        ...

    @property
    def gibbsmass_excess(self) -> float:
        ...

    @property
    def gibbsmolar(self) -> float:
        ...

    @property
    def gibbsmolar_excess(self) -> float:
        ...

    @property
    def gibbsmolar_residual(self) -> float:
        ...

    def has_melting_line(self) -> bool:
        ...

    @property
    def helmholtzmass(self) -> float:
        ...

    @property
    def helmholtzmass_excess(self) -> float:
        ...

    @property
    def helmholtzmolar(self) -> float:
        ...

    @property
    def helmholtzmolar_excess(self) -> float:
        ...

    @property
    def hmass(self) -> float:
        ...

    @property
    def hmass_excess(self) -> float:
        ...

    @property
    def hmass_idealgas(self) -> float:
        ...

    @property
    def hmolar(self) -> float:
        ...

    @property
    def hmolar_excess(self) -> float:
        ...

    @property
    def hmolar_idealgas(self) -> float:
        ...

    @property
    def hmolar_residual(self) -> float:
        ...

    @property
    def ideal_curve(self) -> float:
        ...

    @property
    def isobaric_expansion_coefficient(self) -> float:
        ...

    @property
    def isothermal_compressibility(self) -> float:
        ...

    def keyed_output(self, param_key: int) -> float | str:
        ...

    @property
    def melting_line(self) -> float:
        ...

    @property
    def molar_mass(self) -> float:
        ...

    @property
    def mole_fractions_liquid(self) -> float:
        ...

    @property
    def mole_fractions_vapor(self) -> float:
        ...

    @property
    def name(self) -> float:
        ...

    @property
    def neff(self) -> float:
        ...

    @property
    def p(self) -> float:
        ...

    @property
    def p_critical(self) -> float:
        ...

    def phase(self) -> int:
        ...

    @property
    def pmax(self) -> float:
        ...

    @property
    def rhomass(self) -> float:
        ...

    @property
    def rhomass_critical(self) -> float:
        ...

    @property
    def rhomass_reducing(self) -> float:
        ...

    @property
    def rhomolar(self) -> float:
        ...

    @property
    def rhomolar_critical(self) -> float:
        ...

    @property
    def rhomolar_reducing(self) -> float:
        ...

    def saturated_liquid_keyed_output(self, param_key: int) -> float:
        ...

    def saturated_vapor_keyed_output(self, param_key: int) -> float:
        ...

    @property
    def saturation_ancillary(self) -> float:
        ...

    @property
    def second_partial_deriv(self) -> float:
        ...

    @property
    def second_saturation_deriv(self) -> float:
        ...

    @property
    def second_two_phase_deriv(self) -> float:
        ...

    def set_binary_interaction_double(self) -> float:
        ...

    def set_binary_interaction_string(self) -> float:
        ...

    def set_cubic_alpha_C(self) -> float:
        ...

    def set_fluid_parameter_double(self) -> float:
        ...

    def set_mass_fractions(self, mass_fractions: Iterable[float]):
        ...

    def set_mole_fractions(self, mole_fractions: Iterable[float]) -> float:
        ...

    def set_volu_fractions(self, volu_fractions: Iterable[float]) -> float:
        ...

    @property
    def smass(self) -> float:
        ...

    @property
    def smass_excess(self) -> float:
        ...

    @property
    def smass_idealgas(self) -> float:
        ...

    @property
    def smolar(self) -> float:
        ...

    @property
    def smolar_excess(self) -> float:
        ...

    @property
    def smolar_idealgas(self) -> float:
        ...

    @property
    def smolar_residual(self) -> float:
        ...

    def specify_phase(self, imposed_phase_key: int) -> float:
        ...

    @property
    def speed_sound(self) -> float:
        ...

    @property
    def surface_tension(self) -> float:
        ...

    @property
    def tangent_plane_distance(self) -> float:
        ...

    @property
    def tau(self) -> float:
        ...

    def trivial_keyed_output(self, params_key: int) -> float:
        ...

    @property
    def true_critical_point(self) -> float:
        ...

    @property
    def umass(self) -> float:
        ...

    @property
    def umass_excess(self) -> float:
        ...

    @property
    def umass_idealgas(self) -> float:
        ...

    @property
    def umolar(self) -> float:
        ...

    @property
    def umolar_excess(self) -> float:
        ...

    @property
    def umolar_idealgas(self) -> float:
        ...

    def unspecify_phase(self):
        ...

    def update(self, input_pair_key: int, input1: float, input2: float):
        ...

    def update_QT_pure_superanc(self, Q: float, T: float) -> float:
        ...

    def update_with_guesses(
            self, input_pair_key: int, input1: float, input2: float, guesse: Any) -> float:
        ...

    @property
    def viscosity(self) -> float:
        ...

    @property
    def viscosity_contributions(self) -> float:
        ...

    @property
    def volumemass_excess(self) -> float:
        ...

    @property
    def volumemolar_excess(self) -> float:
        ...
