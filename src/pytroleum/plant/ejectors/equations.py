from scipy.constants import g
import numpy as np
from pytroleum.plant.ejectors.inputs import ActiveMediumData, PassiveMediumData

UNIVERSAL_GAS_CONSTANT = 8.314       # Дж/(моль·К)
ATMOSPHERIC_PRESSURE = 101325           # Па
THERMAL_EQUIVALENT_OF_WORK = 0.982   # Дж/Дж


def calculate_gas_constant(molecular_mass: float) -> float:
    """Газовая постоянная среды, Дж/(кг*K).

        R = R_u / M
        где: R_u — универсальная газовая постоянная, Дж/(моль·К),
        M — молекулярная масса среды, кг/моль
    """
    return UNIVERSAL_GAS_CONSTANT / molecular_mass


def calculate_specific_heat_capacity(heat_capacity: float,
                                     molecular_mass: float) -> float:
    """Удельная теплоёмкость среды, Дж/(кг·К).

        Cp = Cp_mol / M
        где: Cp_mol — молярная теплоёмкость, Дж/(моль·К)
        M — молекулярная масса среды, кг/моль
    """
    return heat_capacity / molecular_mass


def calculate_specific_weight(density: float) -> float:
    """Удельный вес, кг/(с²·м²).

        γ = g · ρ
        где: g — ускорение свободного падения, м/с²
        ρ  — плотность среды, кг/м³
    """
    return g * density


def calculate_gas_outflow_velocity(mass_flow: float, temperature: float,
                                   pressure: float, diameter: float,
                                   molecular_mass: float) -> float:
    """Скорость истечения газа в газопроводе, м/с.

        w = 4 · G · R · T / ((P + P_atm) · π · D²)

        где: G — массовый расход, кг/с
        R — газовая постоянная среды, Дж/(кг·К)
        T — температура, К
        P — избыточное давление, Па
        P_atm — атмосферное давление, Па
        D — диаметр трубопровода, м
    """
    return (4 * mass_flow * calculate_gas_constant(molecular_mass) * temperature /
            ((pressure + ATMOSPHERIC_PRESSURE) * np.pi * diameter ** 2))


def calculate_adiabatic_index(active: ActiveMediumData,
                              passive: PassiveMediumData,
                              entrainment_ratio: float) -> float:
    """Показатель адиабаты смеси активной и пассивной сред.

        k = 1 / (1 - A · (q · R_n + R_a) / (Cp_n · q + Cp_a))

        где:
        k — показатель адиабаты
        A — тепловой эквивалент работы, Дж/Дж
        q — коэффициент эжекции
        R_a  — газовая постоянная активной среды, Дж/(кг·К)
        R_n  — газовая постоянная пассивной среды, Дж/(кг·К)
        Cp_a — удельная теплоёмкость активной среды, Дж/(кг·К)
        Cp_n — удельная теплоёмкость пассивной среды, Дж/(кг·К)
    """
    R_active = calculate_gas_constant(active.molecular_mass)
    R_passive = calculate_gas_constant(passive.molecular_mass)
    Cp_active = calculate_specific_heat_capacity(
        active.heat_capacity, active.molecular_mass)
    Cp_passive = calculate_specific_heat_capacity(
        passive.heat_capacity, passive.molecular_mass)
    return 1 / (1 - THERMAL_EQUIVALENT_OF_WORK *
                (entrainment_ratio * R_passive + R_active) /
                (Cp_passive * entrainment_ratio + Cp_active))


def calculate_critical_pressure_ratio(adiabatic_index: float) -> float:
    """Критическое отношение давлений.

        β = (2 / (k + 1)) ^ (k / (k - 1))
        где:
        k — показатель адиабаты
    """
    return (2 / (adiabatic_index + 1)) ** (adiabatic_index / (adiabatic_index - 1))


def calculate_critical_pressure(critical_pressure_ratio: float,
                                active_inlet_pressure: float) -> float:
    """Давление в критическом сечении сопла, Па.

        P_кр = β · P_a
        где:
        β — критическое отношение давлений
        P_a — давление активной среды на входе, Па
    """
    return critical_pressure_ratio * active_inlet_pressure


def calculate_critical_temperature(active_temperature: float,
                                   critical_pressure_ratio: float,
                                   adiabatic_index: float) -> float:
    """Температура в критическом сечении сопла, К.

        T_кр = T_a · β ^ ((k - 1) / k)
        где:
        T_a — температура активной среды, К
        β — критическое отношение давлений
        k — показатель адиабаты
    """
    return active_temperature * critical_pressure_ratio ** ((adiabatic_index - 1) / adiabatic_index)


def calculate_section_temperature(inlet_temperature: float,
                                  outlet_pressure: float,
                                  inlet_pressure: float,
                                  adiabatic_index: float) -> float:
    """Температура в сечении через адиабатный процесс, К.

        T_вых = T_вх · (P_вых / P_вх) ^ ((k - 1) / k)

        где:
        T_вых — температура на выходе, К
        T_вх — температура на входе, К
        P_вых — давление на выходе, Па
        P_вх — давление на входе, Па
        k — показатель адиабаты
    """
    return (inlet_temperature *
            (outlet_pressure / inlet_pressure) ** ((adiabatic_index - 1) / adiabatic_index))


def calculate_circle_area(diameter: float) -> float:
    """Площадь круглого сечения, м².

        F = π · D² / 4
        где:
        D — диаметр сечения, м
    """
    return np.pi * diameter ** 2 / 4


def calculate_circle_diameter(area: float) -> float:
    """Диаметр круглого сечения, м.

        D = √(4 · F / π)
        где:
        F — площадь сечения, м²
    """
    return np.sqrt(4 * area / np.pi)


def calculate_reynolds_number(density: float,
                              velocity: float,
                              diameter: float,
                              dynamic_viscosity: float) -> float:
    """Число Рейнольдса.

        Re = γ · w · D / (g · η)
        где:
        Re — число Рейнольдса
        γ — удельный вес среды, кг/(с²·м²)
        w — скорость потока, м/с
        D — диаметр трубопровода, м
        g — ускорение свободного падения, м/с²
        η — динамическая вязкость, Па·с
    """
    return (calculate_specific_weight(density) * velocity * diameter /
            (g * dynamic_viscosity))


def calculate_nozzle_throat_area(active: ActiveMediumData,
                                 psi: float) -> float:
    """Площадь сечения узкой части сопла, м².

        F_кр = G_a / (ψ · √(P_a / v_a))
        где:
        G_a — массовый расход активной среды, кг/с
        ψ = 2,14 для газов и ψ = 2,03 для перегретого и насыщенного водяного пара
        P_a — давление активной среды, Па
        v_a — удельный объём активной среды, м³/кг
    """
    return active.mass_flow / (psi * np.sqrt(active.inlet_pressure / active.specific_volume))


def calculate_diffuser_length(diameter_exit: float, diameter_inlet: float,
                              opening_angle: float) -> float:
    """Длина диффузора, м.

        L_диф = (D_вых - D_вх) / (2 · tan(α / 2))

        где:
        D_вых — диаметр выходного сечения диффузора, м
        D_вх  — диаметр входного сечения диффузора, м
        α     — угол раскрытия диффузора, °
                (рекомендовано 6°, допустимо 2°–13°; при α > 14° поток
                не заполняет сечения равномерно, усиливается вихреобразование
                вдоль стенок, возникают обратные токи, коэффициент φ резко падает)
    """
    return (diameter_exit - diameter_inlet) / (2 * np.tan(np.radians(opening_angle / 2)))


def calculate_section_area(area: float, ratio: float) -> float:
    """Площадь сечения через отношение площадей, м².

        F = area / ratio
        где:
        area  — заданная площадь, м²
        ratio — отношение площадей
    """
    return area / ratio


def calculate_section_velocity(velocity: float, ratio: float) -> float:
    """Скорость потока через отношение площадей сечений, м/с.

        w = velocity / ratio
        где:
        w — скорость в искомом сечении, м/с
        velocity — скорость в исходном сечении, м/с
        ratio — отношение площадей сечений
    """
    return velocity / ratio
