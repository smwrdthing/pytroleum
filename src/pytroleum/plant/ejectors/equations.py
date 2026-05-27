from scipy.constants import g
import numpy as np
from pytroleum.plant.ejectors.inputs import (OperationConditions,
                                             ACTIVE, PASSIVE)
from scipy.constants import R as UNIVERSAL_GAS_CONSTANT
from pytroleum.plant.ejectors.utils import ATMOSPHERIC_PRESSURE

# NOTE Те функции, которые следует оставить, можно переписать так, чтобы они принимали на
# NOTE вход интерфейс уравнения состояния от CoolProp


THERMAL_EQUIVALENT_OF_WORK = 0.982   # Дж/Дж


def calculate_gas_constant(molar_mass: float) -> float:
    """Газовая постоянная среды, Дж/(кг*K).

        R = R_u / M
        где: R_u — универсальная газовая постоянная, Дж/(моль·К),
        M — молярная масса среды, кг/моль
    """

    # NOTE газовую постоянную можно считать через интерфейс к уравнению состояния
    # NOTE из CoolProp, универасльную можно взять из scipy или завести файл под константы
    # NOTE и определить там
    # NOTE
    # NOTE from scipy.constants import R as UNIVERSAL_GAS_CONSTANT
    # NOTE или
    # NOTE from pytroleum.plant.ejector.constants import UNIVERSAL_GAS_CONSTANT
    # NOTE
    # NOTE дальше если где-то есть
    # NOTE eos = AbstractState("HEOS", <смесь>)
    # NOTE то газовая постоянная будет
    # NOTE UNIVERSAL_GAS_CONSTANT/eos.molar_mass()
    # NOTE
    # NOTE В CoolProp молярные массы в СИ, поэтому газовая постоянная должна быть
    # NOTE записана как 8.314... Дж/К/моль, если определяем сами

    return UNIVERSAL_GAS_CONSTANT / molar_mass


def calculate_specific_weight(density: float) -> float:
    """Удельный вес, кг/(с²·м²).

        γ = g · ρ
        где: g — ускорение свободного падения, м/с²
        ρ  — плотность среды, кг/м³
    """

    # NOTE можно считать на месте через CoolProp

    return g * density


def calculate_gas_outflow_velocity(mass_flow: float, temperature: float,
                                   pressure: float, diameter: float,
                                   molar_mass: float) -> float:
    """Скорость истечения газа в газопроводе, м/с.

        w = 4 · G · R · T / ((P + P_atm) · π · D²)

        где: G — массовый расход, кг/с
        R — газовая постоянная среды, Дж/(кг·К)
        T — температура, К
        P — избыточное давление, Па
        P_atm — атмосферное давление, Па
        D — диаметр трубопровода, м
    """

    # NOTE Эта функция работает с избыточным давлением, лучше в этом случае работать с
    # NOTE абсолютным - можно пользоваться значением от CoolProp "из коробки"

    return (4 * mass_flow * calculate_gas_constant(molar_mass) * temperature /
            ((pressure + ATMOSPHERIC_PRESSURE) * np.pi * diameter ** 2))


def calculate_adiabatic_index(conditions: OperationConditions,
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
    R_active = calculate_gas_constant(conditions.phase[ACTIVE].molar_mass())
    R_passive = calculate_gas_constant(conditions.phase[PASSIVE].molar_mass())
    Cp_active = conditions.phase[ACTIVE].cpmass()
    Cp_passive = conditions.phase[PASSIVE].cpmass()

    # NOTE можно считать "на месте" через CoolProp
    # NOTE показатель адиабаты по определению c_p(T)/c_v(T),
    # NOTE если состояние известно, то можно из eos сразу взять
    # NOTE теплоёмкости
    # NOTE
    # NOTE ещё по процессам, адиабатический процесс - изоэнтропный
    # NOTE чтобы не считать его "руками" можно также пользоваться
    # NOTE CoolProp, достаточно обновлять состояние по энтропии (она постоянная) и
    # NOTE какому-то второму параметру
    # NOTE
    # NOTE Пример
    # NOTE from CoolProp.CoolProp import AbstractState
    # NOTE from CoolProp.constants import PT_INPUTS, SmassT_INPUTS, PSmass_INPUTS
    # NOTE
    # NOTE P1, T1 = 1e5, 15+273.15
    # NOTE eos = AbstractState("HEOS",<жидкость/газ>)
    # NOTE eos.update(PT_INPUTS,P1,T1)
    # NOTE S1 = eos.smass() # записываем энтропию
    # NOTE # Адиабатное сжатие до 2 бар
    # NOTE P2 = 2e5
    # NOTE eos.update(PSmass_INPUTS, P2, S1)
    # NOTE Читаем температуру
    # NOTE T2 = eos.T()
    # NOTE
    # NOTE Код выше сработает только для чистых веществ в CoolProp, интерфейс смесей
    # NOTE допускает обновление состояния только через пару PQ или QT в двухфазном регионе
    # NOTE (давление-качество пара или качество пара-температура) или через пару PT в
    # NOTE однофазном. Это можно обойти, если дописать собственную функцию, которая будет
    # NOTE решать нелинейное уравнение вида S(P2,T) - S1 = 0
    # NOTE
    # NOTE При прочих равных считать по формуле будет быстрее, так что она может
    # NOTE быть уместна, код из заметки можно использовать для проверки, либо когда
    # NOTE этот расчёт не выполняется в больших количествах и выводить формулу заново
    # NOTE или где-то её искать накладнее + при её переписывании легко ошибиться
    # NOTE
    # NOTE Дополнительно : k на самом деле зависит от температуры, чаще всего слабо,
    # NOTE но может проявляться для больших перепадов температур/некоторых веществ

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


# NOTE пример как переписать функцию выше
# NOTE def critical_pressure_ratio(eos):
# NOTE     # Считаем, что нужное состояние уже установлено в eos
# NOTE     k = eos.cpmass()/eos.cvmass() # adiabatic index
# NOTE     beta_crit = (2/(k+1))**(k/(k-1))
# NOTE     return beta_crit
# NOTE
# NOTE Иногда выгодно давать переменным "плохие" имена, как в данном случае:
# NOTE В функции можно обозвать переменную как k - у неё непродолжительное время жизни,
# NOTE функция маленькая и модульная, из контекста и комментария понятно, что это
# NOTE показатель адиабаты + выражение больше похоже на математическую формулу


def calculate_critical_pressure(critical_pressure_ratio: float,
                                active_inlet_pressure: float) -> float:
    """Давление в критическом сечении сопла, Па.

        P_кр = β · P_a
        где:
        β — критическое отношение давлений
        P_a — давление активной среды на входе, Па
    """

    # NOTE если это нужно считать 1-2-3 раза лучше не заводить функцию, а считать
    # NOTE "на месте":
    # NOTE critical_pressure = critical_pressure_ratio(eos)*eos.p() # <- вместо функции

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

    # NOTE То же, что и для критического давления

    return (active_temperature *
            critical_pressure_ratio ** ((adiabatic_index - 1) / adiabatic_index))


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
            (outlet_pressure / inlet_pressure) ** ((adiabatic_index - 1) /
                                                   adiabatic_index))


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

    # NOTE Число Рейнольдса можно считать без удельного веса
    # NOTE
    # NOTE В целом на мой взгляд лучше уйти в уравнениях от удельного веса/силовых единиц
    # NOTE и пользоваться более привычными формами записи и единицами для напоров,
    # NOTE расходов и т.д.

    return (calculate_specific_weight(density) * velocity * diameter /
            (g * dynamic_viscosity))


def calculate_nozzle_throat_area(conditions: OperationConditions,
                                 psi: float) -> float:
    """Площадь сечения узкой части сопла, м².

        F_кр = G_a / (ψ · √(P_a / v_a))
        где:
        G_a — массовый расход активной среды, кг/с
        ψ = 2,14 для газов и ψ = 2,03 для перегретого и насыщенного водяного пара
        P_a — давление активной среды, Па
        v_a — удельный объём активной среды, м³/кг
    """
    # NOTE в докстрингах пишем только обозначения из сигнатуры вызова функций,
    # NOTE всё остальное описывается при необходимости комментариями

    return (conditions.mass_flow_rate[ACTIVE] /
            (psi * np.sqrt(conditions.pressure[ACTIVE] *
                           conditions.phase[ACTIVE].rhomass())))


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
