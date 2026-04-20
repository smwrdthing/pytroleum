from pytroleum.plant.tps.utils import (_major_header, _minor_divider,
                                       _TO_MM, _TO_MICRON, PERCENT, SECONDS_PER_MINUTE)
from pytroleum.plant.tps.inputs import (SeparatorParameters,
                                        OperationConditions,
                                        PhysicalProperties,
                                        FlowRates,
                                        CoalescerPacking)

import numpy as np
from scipy.constants import g


class Separator:
    """Расчёт сепаратора"""

    def __init__(self, design: SeparatorParameters,
                 conditions: OperationConditions,
                 properties: PhysicalProperties,
                 flows: FlowRates,
                 diameter_water_droplet: float,
                 diameter_oil_droplet: float):
        self.design = design
        self.conditions = conditions
        self.properties = properties
        self.flows = flows
        self.diameter_water_droplet = diameter_water_droplet
        self.diameter_oil_droplet = diameter_oil_droplet

        # NOTE многие методы в этом классе работают как функции доступа, вместо этого
        # NOTE в python обычно просто предоставляется прямой досутп к атрибутам, не
        # NOTE помешает перевести такие методы в атрибуты, вычисляемые либо при создании
        # NOTE объекта в __init__, либо где-то ещё в каком-нибудь контексте

    # --- Сепаратор ---

    def residence_time(self) -> float:
        """Время пребывания жидкости в сепараторе, с"""
        return (self.design.volume_separator * self.design.fill_coeff /
                self.conditions.flow_liquid)

    def capacity(self):
        """Максимальная производительность аппарата по жидкости с
        учетом коэффициента заполнения"""
        return (self.design.volume_separator * self.design.fill_coeff /
                self.residence_time())

    # --- Первая секция (Н+В) ---

    def volume_first_section(self) -> float:
        """Объём первой секции, м³"""
        # NOTE это тоже можно сделать атрибутом вместо метода
        return (np.pi * self.design.inner_diameter ** 2 / 4 *
                self.design.length_first_section)

    # NOTE вместо
    # NOTE def volume_first_section(self):
    # NOTE     ...
    # NOTE
    # NOTE __init__(self, design : SeparatorDesign, ...):
    # NOTE     # на случай, если нам нужны конструктивные параметры после создания объекта
    # NOTE     self.design = design
    # NOTE     self.volume_first_section = np.pi*design.inner_diameter**2/4 ...
    # NOTE
    # NOTE ещё лучше держать конструктивные параметры в объекте отдельного класса
    # NOTE то есть объем можно хранить в design, как и все размеры
    # NOTE тогда design можно передавать функциям и на месте брать нужные значения
    # NOTE
    # NOTE Например, нам нужно уметь считать время пребывания, это можно сделать
    # NOTE в таком виде (вне класса, обычой функцией):
    # NOTE
    # NOTE def compute_residence_time(design:SeparatorDesign,
    # NOTE                            conditions:OperationCondition):
    # NOTE     volume = design.volume
    # NOTE     volumetric_flow_rate = conditions.volumetric_flow_rate
    # NOTE
    # NOTE     return volume/volumetric_flow_rate
    # NOTE
    # NOTE Это dependency injection - мы передаём созданный объект и работаем с ним,
    # NOTE так мы разделяяем создание объекта и его использование
    # NOTE
    # NOTE Можно, конечно, считать и хранить где-то и время пребывания, но оно зависит
    # NOTE от двух разных вещей - от конструкции и от рабочих параметров, здесь сложнее
    # NOTE понять, кто должен хранить время пребывания
    # NOTE (+ оно может меняться, если меняются рабочие условия) - поэтому
    # NOTE расчёт времени пребывания следует выполнить скорее в виде вызваемой функции
    # NOTE (или заносить как атрибут в третий класс-калькулятор, если его создание
    # NOTE оправдано)

    def residence_time_first_section(self) -> float:
        """Время пребывания жидкости в первой секции, с"""
        return (self.volume_first_section() * self.design.fill_coeff_first_section /
                self.conditions.flow_liquid)

    def capacity_first_section(self) -> float:
        """Пропускная способность первой секции, м³/с"""
        return (self.volume_first_section() * self.design.fill_coeff_first_section /
                self.residence_time_first_section())

    # --- Сборник нефти после перегородки ---

    def volume_second_section(self) -> float:
        """Объём сборника нефти после перегородки, м³"""
        return (np.pi * self.design.inner_diameter ** 2 / 4 *
                self.design.length_second_section +
                self.design.volume_ell_head)

    def residence_time_second_section(self) -> float:
        """Время пребывания нефти в сборнике после перегородки, с"""
        return self.residence_time() - self.residence_time_first_section()

    def capacity_second_section(self) -> float:
        """Пропускная способность сборника нефти после перегородки, м³/с"""
        return (self.volume_second_section() * self.design.fill_coeff_second_section /
                self.residence_time_second_section())

    # NOTE ещё пример, как можно провести рефакторинг, функцию выше я бы вынес из класса
    # NOTE следующим образом:
    # NOTE
    # NOTE def compute_separator_capacity(design:SeparatorDesign,
    # NOTE                                conditions:OperationConditions,
    # NOTE                                fill_coeffs:Iterable[float] = FILL_COEFFS
    # NOTE ) -> tuple[float, float]:
    # NOTE
    # NOTE     # если у нас есть функция, которая вычисляет время пребывания в сепараторе
    # NOTE     rt = compute_residence_time(design, conditions)
    # NOTE
    # NOTE     first_section_capacity = (
    # NOTE        design.volume[FIRST_SECTION]*fill_coeffs[FIRST_SECTION]/
    # NOTE        rt[FIRST_SECTION])
    # NOTE
    # NOTE     second_section_capacity = (
    # NOTE        design.volume[FIRST_SECTION]*fill_coeffs[FIRST_SECTION]/
    # NOTE        rt[FIRST_SECTION])
    # NOTE
    # NOTE     return first_section_capacity, second_section_capacity
    # NOTE
    # NOTE Комментарии к функции :
    # NOTE
    # NOTE Коэффициенты заполнения теперь передаваемые параметры со значением по
    # NOTE умолчанию, значение по умолчанию выполнено в виде константы того же типа,
    # NOTE определённой где-то выше в коде - теперь нам не нужно заботиться об атрибутах,
    # NOTE которые хранят эти значения. Здесь предполагается, что для большого числа
    # NOTE случаев подойдут типовые значения, если их нужно будет изменить - мы можем
    # NOTE передать новые значения на месте
    # NOTE
    # NOTE ---
    # NOTE
    # NOTE Объёмы секций можно держать в списке/кортеже,
    # NOTE индексы выполнены константами с говорящими именами, благодаря этому
    # NOTE код читается практически как обычный английский :
    # NOTE
    # NOTE легко интерпертировать :
    # NOTE design.volume[FIRST_SECTION] -> Volume of first section of given design
    # NOTE
    # NOTE Идею можно распространить на все размеры/параметры, которые хранятся в design -
    # NOTE это немного усложняет структуру объекта (надо знать, что атрибуты - контейнеры
    # NOTE и что где-то лежат константы-индексы, по которым надо обращаться к конкретному
    # NOTE элементу в контейнере), но при этом в design количество атрибутов сокращается
    # NOTE в 2 раза (если секций две) - класс становится менее "захламлённым".
    # NOTE
    # NOTE В этом случае структурное усложнение очень незначительно и легко
    # NOTE компенсируется аннотациями типов, автокомплитом и/или документацией,
    # NOTE а выигрыш большой - в атрибутах/методах объекта легче ориентироваться,
    # NOTE т.к. их меньше
    # NOTE
    # NOTE ---
    # NOTE
    # NOTE Вероятнее всего, когда нам нужна пропускная способность - мы хотим
    # NOTE пропускнуые способности по всем cекциям, поэтому вместо двух функций
    # NOTE может быть целесообразнее написать одну, которая сразу возвращает
    # NOTE пропускные способности обеих секций - поэтому функция выше возвращает кортеж
    # NOTE
    # NOTE Время пребывания считается внутри функции, т.к. может зависеть от рабочих
    # NOTE условий, которые для фиксированной конструкии (design) может меняться
    # NOTE (например, мы хотим провести расчёт на разные рабочие условия) - тащить время
    # NOTE пребывания как атрибут в этом случае утомительно, нужно помнить об этом и
    # NOTE следить за его актуальностью
    # NOTE
    # NOTE ---
    # NOTE
    # NOTE Ещё тут важно отметить, что при таком подходе у нас намечается унифицированная
    # NOTE сигнатура вызова - мы передаём в compute_separator_capacity design и
    # NOTE conditions, те же два параметра мы передаём функции, которая считает нам время
    # NOTE пребывания (compute_residence_time) - это в общем случае скорее хорошо:
    # NOTE
    # NOTE Снижается нагрузка на пользователя - фреймворк становится более
    # NOTE унифицированным, сигнатуры функций становятся схожими, не нужно каждый раз
    # NOTE разбираться с чем работают разные функции
    # NOTE
    # NOTE Легко писать код, который использует эти фунции, легко вносить изменения,
    # NOTE автоматизировать работу и т.д.

    # --- Скорость движения жидкой фазы и газовой фазы в сечении сепаратора ---

    def liquid_flow_area(self) -> float:
        """Площадь сечения для прохода жидкости"""
        return (np.pi * self.design.inner_diameter ** 2 / 4 *
                self.design.fill_coeff_first_section)

    def gas_flow_area(self) -> float:
        """Площадь сечения для прохода газа"""
        return (np.pi * self.design.inner_diameter ** 2 / 4 - self.liquid_flow_area())

    def water_flow_area(self) -> float:
        """Площадь сечения для прохода воды"""
        return self.liquid_flow_area()*self.properties.water_cut

    def oil_flow_area(self) -> float:
        """Площадь сечения для прохода нефти"""
        return self.liquid_flow_area()-self.water_flow_area()

    def gas_velocity(self) -> float:
        """Скорость движения газа"""
        return self.flows.flow_gas_work/self.gas_flow_area()

    def oil_velocity(self) -> float:
        """Скорость движения нефти"""
        return self.flows.flow_oil/self.oil_flow_area()

    def water_velocity(self) -> float:
        """Скорость движения воды"""
        return self.flows.flow_water/self.water_flow_area()

    # NOTE выше три функции, которые делают одно и то же
    # NOTE это можно заменить одной функцией
    # NOTE
    # NOTE def compute_velocities(design : SeparatorDesign,
    # NOTE                      conditions : OperationConditions) -> None:
    # NOTE
    # NOTE     conditions.velocity[VAPOR] = (conditions.flow_rate[VAPOR]/
    # NOTE                                   design.flow_area[VAPOR])
    # NOTE     conditions.velocity[OIL] = ...
    # NOTE     conditions.velocity[WATER] = ...

    # --- Осаждение капель воды ---
    def velocity_water_settling(self) -> float:
        """Скорость осаждения капель воды в слое нефти"""
        return (self.diameter_water_droplet**2 *
                (self.properties.water_density-self.properties.oil_density)*g /
                (18*self.properties.viscosity_oil))

    # NOTE расчёт скорости осаждения тоже можно сделать внешней функцией
    # NOTE
    # NOTE def compute_settling_velocity(drop_diameter: float,
    # NOTE                               continuous_phase: EquationOfState,
    # NOTE                               dispersed_phase: EquationOfState) -> float:
    # NOTE
    # NOTE     density_diff = continuous_phase.rhomass() - dispresed_phase.rhomass()
    # NOTE
    # NOTE     settling_velocity = gravity*diameter**2*density_diff/(
    # NOTE                         18*continuous_phase.viscosity())
    # NOTE
    # NOTE     return settling_velocity
    # NOTE
    # NOTE Тут мы используем интерфейсы к уравнениям состояния непрерывной и
    # NOTE диспергированной фазы (см. tdyna)
    # NOTE
    # NOTE Разность плотностей определяет знак скорости, "положительное" значение
    # NOTE выбирается один раз, потом знак скорости интерпертируется как "всплытие"
    # NOTE или "осаждение"
    # NOTE
    # NOTE В этом случае плотность непрерывной фазы вычитается из плотсноти
    # NOTE диспергированной фазы - если непрерывная фаза тяжелее капля всплывает вверх,
    # NOTE положительная скорость = всплытие (скорость вверх)
    # NOTE отрицательная скорость = осаждение (скорость вниз)
    # NOTE
    # NOTE Эту функцию можно использовать для любых двух фаз, где мы считаем скорость
    # NOTE по закону Стокса

    def oil_transit_time(self) -> float:
        """Время прохождения нефтью расстояния"""
        return self.design.L_c/self.oil_velocity()

    def water_settling_height(self) -> float:
        """За это время капли воды опустятся на высоту"""
        return self.velocity_water_settling()*self.oil_transit_time()

    # ---Всплытие капель нефти ---
    def velocity_oil_rising(self) -> float:
        """Скорость подъёма капель нефти в слое воды"""
        return (self.diameter_oil_droplet**2 *
                (self.properties.water_density-self.properties.oil_density)*g /
                (18*self.properties.viscosity_water))

    def water_transit_time(self) -> float:
        """Время прохождения водой расстояния"""
        return self.design.L_c/self.water_velocity()

    def oil_rising_height(self) -> float:
        """За это время капли нефти поднимутся на высоту"""
        return self.velocity_oil_rising()*self.water_transit_time()


class Coalescer:
    def __init__(self, coalescer_packing: CoalescerPacking,
                 separator: Separator) -> None:
        self.coalescer_packing = coalescer_packing
        self.separator = separator

    # ---Для верхнего коалесцера ---
    def droplet_water_settling_time(self):
        """"Время осаждения капель воды в зазоре"""
        return (self.coalescer_packing.coalescer_top_gap /
                (self.separator.velocity_water_settling() *
                 np.cos(np.radians(self.coalescer_packing.angle))))

    def coalescer_top_channel_length(self):
        """Длина канала верхнего коалесцера"""
        return self.separator.oil_velocity() * self.droplet_water_settling_time()

    # ---Для нижнего коалесцера ---
    def droplet_oil_rising_time(self):
        """"Время всплытия капель нефти в зазоре"""
        return (self.coalescer_packing.coalescer_bottom_gap /
                (self.separator.velocity_oil_rising() *
                 np.cos(np.radians(self.coalescer_packing.angle))))

    def coalescer_bottom_channel_length(self):
        """Длина канала нижнего коалесцера"""
        return self.separator.water_velocity()*self.droplet_oil_rising_time()


if __name__ == "__main__":
    from pytroleum.plant.tps.utils import SECONDS_PER_DAY

    design = SeparatorParameters(
        inner_diameter=2.0,
        length_cylindrical_part=9.5,
        fill_coeff=0.858,
        fill_coeff_second_section=0.858,
        fill_coeff_first_section=0.858,
        length_semiaxis=0.618,
        length_first_section=8.2,
        length_second_section=1.3,
        L_c=4.7
    )
    conditions = OperationConditions(
        pressure_work=4e6,
        temperature_work=353,
        flow_gas_norm=300000 / SECONDS_PER_DAY,
        flow_liquid=500 / SECONDS_PER_DAY,
    )

    properties = PhysicalProperties(
        gas_density_norm=0.94,
        oil_density=933,
        water_density=966,
        water_cut=0.6,
        gas_factor=267.9,
        oil_surface_tension=0.02848,
        viscosity_oil=3.073e-3,
        viscosity_water=0.544e-3
    )

    flows = FlowRates(conditions=conditions, properties=properties)

    diameter_water_droplet = 100e-6
    diameter_oil_droplet = 50e-6

    separator = Separator(design=design, conditions=conditions,
                          properties=properties, flows=flows,
                          diameter_water_droplet=diameter_water_droplet,
                          diameter_oil_droplet=diameter_oil_droplet)

    coalescer_packing = CoalescerPacking(
        coalescer_top_gap=15e-3,
        coalescer_bottom_gap=25e-3,
    )
    coalescer = Coalescer(
        coalescer_packing=coalescer_packing, separator=separator)

    _major_header("РАСЧЁТ ПРОПУСКНОЙ СПОСОБНОСТИ СЕПАРАТОРА ПО ЖИДКОСТИ")

    print(f"Внутренний диаметр сепаратора: "
          f"{design.inner_diameter * _TO_MM:.0f} мм")
    print(f"Длина цилиндрической части сепаратора: "
          f"{design.length_cylindrical_part:.1f} м")
    print(f"Длина полуоси эллиптического днища: "
          f"{design.length_semiaxis:.3f} м")
    print(f"Объёмный расход жидкости: "
          f"{conditions.flow_liquid * SECONDS_PER_DAY:.1f} м³/сут")
    print(f"Коэффициент заполнения: "
          f"{design.fill_coeff * PERCENT:.1f} %")
    print(f"Номинальный объём сепаратора: "
          f"{design.volume_separator:.3f} м³")
    print(f"Время пребывания жидкости: "
          f"{separator.residence_time() / SECONDS_PER_MINUTE:.2f} мин")
    print(f"Пропускная способность: "
          f"{separator.capacity() * SECONDS_PER_DAY:.2f} м³/сут")

    _minor_divider()
    print("ПЕРВАЯ СЕКЦИЯ (НЕФТЬ + ВОДА)")
    _minor_divider()
    print(f"Длина первой секции: "
          f"{design.length_first_section:.1f} м")
    print(f"Коэффициент заполнения: "
          f"{design.fill_coeff_first_section * PERCENT:.1f} %")
    print(f"Объём первой секции: "
          f"{separator.volume_first_section():.3f} м³")
    print(f"Время пребывания (Н+В): "
          f"{separator.residence_time_first_section() / SECONDS_PER_MINUTE:.3f} мин")
    print(f"Пропускная способность: "
          f"{separator.capacity_first_section() * SECONDS_PER_DAY:.2f} м³/сут")

    _minor_divider()
    print("СБОРНИК НЕФТИ (ПОСЛЕ ПЕРЕГОРОДКИ)")
    _minor_divider()
    print(f"Длина секции после перегородки: "
          f"{design.length_second_section:.1f} м")
    print(f"Коэффициент заполнения: "
          f"{design.fill_coeff_second_section * PERCENT:.1f} %")
    print(f"Объём секции после перегородки: "
          f"{separator.volume_second_section():.3f} м³")
    print(f"Время пребывания: "
          f"{separator.residence_time_second_section() / SECONDS_PER_MINUTE:.3f} мин")
    print(f"Пропускная способность: "
          f"{separator.capacity_second_section() * SECONDS_PER_DAY:.3f} м³/сут")

    _minor_divider()
    print("РАСЧЕТ СКОРОСТЕЙ ДВИЖЕНИЯ ЖИДКОЙ ФАЗЫ И ГАЗОВОЙ ФАЗЫ В СЕЧЕНИИ СЕПАРАТОРА")
    _minor_divider()
    print(f"Площадь сечения для прохода жидкости: "
          f"{separator.liquid_flow_area():.3f} м²")
    print(f"Площадь сечения для прохода газа: "
          f"{separator.gas_flow_area():.3f} м²")
    print(f"Площадь сечения для прохода воды: "
          f"{separator.water_flow_area():.3f} м²")
    print(f"Площадь сечения для прохода нефти: "
          f"{separator.oil_flow_area():.3f} м²")

    _minor_divider()
    print(f"Скорость движения газа: "
          f"{separator.gas_velocity():.4f} м/с")
    print(f"Скорость движения нефти: "
          f"{separator.oil_velocity()*_TO_MM:.4f} мм/с")
    print(f"Скорость движения воды: "
          f"{separator.water_velocity()*_TO_MM:.4f} мм/с")

    _minor_divider()
    print("ОСАЖДЕНИЕ КАПЕЛЬ ВОДЫ В СЛОЕ НЕФТИ")
    _minor_divider()
    print(f"Расстояние от распределительной решетки до сливной перегородки: "
          f"{design.L_c:.1f} м")
    print(f"Диаметр капли воды: "
          f"{diameter_water_droplet * _TO_MICRON:.0f} мкм")
    print(f"Скорость осаждения капель воды: "
          f"{separator.velocity_water_settling() * _TO_MM:.4f} мм/с")
    print(f"Время прохождения нефтью расстояния: "
          f"{separator.oil_transit_time():.2f} с")
    print(f"Высота осаждения капель воды: "
          f"{separator.water_settling_height() * _TO_MM:.2f} мм")

    _minor_divider()
    print("ВСПЛЫТИЕ КАПЕЛЬ НЕФТИ В СЛОЕ ВОДЫ")
    _minor_divider()
    print(f"Расстояние от распределительной решетки до сливной перегородки: "
          f"{design.L_c:.1f} м")
    print(f"Диаметр капли нефти: "
          f"{diameter_oil_droplet * _TO_MICRON:.0f} мкм")
    print(f"Скорость подъёма капель нефти: "
          f"{separator.velocity_oil_rising() * _TO_MM:.4f} мм/с")
    print(f"Время прохождения водой расстояния: "
          f"{separator.water_transit_time():.2f} с")
    print(f"Высота подъёма капель нефти: "
          f"{separator.oil_rising_height() * _TO_MM:.2f} мм")

    _minor_divider()
    print("ВЕРХНИЙ КОАЛЕСЦЕР")
    _minor_divider()
    print(f"Угол наклона пластин: "
          f"{coalescer_packing.angle:.0f}°")
    print(f"Зазор между пластинами: "
          f"{coalescer_packing.coalescer_top_gap * _TO_MM:.0f} мм")
    print(f"Время осаждения капель воды в зазоре: "
          f"{coalescer.droplet_water_settling_time() / SECONDS_PER_MINUTE:.2f} мин")
    print(f"Длина канала: "
          f"{coalescer.coalescer_top_channel_length():.4f} м")

    _minor_divider()
    print("НИЖНИЙ КОАЛЕСЦЕР")
    _minor_divider()
    print(f"Угол наклона пластин: "
          f"{coalescer_packing.angle:.0f}°")
    print(f"Зазор между пластинами: "
          f"{coalescer_packing.coalescer_bottom_gap * _TO_MM:.0f} мм")
    print(f"Время всплытия капель нефти в зазоре: "
          f"{coalescer.droplet_oil_rising_time() / SECONDS_PER_MINUTE:.2f} мин")
    print(f"Длина канала: "
          f"{coalescer.coalescer_bottom_channel_length():.4f} м")
    _minor_divider()
