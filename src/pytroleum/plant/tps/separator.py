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

    def volume_separator(self) -> float:
        # NOTE у нас есть геометрический модуль, там есть функции для расчёта объёма,
        # NOTE можно взять оттуда
        """Номинальный объём сепаратора, м³"""
        return (np.pi * self.design.inner_diameter ** 2 / 4 *
                self.design.length_cylindrical_part +
                2 * self.design.volume_ell_head)

    def residence_time(self) -> float:
        """Время пребывания жидкости в сепараторе, с"""
        # NOTE если volume_separator() возвращает всегда что-то постоянное -
        # NOTE то вместо метода лучше сделать это в виде атрибута
        # NOTE self.volume <- полный объём тут
        # NOTE
        # NOTE в методах мы обычно хотим реализовать какую-то сложную логику, связанную с
        # NOTE объектом, если нам нужно получить свойство объекта - его можно просто
        # NOTE записать как атрибут
        # NOTE
        # NOTE В других языках иногда методы используются для безопасного доступа к
        # NOTE приватным атрибутам (см. "инкапсуляция") - там классы полны методов вроде
        # NOTE getX и setX, в python настоящей инкапсуляции нет, потому что всё публичное
        # NOTE поэтому чаще всего атрибуты просто оставляют как есть
        # NOTE вместо self.getX() просто self.X
        # NOTE вместо self.setX(new_X) просто self.X = new_X и т.д.

        return (self.volume_separator() * self.design.fill_coeff /
                self.conditions.flow_liquid)

    def capacity(self):
        """Максимальная производительность аппарата по жидкости с
        учетом коэффициента заполнения"""
        return (self.volume_separator() * self.design.fill_coeff /
                self.residence_time())

    # --- Первая секция (Н+В) ---

    def volume_first_section(self) -> float:
        """Объём первой секции, м³"""
        # NOTE это тоже можно сделать атрибутом вместо метода
        return (np.pi * self.design.inner_diameter ** 2 / 4 *
                self.design.length_first_section)

    def residence_time_first_section(self) -> float:
        """Время пребывания жидкости в первой секции, с"""
        return (self.volume_first_section() * self.design.fill_coeff_first_section /
                self.conditions.flow_liquid)

    def capacity_first_section(self) -> float:
        """Пропускная способность первой секции, м³/с"""
        return (self.volume_first_section() * self.design.fill_coeff_first_section /
                self.residence_time_first_section())

    # --- Сборник нефти после перегородки ---

    def volume_after_wall(self) -> float:
        """Объём сборника нефти после перегородки, м³"""
        return (np.pi * self.design.inner_diameter ** 2 / 4 *
                self.design.length_section_after_wall +
                self.design.volume_ell_head)

    def residence_time_after_wall(self) -> float:
        """Время пребывания нефти в сборнике после перегородки, с"""
        return self.residence_time() - self.residence_time_first_section()

    def capacity_after_wall(self) -> float:
        """Пропускная способность сборника нефти после перегородки, м³/с"""
        return (self.volume_after_wall() * self.design.fill_coeff_after_wall /
                self.residence_time_after_wall())

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

    # --- Осаждение капель воды ---
    def velocity_water_settling(self) -> float:
        """Скорость осаждения капель воды в слое нефти"""
        return (self.diameter_water_droplet**2 *
                (self.properties.water_density-self.properties.oil_density)*g /
                (18*self.properties.viscosity_oil))

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
        fill_coeff_after_wall=0.858,
        fill_coeff_first_section=0.858,
        volume_ell_head=1.294,
        length_first_section=8.2,
        length_section_after_wall=1.3,
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
    print(f"Объём эллиптического днища: "
          f"{design.volume_ell_head:.3f} м³")
    print(f"Объёмный расход жидкости: "
          f"{conditions.flow_liquid * SECONDS_PER_DAY:.1f} м³/сут")
    print(f"Коэффициент заполнения: "
          f"{design.fill_coeff * PERCENT:.1f} %")
    print(f"Номинальный объём сепаратора: "
          f"{separator.volume_separator():.3f} м³")
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
          f"{design.length_section_after_wall:.1f} м")
    print(f"Коэффициент заполнения: "
          f"{design.fill_coeff_after_wall * PERCENT:.1f} %")
    print(f"Объём секции после перегородки: "
          f"{separator.volume_after_wall():.3f} м³")
    print(f"Время пребывания: "
          f"{separator.residence_time_after_wall() / SECONDS_PER_MINUTE:.3f} мин")
    print(f"Пропускная способность: "
          f"{separator.capacity_after_wall() * SECONDS_PER_DAY:.3f} м³/сут")

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
