"""
Модель расчёта гидроциклона на основе статей:
- Coelho & Medronho (2001): "A model for performance prediction of hydrocyclones"
- Pana-Suppamassadu et al. (2007): "Size Separation of Rubber Particles from
  Natural Rubber Latex by Hydrocyclone Technique"

Все размерности в системе СИ:
- длина: м
- давление: Па
- расход: м³/с
- плотность: кг/м³
- вязкость: Па·с
- концентрации Cv: доля (0-1)

"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Tuple, Literal, List, Optional

# Константы модели для разных типов гидроциклонов
MODEL_PARAMS = {
    'rietema': (4.23, 2.45),
    'bradley': (5.10, 3.12),
    'demco': (5.40, 3.30)
}

type HydrocycloneType = Literal['rietema', 'bradley', 'demco']


@dataclass
class GeometryParameters:
    """Геометрические параметры гидроциклона"""
    Dc: float  # Диаметр циклона, м
    Di: float  # Диаметр входа, м
    Do: float  # Диаметр вихреуловителя, м
    Du: float  # Диаметр нижнего слива, м
    L: float   # Длина гидроциклона, м
    l_vortex: float  # Выбег вихреуловителя, м
    theta: float = 15.0  # Угол конуса, градусы

    def check_proportions(self) -> List[str]:
        """
        Проверка соответствия геометрических пропорций допустимому диапазону.
        """
        violations = []

        # Вычисляем пропорции
        Di_Dc = self.Di / self.Dc
        Do_Dc = self.Do / self.Dc
        Du_Dc = self.Du / self.Dc
        l_Dc = self.l_vortex / self.Dc
        L_Dc = self.L / self.Dc

        # Проверяем каждую пропорцию
        if not (0.14 <= Di_Dc <= 0.28):
            violations.append(f"Di/Dc={Di_Dc:.3f} вне диапазона [0.14-0.28]")

        if not (0.20 <= Do_Dc <= 0.34):
            violations.append(f"Do/Dc={Do_Dc:.3f} вне диапазона [0.20-0.34]")

        if not (0.04 <= Du_Dc <= 0.28):
            violations.append(f"Du/Dc={Du_Dc:.3f} вне диапазона [0.04-0.28]")

        if not (0.33 <= l_Dc <= 0.55):
            violations.append(f"l/Dc={l_Dc:.3f} вне диапазона [0.33-0.55]")

        if not (3.30 <= L_Dc <= 6.93):
            violations.append(f"L/Dc={L_Dc:.3f} вне диапазона [3.30-6.93]")

        if not (9.0 <= self.theta <= 20.0):
            violations.append(f"θ={self.theta:.1f}° вне диапазона [9°-20°]")

        return violations

    def get_proportions(self) -> Dict[str, float]:
        """Возвращает словарь со всеми пропорциями"""
        return {
            'Di/Dc': self.Di / self.Dc,
            'Do/Dc': self.Do / self.Dc,
            'Du/Dc': self.Du / self.Dc,
            'l/Dc': self.l_vortex / self.Dc,
            'L/Dc': self.L / self.Dc,
            'θ': self.theta
        }


@dataclass
class PhysicalProperties:
    """Физические свойства среды"""
    mu: float    # Динамическая вязкость жидкости, Па·с
    rho: float   # Плотность жидкости, кг/м³
    rhos: float  # Плотность твёрдой фазы, кг/м³


@dataclass
class OperatingParametersQ:
    """Рабочие параметры гидроциклона при заданном расходе"""
    Q: float   # Входной расход, м³/с
    Cv: float  # Концентрация твёрдых частиц на входе, доля


@dataclass
class OperatingParametersDeltaP:
    """Рабочие параметры гидроциклона при заданном перепаде давления"""
    delta_p: float  # Перепад давления, Па
    Cv: float       # Концентрация твёрдых частиц на входе, доля


OperatingParameters = OperatingParametersQ | OperatingParametersDeltaP


@dataclass
class HydrocycloneConfig:
    """Конфигурация гидроциклона для расчёта"""
    name: str
    geometry: GeometryParameters
    htype: HydrocycloneType


def _euler_number(
    geometry: GeometryParameters,
    Cv: float,
    Re: float,
) -> float:
    """
    Расчёт числа Эйлера Eu
    """
    L_minus_l = geometry.L - geometry.l_vortex
    return (43.5 * geometry.Dc**0.57 *
            (geometry.Dc / geometry.Di)**2.61 *
            (geometry.Dc / (geometry.Do**2 + geometry.Du**2))**0.42 *
            (geometry.Dc / L_minus_l)**0.98 *
            Re**0.12 * np.exp(-0.51 * Cv))


def calculate_hydrocyclone_parameters(
    geometry: GeometryParameters,
    properties: PhysicalProperties,
    operating: OperatingParameters,
    model_params: Tuple[float, float],
) -> Dict:
    """
    Расчёт параметров гидроциклона при заданном расходе Q или перепаде давления ΔP.
    """
    alpha, m = model_params
    L_minus_l = geometry.L - geometry.l_vortex
    Cv = operating.Cv

    # Коэффициент K: Q = K * ΔP^0.472
    K = (0.184 * geometry.Dc**(-0.217) *
         geometry.Di**(1.231) *
         (geometry.Do**2 + geometry.Du**2)**0.198 *
         L_minus_l**0.462 *
         properties.mu**(0.0566) *
         properties.rho**(-0.528) *
         np.exp(0.241 * Cv))

    match operating:
        case OperatingParametersQ(Q=Q):
            delta_p = (Q / K) ** (1 / 0.472)
        case OperatingParametersDeltaP(delta_p=delta_p):
            Q = K * delta_p**0.472

    # Число Рейнольдса Re
    Re = (4 * properties.rho * Q) / (np.pi * properties.mu * geometry.Dc)

    Eu = _euler_number(geometry, Cv, Re)

    # Водное отношение Rw
    Rw = (1.18 * (geometry.Dc / geometry.Do)**5.97 *
          (geometry.Du / geometry.Dc)**3.10 *
          Eu**(-0.54))

    # Приведённый отсечной размер частицы
    rhos_minus_rho = properties.rhos - properties.rho

    reduced_cut_size = (1.173 * geometry.Dc**0.64 /
                        (geometry.Do**0.475 * L_minus_l**0.665) *
                        np.sqrt((properties.mu * properties.rho * Q) /
                                (rhos_minus_rho * delta_p)) *
                        np.log(1 / Rw)**0.395 * np.exp(6.0 * Cv))

    return {
        'Q': Q,
        'delta_p': delta_p,
        'Rw': Rw,
        'Re': Re,
        'Eu': Eu,
        'reduced_cutoff_diameter': reduced_cut_size,
        'alpha': alpha,
        'm': m,
    }


def calculate_reduced_grade_efficiency(
    d: np.ndarray,
    reduced_cut_size: float,
    model: Literal['plitt', 'lynch_rao'],
    m: float,
    alpha: float,
) -> np.ndarray:
    """
    Расчёт приведённой вероятности уноса G'(d).
    """
    ratio = d / reduced_cut_size

    match model:
        case 'plitt':
            return 1 - np.exp(-0.693 * ratio**m)
        case 'lynch_rao':
            exp_term = np.exp(alpha * ratio)
            return (exp_term - 1) / (exp_term + np.exp(alpha) - 2)
        case _:
            raise ValueError(f"Неизвестная модель: {model}")


def calculate_cumulative_particle_size_distribution(
    d: np.ndarray,
    k: float,
    n: float,
) -> np.ndarray:
    """
    Кумулятивное распределение частиц по Розин-Раммлеру.
    """
    return 1.0 - np.exp(-(d / k)**n)


def cumulative_particle_size_distribution_derivative(
    d: np.ndarray,
    k: float,
    n: float,
) -> np.ndarray:
    """
    Производная кумулятивного распределения Розин-Раммлера.
    """
    ratio = d / k
    return (n / k) * ratio**(n - 1) * np.exp(-(ratio)**n)


def calculate_reduced_total_efficiency(
    d: np.ndarray,
    k: float,
    n: float,
    reduced_grade_efficiency: np.ndarray,
) -> float:
    """
    Расчёт приведённой полной эффективности E_T'.
    """
    dy_dd = cumulative_particle_size_distribution_derivative(d, k, n)
    return float(np.trapezoid(reduced_grade_efficiency * dy_dd, d))


def calculate_total_efficiency(
    reduced_total_efficiency: float,
    water_flow_ratio: float,
) -> float:
    """
    Расчёт полной эффективности E_T по приведённой E_T'.
    """
    return reduced_total_efficiency * (1 - water_flow_ratio) + water_flow_ratio


def print_proportions(geometry: GeometryParameters, name: str) -> None:
    """
    Вывод геометрических пропорций и проверка их соответствия диапазонам.
    """
    props = geometry.get_proportions()
    violations = geometry.check_proportions()

    print(f"\n=== {name} proportions ===")
    print(f"Di/Dc = {props['Di/Dc']:.3f} [0.14-0.28]")
    print(f"Do/Dc = {props['Do/Dc']:.3f} [0.20-0.34]")
    print(f"Du/Dc = {props['Du/Dc']:.3f} [0.04-0.28]")
    print(f"l/Dc  = {props['l/Dc']:.3f} [0.33-0.55]")
    print(f"L/Dc  = {props['L/Dc']:.3f} [3.30-6.93]")
    print(f"θ     = {props['θ']:.1f}° [9°-20°]")

    if violations:
        print("Пропорции нарушены:")
        for v in violations:
            print(f"  {v}")
    else:
        print("Все пропорции в допустимых диапазонах")
    print()


def _plot_row(
    axes_row,
    config: HydrocycloneConfig,
    results: Dict,
    d: np.ndarray,
    k: float,
    n: float,
    mode: Literal['Q', 'delta_p'],
) -> None:
    """Рисует строку из трёх графиков для одного типа гидроциклона."""
    reduced_cut_size = results['reduced_cutoff_diameter']
    Rw = results['Rw']
    m = results['m']
    alpha = results['alpha']
    Q_res = results['Q']
    delta_p_res = results['delta_p']

    d_norm = d / reduced_cut_size

    # 1. Кумулятивное распределение
    ax = axes_row[0]
    y = calculate_cumulative_particle_size_distribution(d, k, n)
    ax.plot(d_norm, y, 'b-', linewidth=2)
    ax.set_xlabel("$d/d_{50}'$", fontsize=10)
    ax.set_ylabel('y(d)', fontsize=10)
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 1.2])
    ax.grid(True, alpha=0.3)
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    ax.text(
        0.95, 0.05,
        f'{config.name}\nk={k*1e6:.0f} мкм\nn={n:.1f}',
        transform=ax.transAxes, fontsize=9,
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5),
    )

    # 2. Плотность распределения
    ax = axes_row[1]
    dy_dd = cumulative_particle_size_distribution_derivative(d, k, n)
    dy_dd_scaled = dy_dd * reduced_cut_size
    ax.plot(d_norm, dy_dd_scaled, 'b-', linewidth=2)
    ax.set_xlabel("$d/d_{50}'$", fontsize=10)
    ax.set_ylabel("$dy/d(d/d_{50}')$", fontsize=10)
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 0.3])
    ax.grid(True, alpha=0.3)
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    ax.text(
        0.95, 0.95,
        f'{config.name}\nk={k*1e6:.0f} мкм\nn={n:.1f}',
        transform=ax.transAxes, fontsize=9,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5),
    )

    # 3. Вероятность уноса
    ax = axes_row[2]
    reduced_grade_efficiency = calculate_reduced_grade_efficiency(
        d, reduced_cut_size, 'plitt', m, alpha)
    G = reduced_grade_efficiency * (1 - Rw) + Rw

    ax.plot(d_norm, reduced_grade_efficiency, 'r-',
            linewidth=2, label="$G'(d/d_{50}')$")
    ax.plot(d_norm, G, 'b--', linewidth=2, label="$G(d/d_{50}')$")
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    ax.set_xlabel("$d/d_{50}'$", fontsize=10)
    ax.set_ylabel("$G(d)$, $G'(d)$", fontsize=10)
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 1.2])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    if mode == 'Q':
        info_text = (
            f'{config.name}\n'
            f"$d_{{50}}'$={reduced_cut_size*1e6:.1f} мкм\n"
            f'm={m:.2f}\n'
            f'$R_w$={Rw:.3f}\n'
            f'Q={Q_res*1000*60:.1f} л/мин'
        )
    else:
        info_text = (
            f'{config.name}\n'
            f"$d_{{50}}'$={reduced_cut_size*1e6:.1f} мкм\n"
            f'm={m:.2f}\n'
            f'$R_w$={Rw:.3f}\n'
            f'ΔP={delta_p_res/1000:.2f} кПа'
        )

    ax.text(
        0.95, 0.05, info_text,
        transform=ax.transAxes, fontsize=9,
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5),
    )

    # Расчёт эффективностей
    reduced_total_efficiency = calculate_reduced_total_efficiency(
        d, k, n, reduced_grade_efficiency)
    total_efficiency = calculate_total_efficiency(reduced_total_efficiency, Rw)

    print(f"=== {config.name} ===")
    print(f"Q = {Q_res*1000*60:.2f} л/мин, ΔP = {delta_p_res/1000:.2f} кПа")
    print(f"E_T' = {reduced_total_efficiency*100:.1f}%")
    print(f"E_T  = {total_efficiency*100:.1f}%")
    print()


def _build_configs(Dc: float) -> List[HydrocycloneConfig]:
    """
    Возвращает список стандартных конфигураций гидроциклонов
    с пропорциями, строго соответствующими допустимым диапазонам.

    """

    # Для Rietema
    rietema_props = {
        'Dc': Dc,
        'Di': Dc * 0.20,      # Di/Dc = 0.20
        'Do': Dc * 0.25,      # Do/Dc = 0.25
        'Du': Dc * 0.15,      # Du/Dc = 0.15
        'L': Dc * 4.50,       # L/Dc = 4.50
        'l_vortex': Dc * 0.40,  # l/Dc = 0.40
        'theta': 15.0
    }

    # Для Bradley
    bradley_props = {
        'Dc': Dc,
        'Di': Dc * 0.16,      # Di/Dc = 0.16
        'Do': Dc * 0.22,      # Do/Dc = 0.22
        'Du': Dc * 0.08,      # Du/Dc = 0.08
        'L': Dc * 5.50,       # L/Dc = 5.50
        'l_vortex': Dc * 0.45,  # l/Dc = 0.45
        'theta': 12.0
    }

    # Для Demco
    demco_props = {
        'Dc': Dc,
        'Di': Dc * 0.25,      # Di/Dc = 0.25
        'Do': Dc * 0.30,      # Do/Dc = 0.30
        'Du': Dc * 0.20,      # Du/Dc = 0.20
        'L': Dc * 5.00,       # L/Dc = 5.00
        'l_vortex': Dc * 0.50,  # l/Dc = 0.50
        'theta': 18.0
    }

    configs = [
        HydrocycloneConfig(
            name='Rietema',
            geometry=GeometryParameters(**rietema_props),
            htype='rietema',
        ),
        HydrocycloneConfig(
            name='Bradley',
            geometry=GeometryParameters(**bradley_props),
            htype='bradley',
        ),
        HydrocycloneConfig(
            name='Demco',
            geometry=GeometryParameters(**demco_props),
            htype='demco',
        ),
    ]

    # Выводим пропорции для проверки
    print("\n" + "="*60)
    print(f"ПРОВЕРКА ГЕОМЕТРИЧЕСКИХ ПРОПОРЦИЙ (Dc = {Dc*1000:.1f} мм)")
    print("="*60)
    for config in configs:
        print_proportions(config.geometry, config.name)

    return configs


def plot_hydrocyclone_analysis_Q(Q: float, Cv: float, Dc: float) -> None:
    """
    Построение графиков анализа гидроциклонов при заданном расходе.
    """
    operating = OperatingParametersQ(Q=Q, Cv=Cv)
    _plot_hydrocyclone_analysis(operating, mode='Q', Dc=Dc)


def plot_hydrocyclone_analysis_delta_p(delta_p: float, Cv: float, Dc: float) -> None:
    """
    Построение графиков анализа гидроциклонов при заданном перепаде давления.
    """
    operating = OperatingParametersDeltaP(delta_p=delta_p, Cv=Cv)
    _plot_hydrocyclone_analysis(operating, mode='delta_p', Dc=Dc)


def _plot_hydrocyclone_analysis(
    operating: OperatingParameters,
    mode: Literal['Q', 'delta_p'],
    Dc: float,
) -> None:
    """Общая логика построения графиков."""
    configs = _build_configs(Dc)

    properties = PhysicalProperties(
        mu=0.001,   # Па·с, вода
        rho=1000,   # кг/м³, вода
        rhos=2650,  # кг/м³, песок
    )

    k = 50e-6  # характерный размер Розин-Раммлера, м
    n = 1.5    # параметр формы Розин-Раммлера
    d = np.linspace(0, 1e-3, 500)

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))

    for row, config in enumerate(configs):
        results = calculate_hydrocyclone_parameters(
            geometry=config.geometry,
            properties=properties,
            operating=operating,
            model_params=MODEL_PARAMS[config.htype],
        )
        _plot_row(axes[row], config, results, d, k, n, mode)

    plt.tight_layout()
    plt.show()


def plot_contour_graphs():
    """Контурные графики для трёх типов циклонов"""
    # Диапазоны
    Dc = np.linspace(0.01, 0.08, 50)
    Q = np.linspace(0.0001, 0.005, 50)
    Q_grid, Dc_grid = np.meshgrid(Q, Dc)

    # Параметры
    properties = PhysicalProperties(0.001, 1000, 2650)
    Cv = 0.05
    k = 50e-6
    n = 1.5
    d = np.linspace(0, 1e-3, 500)

    # Конфигурации циклонов
    configs = _build_configs(Dc[0])

    # Графики
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))

    for row, config in enumerate(configs):
        props = config.geometry.get_proportions()

        # Массивы для результатов
        reduced_cutoff_results = np.zeros_like(Dc_grid)
        Rw_results = np.zeros_like(Dc_grid)
        efficiency_results = np.zeros_like(Dc_grid)

        # Расчёт для всех точек сетки
        for i in range(50):
            for j in range(50):
                geometry = GeometryParameters(
                    Dc=Dc_grid[i, j],
                    Di=Dc_grid[i, j] * props['Di/Dc'],
                    Do=Dc_grid[i, j] * props['Do/Dc'],
                    Du=Dc_grid[i, j] * props['Du/Dc'],
                    L=Dc_grid[i, j] * props['L/Dc'],
                    l_vortex=Dc_grid[i, j] * props['l/Dc'],
                    theta=props['θ']
                )
                operating = OperatingParametersQ(Q_grid[i, j], Cv)
                results = calculate_hydrocyclone_parameters(
                    geometry, properties, operating, MODEL_PARAMS[config.htype]
                )

                # Сохраняем результаты
                # в мкм
                reduced_cutoff_results[i,
                                       j] = results['reduced_cutoff_diameter'] * 1e6
                Rw_results[i, j] = results['Rw']

                # Расчёт приведённой эффективности
                reduced_grade = calculate_reduced_grade_efficiency(
                    d, results['reduced_cutoff_diameter'], 'plitt',
                    results['m'], results['alpha']
                )
                efficiency_results[i, j] = calculate_reduced_total_efficiency(
                    d, k, n, reduced_grade) * 100

        # Первый график - отсечной размер
        ax = axes[row, 0]
        contour = ax.contourf(
            Q_grid*1000*60, Dc_grid*1000, reduced_cutoff_results,
            levels=20, cmap='viridis')
        plt.colorbar(contour, ax=ax, label="$d_{50}'$, мкм")
        ax.set_xlabel('$Q$, л/мин')
        ax.set_ylabel('$D_c$, мм')
        if row == 0:
            ax.set_title('Отсечной размер')
        # Добавляем название циклона слева от первого графика в строке
        ax.text(-0.3, 0.5, config.name, transform=ax.transAxes,
                fontsize=12, fontweight='bold', rotation=90, va='center')
        ax.grid(True, alpha=0.3)

        # Второй график - водное отношение
        ax = axes[row, 1]
        contour = ax.contourf(
            Q_grid*1000*60, Dc_grid*1000, Rw_results,
            levels=20, cmap='viridis')
        plt.colorbar(contour, ax=ax, label='$R_w$')
        ax.set_xlabel('$Q$, л/мин')
        ax.set_ylabel('$D_c$, мм')
        if row == 0:
            ax.set_title('Водное отношение')
        ax.grid(True, alpha=0.3)

        # Третий график - приведённая эффективность
        ax = axes[row, 2]
        contour = ax.contourf(
            Q_grid*1000*60, Dc_grid*1000, efficiency_results,
            levels=20, cmap='RdYlGn')
        plt.colorbar(contour, ax=ax, label="$E_T'$, %")
        ax.set_xlabel('$Q$, л/мин')
        ax.set_ylabel('$D_c$, мм')
        if row == 0:
            ax.set_title('Приведённая эффективность')
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Характеристики гидроциклонов ($C_v = {Cv}$)', fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("МОДЕЛЬ РАСЧЁТА ГИДРОЦИКЛОНА")
    print("="*60)

    # print("\n[Режим 1] Фиксированный перепад давления")
    # plot_hydrocyclone_analysis_delta_p(delta_p=100000, Cv=0.05, Dc=50e-3)

    print("\n[Режим 2] Фиксированный расход")
    plot_hydrocyclone_analysis_Q(Q=0.001, Cv=0.05, Dc=75e-3)

    print("\n[Контурные графики для трёх циклонов вместе]")
    plot_contour_graphs()
