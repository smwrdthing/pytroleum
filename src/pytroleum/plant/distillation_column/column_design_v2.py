"""
Расчёт ректификационной колонны с насадкой Паля для многокомпонентной смеси
по формулам из книги И.А. Александров "Ректификационные и абсорбционные
аппараты", 1971:

  R_мин  - уравнения Андервуда                    (II.35), (II.36), стр. 53-54
  N_мин  - уравнение Фенске-Андервуда              (II.108),          стр. 74
  N      - корреляция Джиллиленда, график II-14     (II.108 -> N),    стр. 74
  D      - диаметр насадочной колонны              (V.5), график V-4, стр. 158-159
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from scipy import interpolate


# плотность воды, кг/м3
WATER_DENSITY = 1000.0

# Для эффективной и устойчивой работы колонны необходимо,
# чтобы её рабочие нагрузки были примерно на 20% ниже точки захлёбывания
ADMISSIBLE_VELOCITY_FACTOR = 0.8

# коэффициент избытка флегмы beta = R/R_мин
BETA = 1.15
# ВЭТТ h_экв для колец Паля, м
H_EKV = 0.60

# коэффициенты пересчёта единиц измерения
HOUR_TO_SEC = 3600.0
# 1 сантипуаз (сПз) = 1e-3 Па*с
CP_TO_PAS = 1.0e-3

# ============================================================
# ГРАФИК V-4: коэффициент k для уравнения (V.5)
# ============================================================

_X_DATA = np.array([
    0.0105, 0.0119, 0.0136, 0.0154, 0.0175, 0.0199, 0.0226, 0.0257,
    0.0293, 0.0333, 0.0378, 0.0430, 0.0488, 0.0555, 0.0631, 0.0717,
    0.0815, 0.0927, 0.1053, 0.1197, 0.1361, 0.1547, 0.1758, 0.1998,
    0.2271, 0.2582, 0.2935, 0.3336, 0.3791, 0.4309, 0.4898, 0.5568,
    0.6322, 0.7193, 0.8176, 0.9339, 1.0059,
])

_K_DATA = np.array([
    0.4542, 0.4533, 0.4512, 0.4431, 0.4375, 0.4322, 0.4245, 0.4213,
    0.4153, 0.4081, 0.3996, 0.3914, 0.3867, 0.3808, 0.3756, 0.3663,
    0.3550, 0.3468, 0.3403, 0.3317, 0.3219, 0.3124, 0.3028, 0.2921,
    0.2835, 0.2734, 0.2627, 0.2519, 0.2412, 0.2308, 0.2206, 0.2073,
    0.1981, 0.1912, 0.1803, 0.1687, 0.1627,
])

_K_INTERPOLATOR = interpolate.interp1d(_X_DATA, _K_DATA, kind="cubic")

# ============================================================
# ГРАФИК II-14 (Джиллиленд)
# X = (R - R_min)/(R + 1),  Y = (N - N_min)/(N + 1)
# ============================================================

_GILLILAND_X = np.array([
    0.004581050604546444, 0.013728644348202312, 0.01712109257999561,
    0.01799505236006904, 0.023814641487721055, 0.03143969829275817,
    0.03829188269217404, 0.04482389827526351, 0.05598725853220188,
    0.07837060445696481, 0.10550274262724568, 0.13342614378469408,
    0.16628819154739694, 0.20123167506769635, 0.23367355827293135,
    0.2685699034116916, 0.3059950341683564, 0.3421627210087156,
    0.37833515343429724, 0.4134070521846014, 0.45067934034471435,
    0.4868456939016065, 0.5263126697746345, 0.5624933956753425,
    0.5994295012833664, 0.6348444297872946, 0.6710220597585957,
    0.7072005936508915, 0.7433779976419442, 0.7795576614354834,
    0.8139644348513969, 0.8519167630423131, 0.8880964268358525,
    0.9242781244516299, 0.9604571103044232, 0.9948561627285067,
])

_GILLILAND_Y = np.array([
    0.978236351973834, 0.9227008128251633, 0.8354442223602578,
    0.8697314522246209, 0.792948994755745, 0.7266315688984915,
    0.6869622051147477, 0.6532860635061837, 0.5987711909291857,
    0.566288066005697, 0.5144408932582999, 0.48008169406521517,
    0.44702078368367615, 0.41105347646159507, 0.39141902143728813,
    0.364375327697846, 0.3354154663087513, 0.3120694468178641,
    0.29190296942587124, 0.26940002664448515, 0.25111579434204356,
    0.22687647492813368, 0.20899081618604431, 0.19438096712878594,
    0.17701476247942338, 0.15818141707336308, 0.14149729531349275,
    0.12541880062007837, 0.10858327209359409, 0.09326181123324984,
    0.07744556951560877, 0.06246748274594771, 0.04714602188560357,
    0.03318722192478529, 0.017411540764599165, 0.011842257535938705,
])

_GILLILAND_INTERPOLATOR = interpolate.interp1d(
    _GILLILAND_X, _GILLILAND_Y, kind="linear")

# Нормальный ряд диаметров колонны, м
STANDARD_DIAMETERS = np.array([
    1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8,
    3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.5, 5.0, 5.5, 6.0,
    6.4, 7.0, 8.0, 9.0,
])


# ============================================================
# СТРУКТУРЫ ДАННЫХ
# ============================================================

@dataclass
class Mixture:
    """Компонентный состав и относительные летучести смеси."""
    names: list[str]        # названия компонентов
    xF: list[float]         # состав, мол. доли
    alpha: list[float]      # относительные летучести


@dataclass
class SectionProps:
    """Физические свойства фаз в сечении колонны (верх/низ)"""
    rho_liq: float    # плотность жидкости, кг/м3
    rho_vap: float    # плотность пара, кг/м3
    mu_liq: float     # динамическая вязкость жидкости, Па*с
    M_avg: float      # средняя мольная масса пара, кг/кмоль


@dataclass
class Packing:
    """Характеристики насадки."""
    name: str
    a: float     # удельная поверхность, м2/м3
    eps: float   # свободный объём F_св, м3/м3


@dataclass
class MaterialBalance:
    D_flow: float            # общий расход дистиллята, кмоль/с
    R_flow: float            # общий расход кубового остатка, кмоль/с
    flow_D: list[float]      # поток компонента в дистилляте, кмоль/с
    flow_R: list[float]      # поток компонента в остатке, кмоль/с
    xD: list[float]          # мольные доли каждого компонента в дистилляте
    xR: list[float]          # мольные доли каждого компонента в остатке


# ============================================================
# 1. МАТЕРИАЛЬНЫЙ БАЛАНС
# ============================================================

def material_balance(F: float, mixture: Mixture,
                     recovery_D: list[float]) -> MaterialBalance:
    """Покомпонентный материальный баланс колонны."""

    # Для каждого компонента: сколько его уходит в дистиллят или остаток.
    # F - общий расход питания (кмоль/с)
    # xF - доля компонента в питании (от 0 до 1)
    # r - доля этого компонента, которую мы хотим извлечь в дистиллят или остаток
    # Например: F * 0.40 * 0.98 = сколько бензола ушло наверх
    flow_D = [F * xF * r for xF, r in zip(mixture.xF, recovery_D)]
    flow_R = [F * xF - d for xF, d in zip(mixture.xF, flow_D)]

    # Суммарный расход дистиллята и остатка
    D_flow = sum(flow_D)
    R_flow = sum(flow_R)

    # Мольная доля каждого компонента в дистилляте/остатке (от 0 до 1)
    xD = [flow / D_flow for flow in flow_D]
    xR = [flow / R_flow for flow in flow_R]

    return MaterialBalance(D_flow, R_flow, flow_D, flow_R, xD, xR)


# ============================================================
# 2. МИНИМАЛЬНОЕ ФЛЕГМОВОЕ ЧИСЛО - уравнения (II.35), (II.36)
# ============================================================

def underwood_theta(alpha: list[float], xF: list[float], q: float) -> list[float]:
    """Корни уравнения Андервуда (II.36):

         p   α_i · xF_i
        Σ  ————————————  =  1 − q
        i=1  α_i − θ

    (величина 1-q в правой части (II.36) книга обозначает e'; для
    стандартного случая ввода питания частично испарённой жидкостью
    e' = 1 - q, где q - мольная доля жидкости в питании).

    Для p компонентов — (p−1) корней, каждый в интервале (α_{i+1}, α_i).
    R_мин и N_мин безразмерны и от выбора единиц измерения потоков не
    зависят, поэтому эта часть расчёта не требует пересчёта в СИ.
    Начальное приближение — середина интервала, решение — fsolve.
    """
    pairs = sorted(zip(alpha, xF), reverse=True)
    alpha_s = [a for a, _ in pairs]
    xF_s = [xf for _, xf in pairs]

    def f(theta: float) -> float:
        return sum(a * x / (a - theta) for a, x in zip(alpha_s, xF_s)) - (1.0 - q)

    thetas = []
    for i in range(len(alpha_s) - 1):
        theta0 = 0.5 * (alpha_s[i] + alpha_s[i + 1])
        thetas.append(float(fsolve(f, theta0)[0]))
    return thetas


def minimum_reflux_underwood(mixture: Mixture, xD: list[float],
                             q: float,
                             light_key: int,
                             heavy_key: int) -> tuple[float, float]:
    """
    Метод Андервуда для многокомпонентной смеси.

    Возвращает (theta, R_min).
    θ — корень (II.36) между α тяжёлого и лёгкого ключевых;
    R_min — из уравнения (II.35):

               p   α_i · xD_i
    R_min + 1 = Σ  ————————————
               i=1  α_i − θ
    """
    alpha = mixture.alpha
    xF = mixture.xF

    alpha_lk = alpha[light_key]
    alpha_hk = alpha[heavy_key]

    thetas = underwood_theta(alpha, xF, q)
    theta = next(t for t in thetas if alpha_hk < t < alpha_lk)

    R_min = sum(a * x / (a - theta) for a, x in zip(alpha, xD)) - 1
    return theta, R_min


# ============================================================
# 3. МИНИМАЛЬНОЕ ЧИСЛО ТЕОРЕТИЧЕСКИХ ТАРЕЛОК - уравнение (II.108)
# ============================================================

def mean_relative_volatility(alpha_lh_top: float, alpha_lh_bottom: float) -> float:
    """Средняя относительная летучесть по колонне (среднегеометрическая).

    По книге (стр. 74): α_l-h = sqrt( α_верх · α_низ )
    """
    return np.sqrt(alpha_lh_top * alpha_lh_bottom)


def minimum_plates_fenske(xD: list[float], xR: list[float],
                          light_key: int, heavy_key: int,
                          alpha_lh: float) -> float:
    """N_min по уравнению Фенске–Андервуда (II.108), стр. 74:

              lg( xD_l/xD_h · xR_h/xR_l )
    N_min =  ——————————————————————————————
                      lg( α_l-h )
    """
    return np.log10((xD[light_key] / xD[heavy_key]) * (xR[heavy_key] / xR[light_key])) \
        / np.log10(alpha_lh)


def working_reflux(R_min: float, beta: float = BETA) -> float:
    """Рабочее флегмовое число R = beta * R_min, beta = R/R_min (стр. 74)."""
    return beta * R_min


def theoretical_plates_gilliland(R: float, R_min: float, N_min: float) -> float:
    """Число теоретических тарелок N по графику II-14 (Джиллиленд), стр. 74.

        X = (R - R_min) / (R + 1)
        Y = (N - N_min) / (N + 1)   ← с графика
        N = (Y + N_min) / (1 - Y)
    """
    X = (R - R_min) / (R + 1)
    Y = float(_GILLILAND_INTERPOLATOR(X))
    return (Y + N_min) / (1.0 - Y)


def packing_height(N: float, h_ekv: float = H_EKV) -> float:
    """Высота слоя насадки H = N * h_экв, м. (формула не менялась)."""
    return N * h_ekv


# ============================================================
# 4. ДИАМЕТР НАСАДОЧНОЙ КОЛОННЫ - уравнение (V.5), стр. 159
# ============================================================

def get_k(L_mass: float, G_mass: float, rho_vap: float, rho_liq: float) -> float:
    """Коэффициент k с графика V-4 (стр. 159).

    X = (L/G) * sqrt(rho_пар / rho_жид) — величина безразмерная, поэтому
    L и G можно передавать в любых одинаковых единицах массового
    расхода (в этом коде - кг/с); результат от единиц не зависит.
    """
    X = (L_mass / G_mass) * np.sqrt(rho_vap / rho_liq)
    return float(_K_INTERPOLATOR(X))


def vapor_velocity(L: float, G: float, props: SectionProps,
                   packing: Packing) -> float:
    """Скорость захлёбывания w, м/с, по формуле (V.5), стр. 159:

        w = 3.14 * k * ( a/F_св^3 * rho_п/rho_ж * mu_ж^0.12 * psi )^(-0.5)

    L, G — потоки жидкости и пара, КМОЛЬ/С (СИ). Внутри функции они
    переводятся в массовые расходы кг/с через M_avg.

    ВАЖНО про вязкость: показатель степени 0.12 в формуле (V.5) — чисто
    эмпирический коэффициент, полученный автором книги при подстановке
    вязкости, выраженной в САНТИПУАЗАХ (сПз = мПа*с = 10^-3 Па*с) —
    единица измерения книги 1971 года, не входящая в современную СИ.
    Чтобы формула считала так же, как в первоисточнике, но при этом
    принимала на вход вязкость в Па*с (СИ), выполняется локальный
    пересчёт mu_liq[Па*с] -> [сПз] непосредственно перед подстановкой:
        mu_cP = mu_liq[Па*с] / CP_TO_PAS   (т.к. 1 сПз = 1e-3 Па*с)
    """
    L_mass = L * props.M_avg       # кмоль/с * кг/кмоль = кг/с
    G_mass = G * props.M_avg       # кг/с
    k = get_k(L_mass, G_mass, props.rho_vap, props.rho_liq)

    # Па*с -> сПз (только для формулы V.5)
    mu_liq_cP = props.mu_liq / CP_TO_PAS

    psi = props.rho_liq / WATER_DENSITY
    packing_ratio = packing.a / packing.eps ** 3
    density_ratio = props.rho_vap / props.rho_liq

    return 3.14 * k * (packing_ratio * density_ratio * mu_liq_cP ** 0.12 * psi) ** (-0.5)


def admissible_vapor_velocity(flooding_velocity: float,
                              factor: float = ADMISSIBLE_VELOCITY_FACTOR) -> float:
    """Допустимая (рабочая) скорость паров, м/с (стр. 158: 20% ниже точки
    захлёбывания)."""
    return factor * flooding_velocity


def vapor_volume_flow(G: float, props: SectionProps) -> float:
    """Объёмный расход пара, м3/с (СИ).

    G — поток пара, КМОЛЬ/С (СИ), поэтому дополнительного деления на 3600
    (как было в исходной версии кода при G в кмоль/ч) не требуется.
    """
    return G * props.M_avg / props.rho_vap   # (кмоль/с * кг/кмоль) / (кг/м3) = м3/с


def calc_column_diameter(vapor_volume_flow: float, vapor_velocity: float) -> float:
    """Расчётный диаметр колонны, м."""
    return np.sqrt(4 * vapor_volume_flow / (np.pi * vapor_velocity))


def select_column_diameter(*diameters: float,
                           nominal_diameters: np.ndarray = STANDARD_DIAMETERS) -> float:
    """Выбор ближайшего большего номинального диаметра из нормального ряда.

    Принимает один или несколько расчётных диаметров (верх, низ и т.д.),
    берёт наибольший и округляет вверх до стандарта.
    """
    d_max = max(diameters)
    for d_nom in sorted(nominal_diameters):
        if d_nom >= d_max:
            return d_nom
    raise ValueError(
        f"Расчётный диаметр {d_max:.3f} м больше "
        f"{max(nominal_diameters):.1f} м из нормального ряда"
    )


def actual_vapor_velocity(vapor_volume: float, diameter: float) -> float:
    """Фактическая скорость пара при принятом диаметре, м/с."""
    return float(4 * vapor_volume / (np.pi * diameter ** 2))


def plot_graph_V4() -> None:
    """График V-4: коэффициент k для уравнения (V.5), лог-лог масштаб."""
    plt.figure(figsize=(8, 6))
    plt.plot(_X_DATA, _K_DATA, "o", label="точки с графика", markersize=5)

    X_fine = np.linspace(_X_DATA[0], _X_DATA[-1], 200)
    plt.plot(X_fine, _K_INTERPOLATOR(X_fine), "-",
             label="интерполяция", linewidth=2)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(0.01, 1)
    plt.ylim(0.1, 1)
    plt.xlabel(r"$X = (L/G)\sqrt{\rho_{\mathrm{п}}/\rho_{\mathrm{ж}}}$")
    plt.ylabel(r"$k$")
    plt.title("Рис. V-4. Коэффициент $k$ для уравнения (V.5)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_graph_gilliland() -> None:
    """График II-14 (Джиллиленд): Y = f(X) по оцифрованным точкам."""
    plt.figure(figsize=(8, 6))
    plt.plot(_GILLILAND_X, _GILLILAND_Y, "o",
             label="точки с графика", markersize=5)

    X_fine = np.linspace(_GILLILAND_X[0], _GILLILAND_X[-1], 200)
    plt.plot(X_fine, _GILLILAND_INTERPOLATOR(X_fine), "-",
             label="интерполяция", linewidth=2)

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel(r"$X = (R - R_{\mathrm{min}})/(R + 1)$")
    plt.ylabel(r"$Y = (N - N_{\mathrm{min}})/(N + 1)$")
    plt.title("Рис. II-14. Корреляция Джиллиленда")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# ИСХОДНЫЕ ДАННЫЕ И ЗАПУСК РАСЧЁТА
# ============================================================

if __name__ == "__main__":

    plot_graph_V4()
    plot_graph_gilliland()

    # --- смесь (бензол - толуол - о-ксилол) ---
    # порядок во всех списках одинаковый:
    #   0 = Бензол,  1 = Толуол,  2 = о-Ксилол
    mixture = Mixture(
        names=["Бензол", "Толуол", "о-Ксилол"],
        xF=[0.40, 0.35, 0.25],
        alpha=[2.40, 1.00, 0.32],   # относительные летучести по толуолу
    )
    light_key = 0   # Бензол
    heavy_key = 1   # Толуол

    # ---- исходные данные в привычных инженерных единицах ----
    # расход питания, КМОЛЬ/Ч (как обычно задают в условии)
    F_h = 100.0
    # доля жидкости в сырье (1 = кип. жидкость)
    q = 1.0
    recovery_D = [0.98, 0.02, 0.005]  # доли извлечения в дистиллят

    beta = BETA                       # коэфф. избытка флегмы R/R_min

    # физические свойства фаз: вязкость обычно задаётся в сантипуазах (сПз)
    mu_liq_top_cP = 0.32
    mu_liq_bot_cP = 0.26

    # --- пересчёт исходных данных в СИ (делается один раз, здесь) ---
    F = F_h / HOUR_TO_SEC              # кмоль/ч -> кмоль/с

    props_top = SectionProps(rho_liq=810.0, rho_vap=2.67,
                             mu_liq=mu_liq_top_cP * CP_TO_PAS,   # сПз -> Па*с
                             M_avg=80.0)
    props_bot = SectionProps(rho_liq=790.0, rho_vap=2.96,
                             mu_liq=mu_liq_bot_cP * CP_TO_PAS,   # сПз -> Па*с
                             M_avg=80.0)

    # --- насадка: кольца Паля стальные 25x25x0.6 ---
    packing = Packing(name="Кольца Паля",
                      a=170.0, eps=0.90)

    # --- α_лк/α_тк в верху и низу колонны (для N_min) ---
    # В реальном расчёте берутся при температурах верха и низа.
    # Здесь для примера — одно и то же значение из mixture.alpha.
    alpha_lh_top = mixture.alpha[light_key] / mixture.alpha[heavy_key]
    alpha_lh_bottom = mixture.alpha[light_key] / mixture.alpha[heavy_key]
    alpha_lh = mean_relative_volatility(alpha_lh_top, alpha_lh_bottom)

    # ---------------- расчёт (всё в СИ: кмоль/с, кг/с, м3/с, м/с, Па*с) ----------------
    mb = material_balance(F, mixture, recovery_D)
    theta, R_min = minimum_reflux_underwood(
        mixture, mb.xD, q, light_key, heavy_key)
    N_min = minimum_plates_fenske(
        mb.xD, mb.xR, light_key, heavy_key, alpha_lh)

    R = working_reflux(R_min, beta)
    N = theoretical_plates_gilliland(R, R_min, N_min)
    H = packing_height(N)

    # ---- потоки верх / низ, кмоль/с ----
    L_top = R * mb.D_flow
    G_top = (R + 1) * mb.D_flow
    L_bot = L_top + F * q
    G_bot = G_top

    # ---- диаметр верха ----
    w_fl_top = vapor_velocity(L_top, G_top, props_top, packing)
    w_dop_top = admissible_vapor_velocity(w_fl_top)
    V_top = vapor_volume_flow(G_top, props_top)
    D_top = calc_column_diameter(V_top, w_dop_top)

    # ---- диаметр низа ----
    w_fl_bot = vapor_velocity(L_bot, G_bot, props_bot, packing)
    w_dop_bot = admissible_vapor_velocity(w_fl_bot)
    V_bot = vapor_volume_flow(G_bot, props_bot)
    D_bot = calc_column_diameter(V_bot, w_dop_bot)

    D_nom = select_column_diameter(D_top, D_bot)

    # ---------------- вывод ----------------
    print("=" * 70)
    print("МАТЕРИАЛЬНЫЙ БАЛАНС (СИ: кмоль/с; в скобках - кмоль/ч)")
    print("=" * 70)
    print(f"{'Компонент':12}{'F,кмоль/с':>13}{'D,кмоль/с':>13}{'xD':>9}"
          f"{'R,кмоль/с':>13}{'xR':>9}")
    for i, name in enumerate(mixture.names):
        print(f"{name:12}{F*mixture.xF[i]:13.5f}{mb.flow_D[i]:13.5f}"
              f"{mb.xD[i]:9.4f}{mb.flow_R[i]:13.5f}{mb.xR[i]:9.4f}")
    print(f"{'Итого':12}{F:13.5f}{mb.D_flow:13.5f}{'':9}{mb.R_flow:13.5f}")
    print(f"(в привычных единицах: F = {F*HOUR_TO_SEC:.2f} кмоль/ч,  "
          f"D = {mb.D_flow*HOUR_TO_SEC:.2f} кмоль/ч,  "
          f"R = {mb.R_flow*HOUR_TO_SEC:.2f} кмоль/ч)")

    print("\n" + "=" * 70)
    print("МИНИМАЛЬНОЕ ФЛЕГМОВОЕ ЧИСЛО (уравнения II.35 - II.36)")
    print("=" * 70)
    print(f"theta = {theta:.4f}  (между alpha_h="
          f"{mixture.alpha[heavy_key]} и alpha_l="
          f"{mixture.alpha[light_key]})")
    print(f"R_min = {R_min:.3f}   (безразмерная величина)")

    print("\n" + "=" * 70)
    print("ЧИСЛО ТЕОРЕТИЧЕСКИХ ТАРЕЛОК")
    print("=" * 70)
    print(f"N_min = {N_min:.2f}  (Фенске, II.108)")
    print(f"R = beta*R_min = {beta}*{R_min:.3f} = {R:.3f}")
    print(f"N = {N:.2f}  (график II-14, Джиллиленд)")

    print("\n" + "=" * 70)
    print("ВЫСОТА СЛОЯ НАСАДКИ")
    print("=" * 70)
    print(f"h_экв = {H_EKV:.2f} м")
    print(f"H = N * h_экв = {N:.2f} * {H_EKV:.2f} = {H:.2f} м")

    print("\n" + "=" * 70)
    print("ДИАМЕТР КОЛОННЫ (насадка - кольца Паля, формула V.5, СИ)")
    print("=" * 70)
    print(f"mu_ж верх = {props_top.mu_liq:.5f} Па*с "
          f"({props_top.mu_liq/CP_TO_PAS:.2f} сПз)")
    print(f"mu_ж низ  = {props_bot.mu_liq:.5f} Па*с "
          f"({props_bot.mu_liq/CP_TO_PAS:.2f} сПз)")
    print(
        f"w_захл верх = {w_fl_top:.4f} м/с,  w_раб верх = {w_dop_top:.4f} м/с")
    print(
        f"w_захл низ  = {w_fl_bot:.4f} м/с,  w_раб низ  = {w_dop_bot:.4f} м/с")
    print(f"V_top = {V_top:.4f} м3/с,   V_bot = {V_bot:.4f} м3/с")
    print(f"D верх = {D_top:.3f} м")
    print(f"D низ  = {D_bot:.3f} м")
    print(f"Принятый (номинальный) диаметр = {D_nom:.1f} м")

    w_act_top = actual_vapor_velocity(V_top, D_nom)
    w_act_bot = actual_vapor_velocity(V_bot, D_nom)
    print(f"Фактическая скорость пара при D_nom: верх = {w_act_top:.4f} м/с, "
          f"низ = {w_act_bot:.4f} м/с")
