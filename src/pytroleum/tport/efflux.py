import numpy as np

from typing import overload
from numpy.typing import NDArray
from numpy import float64

# TODO (maybe?) :
# 1. Those functions are a bit messy, throwing ton of arguments is rather complicated, try
#    to find workaround?
# 2. Generalisation of lumpded parameters models for flow rate computations?


@overload
def incompressible(
        area_of_orifice: float | float64,
        area_of_pipe: float | float64,
        discharge_coefficient: float | float64,
        density: float | float64,
        upstream_pressure: float | float64,
        donwstream_pressure: float | float64,
        pressure_recovery: bool = False) -> float | float64:
    ...


@overload
def incompressible(
        area_of_orifice: float | float64,
        area_of_pipe: float | float64,
        discharge_coefficient: float | float64,
        density: NDArray[float64],
        upstream_pressure: float | float64,
        donwstream_pressure: float | float64,
        pressure_recovery: bool = False) -> NDArray[float64]:
    ...


@overload
def incompressible(
        area_of_orifice: float | float64,
        area_of_pipe: float | float64,
        discharge_coefficient: float | float64,
        density: float | float64,
        upstream_pressure: NDArray[float64],
        donwstream_pressure: float | float64,
        pressure_recovery: bool = False) -> NDArray[float64]:
    ...


@overload
def incompressible(
        area_of_orifice: float | float64,
        area_of_pipe: float | float64,
        discharge_coefficient: float | float64,
        density: float | float64,
        upstream_pressure: float | float64,
        donwstream_pressure: NDArray[float64],
        pressure_recovery: bool = False) -> NDArray[float64]:
    ...


def incompressible(
        area_of_orifice,
        area_of_pipe,
        discharge_coefficient,
        density,
        upstream_pressure,
        donwstream_pressure,
        pressure_recovery=False):
    """This function computes flow rate for incompressible flow through orifice.
    Well-known orifice equation is employed for calualtions.

    Parameters
    ----------
    area_of_orifice
        Area of orifice.
    area_of_pipe
        Area of pipe.
    discharge_coefficient
        Discharge coefficient.
    density
        Density of fluid.
    upstream_pressure
        Upstream pressure (before orifice).
    donwstream_pressure
        Downstream pressure (after orifice).
    pressure_recovery, optional
        Variable to specify presence  of pressure recovery, by default False.

    Returns
    -------
    mass_flow_rate
        Mass flow rate of incompressible fluid through orifice.
    """

    sign = np.sign(upstream_pressure-donwstream_pressure)
    pressure_diff = np.abs(upstream_pressure-donwstream_pressure)
    area_frac = area_of_orifice/area_of_pipe

    volume_flow_rate = (
        sign*discharge_coefficient *
        area_of_orifice * np.sqrt(2*pressure_diff/density/(1-area_frac**2))
    )
    if pressure_recovery:
        volume_flow_rate = volume_flow_rate/np.sqrt(1-area_frac)

    mass_flow_rate = density*volume_flow_rate

    return mass_flow_rate


@overload
def compressible(
        area: float | float64,
        discharge_coefficient: float | float64,
        adiabatic_index: float | float64,
        gas_constant: float | float64,
        upstream_density: float | float64,
        upstream_temperature: float | float64,
        upstream_pressure: float | float64,
        downstream_density: float | float64,
        downstream_temperature: float | float64,
        downstream_pressure: float | float64) -> float | float64:
    ...


@overload
def compressible(
        area: float | float64,
        discharge_coefficient: float | float64,
        adiabatic_index: float | float64,
        gas_constant: float | float64,
        upstream_density: NDArray[float64],
        upstream_temperature: NDArray[float64],
        upstream_pressure: NDArray[float64],
        downstream_density: float | float64,
        downstream_temperature: float | float64,
        downstream_pressure: float | float64) -> NDArray[float64]:
    ...


@overload
def compressible(
        area: float | float64,
        discharge_coefficient: float | float64,
        adiabatic_index: float | float64,
        gas_constant: float | float64,
        upstream_density: float | float64,
        upstream_temperature: float | float64,
        upstream_pressure: float | float64,
        downstream_density: NDArray[float64],
        downstream_temperature: NDArray[float64],
        downstream_pressure: NDArray[float64]) -> NDArray[float64]:
    ...


def compressible(
        area,
        discharge_coefficient,
        adiabatic_index,
        gas_constant,
        upstream_density,
        upstream_temperature,
        upstream_pressure,
        downstream_density,
        downstream_temperature,
        downstream_pressure):
    """This function computes flow rate for compressible flow through orifice.
    Equation for adiabatic flow through nozzle is employed for calculations.

    Parameters
    ----------
    area
        Area of orifice/contriction/nozzle.
    discharge_coefficient
        Discharge coefficient.
    adiabatic_index
        Adiabatic index (cp/cv).
    gas_constant
        Gas constant.
    upstream_density
        Upstream density.
    upstream_temperature
        Upstream temperature.
    upstream_pressure
        Upstream pressure.
    downstream_density
        Downstream density.
    downstream_temperature
        Downstream temperature.
    downstream_pressure
        Downstream pressure.

    Returns
    -------
    mass_flow_rate
        Mass flow rate of compressible fluid through orifice.
    """

    # TODO :
    # rewrite to exclude density from consideration + adjust wherever this is used

    sign = np.sign(upstream_pressure-downstream_pressure)
    beta_crit = (2/(adiabatic_index+1))**(adiabatic_index/(adiabatic_index-1))
    beta = (downstream_pressure/upstream_pressure)**sign

    # should be ok, np. internal broadcasting will handle this
    beta[beta < beta_crit] = beta_crit

    rho = upstream_density*(sign >= 0) + downstream_density*(sign < 0)
    temperature = (
        upstream_temperature * (sign >= 0) + downstream_temperature*(sign < 0))

    mass_flow_rate = sign*discharge_coefficient*area*rho*np.sqrt(
        2*adiabatic_index/(adiabatic_index-1)*gas_constant*temperature*(
            beta ** (2/adiabatic_index)-beta**((adiabatic_index+1)/adiabatic_index))
    )

    return mass_flow_rate


if __name__ == '__main__':

    # NOTE : type annotations are tricky when we want function to work both with
    # numpy arrays and default python floats. Things get even harder with numpy internal
    # type conversions to dtypes. Look into typing later.

    import matplotlib.pyplot as plt

    D, Dp = 50e-3, 100e-3
    A, Ap = np.pi*D**2/4, np.pi*Dp**2/4
    C = 0.61

    p1 = np.linspace(1e5, (20+1)*1e5, 500)
    p2 = np.mean(p1)
    dp = p1-p2

    rho_inc = 1000
    G_inc = incompressible(A, Ap, C, rho_inc, p1, p2)  # type: ignore
    G_inc_norm = G_inc/np.max(G_inc)
    v_inc = G_inc_norm/A/rho_inc

    T_comp = 300, 300
    k = 1.4
    R = 287
    # constant values yiel crytical flow
    rho_comp = 1.16, 1.16
    # however when rho = rho(p,T) behavior is more like incompressible
    # rho_comp = p1/R/T_comp[0], p2/R/T_comp[0]

    G_comp = compressible(
        A, C, k, R, *rho_comp, *T_comp, p1, p2)  # type: ignore
    G_comp_norm = G_comp/np.max(G_comp)

    fig, ax = plt.subplots(ncols=2)
    fig.set_size_inches(10, 4.5)
    ax[0].set_title('Incompressibel flow')
    ax[0].set_xlabel(r'$\Delta p$ [bar]')
    ax[0].set_ylabel('G [kg/s]')
    ax[0].grid(True)
    ax[0].plot(dp/1e5, G_inc)

    ax[1].set_title('Compressible flow')
    ax[1].set_xlabel(r'$\Delta p$ [bar]')
    ax[1].set_ylabel('G [kg/s]')
    ax[1].grid(True)
    ax[1].plot(dp/1e5, G_comp, 'C1')

    plt.show()
