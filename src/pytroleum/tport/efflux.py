import numpy as np
from numpy.typing import DTypeLike, NDArray

# TODO :
# 1. Fix types (applies to other parts of project too)
# 2. More models for flow rate with lumped parameters
# 3. Generalisation of lumpded parameters models for
#    flow rate computations?


def incompressible(A_orifice: float, A_pipe: float, C: float, rho: float,
                   p1: float, p2: float, pressure_recovery: bool = False) -> float:
    """This function computes flow rate for incompressible flow through orifice.
    Well-known orifice equation is employed for calualtions.

    Parameters
    ----------
    A_orifice
        Area of orifice.
    A_pipe
        Area of pipe.
    C
        Discharge coefficient.
    rho
        Density of fluid.
    p1
        Upstream pressure (before orifice).
    p2
        Downstream pressure (after orifice).
    pressure_recovery, optional
        Variable to specify presence  of pressure recovery, by default False.

    Returns
    -------
        Mass flow rate of incompressible fluid through orifice.
    """

    sign = np.sign(p1-p2)
    dp = np.abs(p1-p2)
    alph = A_orifice/A_pipe

    Q = sign*C*A_orifice*np.sqrt(2*dp/rho/(1-alph**2))
    if pressure_recovery:
        Q = Q/np.sqrt(1-alph)

    G = rho*Q

    return G


def compressible(A: float, C: float, k: float, R: float, rho1: float, rho2: float,
                 T1: float, T2: float, p1: float, p2: float) -> float:
    """This function computes flow rate for compressible flow through orifice.
    Equation for adiabatic flow through nozzle is employed for calculations.

    Parameters
    ----------
    A
        Area of orifice/contriction/nozzle.
    C
        Discharge coefficient.
    k
        Adiabatic index (cp/cv).
    R
        Gas constant.
    rho1
        Upstream density.
    rho2
        Downstream density.
    T1
        Upstream temperature.
    T2
        Downstream temperature.
    p1
        Upstream pressure.
    p2
        Downstream pressure.

    Returns
    -------
        Mass flow rate of compressible fluid through orifice.
    """

    sign = np.sign(p1-p2)
    beta_crit = (2/(k+1))**(k/(k-1))
    beta = (p2/p1)**sign

    # should be ok, np. internal broadcasting will handle this
    beta[beta < beta_crit] = beta_crit

    rho = rho1*(sign >= 0) + rho2*(sign < 0)
    T = T1*(sign >= 0) + T2*(sign < 0)

    G = sign*C*A*rho*np.sqrt(
        2*k/(k-1)*R*T*(beta**(2/k)-beta**((k+1)/k))
    )

    return G


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
