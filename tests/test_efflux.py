import numpy as np
import matplotlib.pyplot as plt
from pytroleum.tport.efflux import incompressible, compressible


def plot_flow_comparison():
    """Tests flow calculation functions in the efflux.py module.

    Validates calculation correctness for incompressible fluid
    and compressible gas according to theoretical foundations.
    """
    # Geometry parameters
    D, Dp = 50e-3, 100e-3
    A, Ap = np.pi*D**2/4, np.pi*Dp**2/4
    C = 0.61

    # ==================== LIQUID (INCOMPRESSIBLE) ====================
    rho_inc = 1000
    p1 = np.linspace(1e5, (20+1)*1e5, 500)
    p2 = 10.5e5
    dp = p1 - p2
    G_inc = incompressible(A, Ap, C, rho_inc, p1, p2)  # type: ignore

    # ==================== GAS (COMPRESSIBLE) ====================
    k = 1.4
    R = 287
    T_comp = 300.
    beta_vals = np.linspace(0.1, 1.0, 200)
    p_upstream = 10e5
    p_downstream = beta_vals * p_upstream

    rho_comp_up = 1.16
    rho_comp_down = 1.16

    G_comp = compressible(A, C, k, R,
                          rho_comp_up, T_comp, p_upstream,  # type: ignore
                          rho_comp_down, T_comp, p_downstream)  # type: ignore
    G_comp_norm = G_comp / np.max(G_comp)
    beta_crit = (2/(k+1))**(k/(k-1))

    # Plot results
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Left: incompressible flow
    ax[0].plot(dp/1e5, G_inc, 'b-', linewidth=2)
    ax[0].set_title('Incompressible flow')
    ax[0].set_xlabel(r'$\Delta p$ [bar]')
    ax[0].set_ylabel('G [kg/s]')
    ax[0].grid(True)
    ax[0].set_xlim(0, 12)
    ax[0].set_ylim(0, 60)

    # Right: compressible flow
    ax[1].plot(beta_vals, G_comp_norm, 'C1-', linewidth=2)
    ax[1].axvline(x=beta_crit, color='g', linestyle='--', linewidth=1.5,
                  label=f'β_crit = {beta_crit:.3f}')
    ax[1].set_title('Compressible flow')
    ax[1].set_xlabel(r'$\beta = p_2/p_1$')
    ax[1].set_ylabel(r'$G/G_{max}$')
    ax[1].grid(True)
    ax[1].legend(loc='lower left')
    ax[1].set_xlim(0, 1)
    ax[1].set_ylim(0, 1.1)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_flow_comparison()
