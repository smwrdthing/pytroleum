from pytroleum.sdyna.conductors import CentrifugalPump
import numpy as np
import matplotlib.pyplot as plt

pump = CentrifugalPump(of_phase=1)

RPM = 1500
pump.angular_velocity = np.pi*RPM/30

volume_flow_rates = (0, 10, 20)
heads = (35, 40, 0)
pump.characteristic_reference(pump.angular_velocity, volume_flow_rates, heads)
k1, k2, k3 = pump.coefficients


def head(speed, flow):
    return k1*speed**2 - 2*k2*speed*flow - k3*flow**2


Q = np.linspace(0, 22)
H = head(pump.angular_velocity, Q)

fig, ax = plt.subplots()
ax.plot(Q, H)
ax.plot(volume_flow_rates, heads, 'ro')
ax.grid(True)
ax.set_xlim((0, 21))
ax.set_ylim((0, 45))
plt.show()
