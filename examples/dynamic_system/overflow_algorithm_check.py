import numpy as np
import matplotlib.pyplot as plt

from numpy.typing import NDArray
from pytroleum.sdyna import network as net
from pytroleum.sdyna import convolumes as cvs
from pytroleum.sdyna import conductors as cds
from pytroleum.sdyna import opdata as opd
from pytroleum.tdyna import eos

from pprint import pprint


class NetworkWithOverflow(net.DynamicNetwork):

    def __init__(self) -> None:
        super().__init__()

        s1 = cvs.SectionHorizontal(800e-3, 1500e-3, 0, 2, lambda h: 0)
        s2 = cvs.SectionHorizontal(0, 4200e-3, 800e-3, 2, lambda h: 0)

        inlet = cds.Fixed([0, 1], None, s1)
        vlv = cds.Valve(1, 32e-3, 25e-3, 0.61, 0.0, 0, s1)
        opass = cds.OverPass(1600e-3, 1e-3, s1, s2)  # type: ignore

        s1.state = opd.factory_state(
            [eos.factory_eos({"air": 1}),
             eos.factory_eos({"water": 1})],
            s1.compute_volume_with_level,
            np.array([2e5+10e3, 2e5]),
            np.array([300., 300.]),
            np.array([2, 1560e-3]))
        s1.advance()

        s2.state = opd.factory_state(
            [eos.factory_eos({"air": 1}),
             eos.factory_eos({"water": 1})],
            s2.compute_volume_with_level,
            np.array([2e5, 2e5]),
            np.array([300., 300.]),
            np.array([2, 0.2]))
        s2.advance()

        inlet.flow = opd.factory_flow(
            [eos.factory_eos({"air": 1}),
             eos.factory_eos({"water": 1})],
            s1.state.pressure.copy(),
            s1.state.temperature.copy(),
            np.pi*(50e-3)**2/4, 1.5, np.array([0.0, 5]))

        vlv.flow = opd.factory_flow(
            [eos.factory_eos({"air": 1}),
             eos.factory_eos({"water": 1})],
            s1.state.pressure.copy(),
            s1.state.temperature.copy(),
            vlv.area_pipe, vlv.elevation,
            np.array([0., 0.]))

        opass.flow = opd.factory_flow(
            [eos.factory_eos({"air": 1}),
             eos.factory_eos({"water": 1})],
            s1.state.pressure.copy(),
            s1.state.temperature.copy(),
            -1, 0, np.array([0., 0.]))

        self.add_control_volume(s1)
        self.add_control_volume(s2)
        self.add_conductor(inlet)
        self.add_conductor(vlv)
        self.add_conductor(opass)

        self.s1 = s1
        self.s2 = s2

        self.inlet = inlet
        self.vlv = vlv
        self.opass = opass

    def ode_system(self, t, y):
        return super().ode_system(t, y)

    def advance(self):
        return super().advance()


network = NetworkWithOverflow()
network.evaluate_size()
network.prepare_solver(0.5)

time = []
param = []

N = 200
for _ in range(N):
    time.append(network.solver.t)
    param.append(network.s1.state.level[1])
    network.advance()
param = np.array(param)

pprint(network.vlv.flow)

plt.plot(time, param*1e3)
# plt.plot(time, param2*1e3)
plt.show()
