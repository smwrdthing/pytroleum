import numpy as np
from scipy.optimize import newton, brentq, fsolve

from pytroleum.sdyna import convolumes as cvs
from pytroleum.sdyna import conductors as cds
from pytroleum.sdyna import network as net
from pytroleum.sdyna.opdata import factory_state, factory_flow
from pytroleum.tdyna.eos import factory_eos

from pprint import pprint

cvs._NUMBER_OF_GRADUATION_POINTS = 700


class NetWithDistributor(net.DynamicNetwork):

    def __init__(self) -> None:
        super().__init__()

        s1 = cvs.SectionHorizontal(800e-3, 1000e-3, 0, 1000e-3, lambda h: 0)
        s2 = cvs.SectionHorizontal(0, 1000e-3, 800e-3, 1000e-3, lambda h: 0)

        upass = cds.UnderPass(400e-3, 0.61, 0)  # type: ignore

        upass.connect_source(s1)
        upass.connect_sink(s2)

        s1.state = factory_state(
            [factory_eos({"air": 1}), factory_eos({"water": 1})],
            s1.compute_volume_with_level,
            np.array([1e5, 1e5]),
            np.array([300., 300.]),
            np.array([1000e-3, 200e-3]))
        # s1.advance()

        s2.state = factory_state(
            [factory_eos({"air": 1}), factory_eos({"water": 1})],
            s2.compute_volume_with_level,
            np.array([1e5, 1e5]),
            np.array([300., 300.]),
            np.array([1000e-3, 200e-3]))
        # s2.advance()

        upass.flow = factory_flow(
            [factory_eos({"air": 1}), factory_eos({"water": 1})],
            s1.state.pressure.copy(),
            s1.state.temperature.copy(),
            -1, 0, np.array([0, 0]))

        self.add_control_volume(s1)
        self.add_control_volume(s2)
        self.add_conductor(upass)

        self.s1 = s1
        self.s2 = s2
        self.upass = upass

    def ode_system(self, t, y):
        return super().ode_system(t, y)

    def advance(self):
        self.upass.advance()
        self.s1.advance()
        self.s2.advance()


network = NetWithDistributor()

V_liquid_init = np.sum(network.s1.state.mass[1:] /
                       network.s1.state.density[1:])+np.sum(network.s2.state.mass[1:] /
                                                            network.s2.state.density[1:])

network.advance()

V_liquid = np.sum(network.s1.state.mass[1:] /
                  network.s1.state.density[1:]) + np.sum(network.s2.state.mass[1:] /
                                                         network.s2.state.density[1:])

dV = V_liquid_init-V_liquid
relativedV = dV/(network.s1.volume+network.s2.volume)

pprint(network.s1.state)
print()
pprint(network.s2.state)
print()

dp = network.s1.state.pressure[-1]-network.s2.state.pressure[-1]
p_min = np.min([network.s1.state.pressure[-1],
                network.s2.state.pressure[-1]])

dh = network.s1.state.level[1]-network.s2.state.level[1]

print(f"Liquid volume before distribution : {V_liquid_init}")
print(f"Liquid volume after distribution : {V_liquid}")
print()
print(f"Volume  difference {dV:.3f} m^3")
print(f"Volume  difference (rel.) {relativedV*100:.3f} %")
print()
print(f"Pressure difference {dp:.3f} Pa")
print(f"Pressure difference (rel. to min) {dp/p_min*100:.3f} %")
print()
print(f"Level difference {dh*1e3:.3f} mm")
