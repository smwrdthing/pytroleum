import numpy as np
from scipy.integrate import OdeSolver
from warnings import warn


class ExplicitEuler(OdeSolver):

    def __init__(self, fun, t0, y0, time_step: float, t_bound=np.inf, vectorized=False,
                 support_complex=False, **extraneous) -> None:

        if extraneous:
            warn("Following arguments have no effect for chosen solver: {}.".format(
                ", ".join(f"`{x}`" for x in extraneous)))

        self.time_step = time_step
        super().__init__(fun, t0, y0, t_bound, vectorized, support_complex)

    def _step_impl(self):
        self.y = self.y+self.fun(self.t, self.y)*self.time_step
        self.t = self.t+self.time_step
        return True, None

    def _dense_output_impl(self):
        return super()._dense_output_impl()
