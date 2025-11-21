# Initial outline for control volumes

import abc  # figure out how to use
import numpy as np


class AbstractCV:

    # Should be abstract base class for control volume

    def __init__(self) -> None:
        # Matter state manager here?
        self.outlets = []
        self.inlets = []

    @np.vectorize  # want to try this vectorization stuff as decorators
    def connect_as_outlet(self, conductor):
        pass

    @np.vectorize
    def connet_as_inlet(self, conductor):
        pass

    def advance(self):
        pass


class Atmosphere(AbstractCV):

    # Class for atmosphere representation. Should imposs nominal infinite
    # volume and constant values for thermodynamic paramters

    def __init__(self) -> None:
        super().__init__()


class Reservoir(AbstractCV):

    # Class to represent petroleum fluids reservoir. In context of dynamical system
    # modelling imposes infinite volume and csontant params (for now?)

    def __init__(self) -> None:
        super().__init__()


class SectionH(AbstractCV):

    # Class for horizontal section

    def __init__(self) -> None:
        super().__init__()


class SectionV(AbstractCV):

    # Class for vertical section, not really needed right now, might be useful later for
    # tests and other equipment?

    def __init__(self) -> None:
        super().__init__()
