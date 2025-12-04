import numpy as np
import CoolProp.constants as CoolConst
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from typing import Iterable, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from ..tdyna.CoolStub import AbstractState  # type: ignore
else:
    from CoolProp.CoolProp import AbstractState

# Listing of some standard temperature and pressure values.
# Maybe move to other place when library is more mature
STP_IUPAC82 = {"T": 0+273.15, "p": 100_000}
STP_NIST = {"T": 0+273.15, "p": 101_325}
STP_SPE = {"T": 15+273.15, "p": 100_000}
STP_ISO13443 = {"T": 15+273.15, "p": 101_325}
STP_GOST2936 = {"T": 20+273.15, "p": 101_330}

# Making everythin nullable is a poor practice, so default values are imposed
# Parameters for 1 m^3 cube of air at standard temperature and pressure are used
# for this purposes.

# EOS
DEFAULT_EOS = AbstractState("HEOS", "Air")

# PVT parameters
DEFAULT_PRESSURE = np.array([STP_GOST2936["p"]])
DEFAULT_TEMPERATURE = np.array([STP_GOST2936["T"]])
DEFAULT_VOLUME = np.array([1])

# Updating default EOS to default parameters
DEFAULT_EOS.update(
    CoolConst.PT_INPUTS, DEFAULT_PRESSURE[0], DEFAULT_TEMPERATURE[0])

# Other default parameters
DEFAULT_DENSITY = np.array([DEFAULT_EOS.rhomass()])
DEFAULT_MASS = DEFAULT_VOLUME*DEFAULT_DENSITY
DEFAULT_SPECIFIC_ENERGY = np.array([DEFAULT_EOS.umass()])
DEFAULT_ENERGY = DEFAULT_MASS*DEFAULT_SPECIFIC_ENERGY
DEFAULT_H = np.array([1])

# Composition
DEFAULT_COMPOSITION = np.array([1])
DEFAULT_COMPOSITION_VAPOR = np.array([1])
DEFAULT_COMPOSITION_LIQUID = np.array([0])
DEFAULT_VAPOR_QUALITY = np.array([0])

# Transport properties
DEFAULT_VISOCISTY = np.array([DEFAULT_EOS.viscosity()])
DEFAULT_THERMAL_CONDUCTIVITY = np.array([DEFAULT_EOS.conductivity()])

# Default operational parametersfor conductors
DEFAULT_ELEVATION = 0
DEFAULT_VELOCITY = np.array([0])
DEFAULT_FLOWRATE = np.array([0])


@dataclass
class OperationData(ABC):
    # Maybe will be extended to store more

    # EOS interfaces for phases
    eos: list[AbstractState] = field(default_factory=lambda: [DEFAULT_EOS])

    # Thermodynamic parameters
    p: NDArray = field(default_factory=lambda: DEFAULT_PRESSURE)
    V: NDArray = field(default_factory=lambda: DEFAULT_VOLUME)
    T: NDArray = field(default_factory=lambda: DEFAULT_TEMPERATURE)
    rho: NDArray = field(default_factory=lambda: DEFAULT_DENSITY)
    u: NDArray = field(default_factory=lambda: DEFAULT_SPECIFIC_ENERGY)

    # For potential VLE
    x_all: NDArray = field(default_factory=lambda: DEFAULT_COMPOSITION)
    x_vap: NDArray = field(
        default_factory=lambda: DEFAULT_COMPOSITION_VAPOR)
    x_liq: NDArray = field(
        default_factory=lambda: DEFAULT_COMPOSITION_LIQUID)
    Q_vap: NDArray = field(
        default_factory=lambda: DEFAULT_VAPOR_QUALITY)

    # Fields for transport properties
    mu: NDArray = field(default_factory=lambda: DEFAULT_VISOCISTY)
    lam: NDArray = field(
        default_factory=lambda: DEFAULT_THERMAL_CONDUCTIVITY)


@dataclass
class StateData(OperationData):
    m: NDArray = field(default_factory=lambda: DEFAULT_MASS)
    E: NDArray = field(default_factory=lambda: DEFAULT_ENERGY)
    h: NDArray = field(default_factory=lambda: DEFAULT_H)


@dataclass
class FlowData(OperationData):
    # This will be used for stream description in Conductors,
    # so we need elevation, velocity and specific energy of stream.
    z: float = DEFAULT_ELEVATION
    w: NDArray = field(
        default_factory=lambda: DEFAULT_VELOCITY)
    j: NDArray = field(
        default_factory=lambda: DEFAULT_SPECIFIC_ENERGY)
    G: NDArray = field(
        default_factory=lambda: DEFAULT_FLOWRATE)
    Q: Optional[NDArray] = field(
        default_factory=lambda: DEFAULT_FLOWRATE*DEFAULT_DENSITY)
    J: Optional[NDArray] = field(
        default_factory=lambda: DEFAULT_FLOWRATE*DEFAULT_SPECIFIC_ENERGY)


# Testing creation
if __name__ == '__main__':
    opd = OperationData()
    opd_state = StateData()
    opd_flow = FlowData()
