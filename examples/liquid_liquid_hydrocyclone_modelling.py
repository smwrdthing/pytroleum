import numpy as np
import matplotlib.pyplot as plt

from pytroleum.plant.liquid_cyclone import cyclone as llh
from pytroleum.plant.liquid_cyclone import separation as sep
from pytroleum.plant.liquid_cyclone import utils


def compute_resistance(density, area, discharge_coeff):
    # auxiliary function
    return discharge_coeff*area/np.sqrt(density)


# Instantiate design, flowsheet and velocity field
design = llh.SouthamptonDesign(80e-3)  # 80 mm Colman-Thew hydrocyclone
flowsheet = llh.FlowSheet()
velocity_field = llh.VelocityField()

# NOTE :
# for flowsheet here default equations of state are utilized, see implementation or docs

# Set resistance coefficients

# Overflow
flowsheet.resistance[llh.ResistanceSpec.O] = compute_resistance(
    flowsheet.light_phase_eos.rhomass(),
    np.pi*design.diameters[llh.SouthamptonDiameters.O]**2/4, 0.11)

# Underflow
flowsheet.resistance[llh.ResistanceSpec.U] = compute_resistance(
    flowsheet.heavy_phase_eos.rhomass(),
    np.pi*design.diameters[llh.SouthamptonDiameters.U]**2/4, 0.12)

# Inlet
light_phase_fraction = 0.1  # fractin of contaminant in inflow for density
flowsheet.resistance[llh.ResistanceSpec.IN] = compute_resistance(
    flowsheet.heavy_phase_eos.rhomass()*(1-light_phase_fraction) +
    flowsheet.light_phase_eos.rhomass()*light_phase_fraction,
    np.pi*design.diameters[llh.SouthamptonDiameters.I]**2/4, 0.32)

# Underflow valve
underflow_valve_diameter = 12e-3
underflow_valve_opening = 1.0
flowsheet.resistance[llh.ResistanceSpec.UV] = compute_resistance(
    flowsheet.heavy_phase_eos.rhomass(),
    np.pi*underflow_valve_diameter**2/4*underflow_valve_opening, 0.61)

# overflow valve
overflow_valve_diameter = 6e-3
overflow_valve_opening = 1.0
flowsheet.resistance[llh.ResistanceSpec.OV] = compute_resistance(
    flowsheet.light_phase_eos.rhomass(),
    np.pi*overflow_valve_diameter**2/4*overflow_valve_opening, 0.61)

# Set inlet flow rate and backpressures OR outlet flow rates and inlet pressure
# (here first set of inputs is utilized)
Q_in = 0.5e-3
P_ob = 1.5e5  # overflow backpressure
P_ub = 4.5e5  # underflow backpressure
backpressures = (P_ob, P_ub)  # order matters

# Solve flowsheet
flowsheet.solve_from_backpressures(Q_in, backpressures)
flowsheet.account_for_recirculation()  # default recirculation rate is used

# Solve velocity field
velocity_field.solve_ndim_profile_coeffs(flowsheet)

# Now almost everything is ready for LLH modelling, collect model components into setup
# tuple for convenience
setup = flowsheet, design, velocity_field

# Plotting velocity fields
fig, ax = utils.plot_velocity_field(setup, "radial")
ax.set_title("Radial component")

fig, ax = utils.plot_velocity_field(setup, "tangent")
ax.set_title("Tangent component")

fig, ax = utils.plot_velocity_field(setup, "axial")
ax.set_title("Axial component")

# Evaluate separation efficiency metrics
d_efficiency, G_efficiency = sep.build_grade_efficiency_curve(
    setup, 30)  # type: ignore
d50 = sep.extract_d50(d_efficiency, G_efficiency)

diameters = np.linspace(0, d_efficiency[-1], 10)  # type: ignore
fig, ax = utils.plot_model_region(design, velocity_field, half=True)
ax.set_title("Droplet trajectories")
for i, d in enumerate(diameters):
    r, z = sep.drop_trajectroy(d, setup)  # type: ignore
    linewidth = 3
    if i == 0:
        color = "b"
    elif i == len(diameters)-1:
        color = "r"
    else:
        linewidth = 1
        color = None
    ax.plot(z*1e3, r*1e3, color=color, linewidth=linewidth)

# Evaluate total efficiency
percentiles = (15e-6, 20e-6, 50e-6)  # specifying size distribution
efficiency = sep.evaluate_total_efficiency(setup, percentiles)  # type: ignore

# Printy design and flowsheet summary, d50 and efficiency
design.summary()
flowsheet.summary()
print(f'd50 : {d50*1e6:.2f} micrometers')
print(f"efficiency : {efficiency*100:.2f} %")

# Now let's do same thing but for range of flow rates
varibale_inflow_range = np.arange(0.1, 2.0+0.1, 0.05)*1e-3
variable_efficiency = []
for Q in varibale_inflow_range:
    flowsheet.solve_from_backpressures(Q, backpressures)  # type: ignore
    velocity_field.solve_ndim_profile_coeffs(flowsheet)
    variable_efficiency.append(
        sep.evaluate_total_efficiency(setup, percentiles))  # type: ignore
variable_efficiency = np.array(variable_efficiency)

# Plotting ahead

# Grade efficiency plotting is not automated yet, but there are no problems with making it
# manually
fig, ax = plt.subplots()
ax.set_title("LLH grade efficiency")
ax.set_xlabel(r"d [$\mu m$]")
ax.set_ylabel("G [%]")
ax.set_title("LLH grade efficiency")
ax.plot(d_efficiency*1e6, G_efficiency*100, '-o')
ax.plot(d50*1e6, 50, 'go')
ax.set_xlim((0, d_efficiency[-1]*1e6))  # type: ignore
ax.set_ylim((0, 100))
ax.vlines(d50*1e6, 0, 100, 'g', '--')
ax.hlines(50, 0, d_efficiency[-1]*1e6, 'g', '--')
ax.grid(True)

# Plot employed distribution functions
d_distribution = np.linspace(0, percentiles[-1]*1.1, 500)
cdf = sep.cumulative_distribution_func(d_distribution, percentiles)
pdf = sep.probability_density_func(d_distribution, percentiles)

cdf_ax = 0
pdf_ax = 1

fig, ax = plt.subplots(figsize=(14.5, 5), ncols=2)
fig.suptitle("Drop size statistics")
ax[cdf_ax].set_xlabel(r"d [$\mu m$]")
ax[cdf_ax].set_ylabel("Cumulative Distribution")
ax[cdf_ax].plot(d_distribution*1e6, cdf)
ax[cdf_ax].set_xlim((0, np.max(d_distribution)*1e6))
ax[cdf_ax].set_ylim((-0.1, 1.1))
ax[cdf_ax].grid(True)

ax[pdf_ax].set_xlabel(r"d [$\mu m$]")
ax[pdf_ax].set_ylabel("Probability Density")
ax[pdf_ax].plot(d_distribution*1e6, pdf)
ax[pdf_ax].set_xlim((0, np.max(d_distribution)*1e6))
ax[pdf_ax].grid(True)

# Plot flow rate - efficiency characteritsic
fig, ax = plt.subplots()
ax.set_title("Separation efficiency")
ax.set_xlabel("Inlet flow rate [l/min]")
ax.set_ylabel("Separation efficiency [%]")
ax.plot(varibale_inflow_range*1e3*60, variable_efficiency*100, '-')
ax.grid(True)
ax.set_xlim((0.0, varibale_inflow_range[-1]*1e3*60))
ax.set_ylim((0, 110))
ax.hlines(90, 0, varibale_inflow_range[-1]
          * 1e3*60, colors='r', linestyles='--')

plt.show()
