"""
Propagation of LUMIO using tudatpy libraries. Propagation starts at the same initial condition as the dataset, but over
it is expected that the propagated by tudat trajectory will deviate due to no stationkeeping and or dynamic errors not
taken into account.
Errors that are taken into account: Spherical harmonic gravity Earth and Moon, pointmass Sun, Mercury, Venus, Mars,
Jupiter, Saturn, Uranus and Neptune and also Solar Radiation Pressure is taken into account.

The results of the propagation are compared with the state elements of the Dataset.

Note that the numbers used for SRP, are an estimate and not factual yet...
"""
import Dataset_reader
import Simulation_setup
import numpy as np
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
import matplotlib.pyplot as plt
spice_interface.load_standard_kernels()
print("Running [Fullpropagation.py]")

# Adjust simulation setting in [Simulation_setup.py]
t0 = Simulation_setup.t0_mjd
tend = t0+Simulation_setup.simulation_time
fixed_time_step = Simulation_setup.fixed_time_step
simulation_start_epoch = Simulation_setup.simulation_start_epoch
simulation_end_epoch = Simulation_setup.simulation_end_epoch

### Dataset, initial state and Moon from ephemeris
# LUMIO state over simulation time provided by dataset
X_LUMIO_Dataset = Dataset_reader.state_lumio(t0, tend)
# Moon state over simulation time provided by dataset
X_Moon_Dataset = Dataset_reader.state_moon(t0, tend)
# Initial state to propagate
X_LUMIO_ini = X_LUMIO_Dataset[0, :]
# Initial Moon state
X_Moon_ini = spice_interface.get_body_cartesian_state_at_epoch("Moon", "Earth", "J2000", "NONE", simulation_start_epoch)

if X_LUMIO_ini.all() == X_LUMIO_Dataset.all():
    print('LUMIO: Initial state equals dataset')
else:
    print('ERROR: LUMIO initial states differs')
    quit()
if X_Moon_ini.all() == X_Moon_Dataset.all():
    print('Moon: Initial state equals dataset')
else:
    print('ERROR: Moon initial state differs')
    quit()

### Environment Setup ###
# The creation of bodies
bodies_to_create = [
    "Earth", "Moon", "Sun", "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"
]
global_frame_origin = "Earth"
global_frame_orientation = "J2000"
initial_time = simulation_start_epoch
final_time = simulation_end_epoch
body_settings = environment_setup.get_default_body_settings_time_limited(
    bodies_to_create, initial_time, final_time, global_frame_origin, global_frame_orientation, fixed_time_step
)

body_system = environment_setup.create_system_of_bodies(body_settings)

# Adding LUMIO to the fray
body_system.create_empty_body("LUMIO")
body_system.get("LUMIO").mass = 22.3

bodies_to_propagate = ["LUMIO"]

central_bodies = ["Earth"]

### Acceleration Setup ###
# SRP
reference_area_radiation = 1.0
radiation_pressure_coefficient = 1.0
occulting_bodies = ["Moon"]
radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
    "Sun", reference_area_radiation, radiation_pressure_coefficient, occulting_bodies
)
environment_setup.add_radiation_pressure_interface(body_system,"LUMIO", radiation_pressure_settings)

acceleration_settings_LUMIO = dict(
    Earth=[propagation_setup.acceleration.spherical_harmonic_gravity(5,5)],
    Moon=[propagation_setup.acceleration.spherical_harmonic_gravity(5,5)],
    Sun=[propagation_setup.acceleration.point_mass_gravity(),
         propagation_setup.acceleration.cannonball_radiation_pressure()],
    Mercury=[propagation_setup.acceleration.point_mass_gravity()],
    Venus=[propagation_setup.acceleration.point_mass_gravity()],
    Mars=[propagation_setup.acceleration.point_mass_gravity()],
    Jupiter=[propagation_setup.acceleration.point_mass_gravity()],
    Saturn=[propagation_setup.acceleration.point_mass_gravity()],
    Uranus=[propagation_setup.acceleration.point_mass_gravity()],
    Neptune=[propagation_setup.acceleration.point_mass_gravity()]
)

acceleration_settings = {
    "LUMIO": acceleration_settings_LUMIO
}

acceleration_models = propagation_setup.create_acceleration_models(
    body_system, acceleration_settings, bodies_to_propagate, central_bodies)

initial_states = np.transpose([X_LUMIO_ini])

### Savings ###
# Is adjustable
dependent_variables_to_save = [
    propagation_setup.dependent_variable.total_acceleration("LUMIO"),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.cannonball_radiation_pressure_type, "LUMIO", "Sun"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.spherical_harmonic_gravity_type, "LUMIO", "Earth"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.spherical_harmonic_gravity_type, "LUMIO", "Moon"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "LUMIO", "Sun"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "LUMIO", "Mercury"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "LUMIO", "Venus"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "LUMIO", "Mars"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "LUMIO", "Jupiter"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "LUMIO", "Saturn"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "LUMIO", "Uranus"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "LUMIO", "Neptune"
    )
]

### Propagating ###
termination_condition = propagation_setup.propagator.time_termination(simulation_end_epoch)
propagation_settings = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models,
    bodies_to_propagate,
    initial_states,
    termination_condition,
    output_variables=dependent_variables_to_save
)
### Integrating ###
integrator_settings = numerical_simulation.propagation_setup.integrator.runge_kutta_4(
    simulation_start_epoch, fixed_time_step
)

### Dynamic Simulator ###
dynamic_simulator = numerical_simulation.SingleArcSimulator(
    body_system, integrator_settings, propagation_settings
)
### RESULTS ###
time = Simulation_setup.simulation_span

output_dict = dynamic_simulator.dependent_variable_history
states_dict = dynamic_simulator.state_history
output = np.vstack(list(output_dict.values()))
states = np.vstack(list(states_dict.values()))

delta_LUMIO_state = np.subtract(X_LUMIO_Dataset, states)

fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, constrained_layout=True, sharey=False)
ax1.plot(0, 0, marker='o', markersize=10, color='blue')
ax1.plot(states[:, 0], states[:, 1], color='orange')
ax1.plot(X_Moon_Dataset[:, 0], X_Moon_Dataset[:, 1], color='red')
ax1.plot(X_LUMIO_Dataset[:, 0], X_LUMIO_Dataset[:, 1], color='green')
ax1.plot(states[0, 0], states[0, 1], marker='o', markersize=4, color='orange')
ax1.plot(X_Moon_Dataset[0, 0], X_Moon_Dataset[0, 1], marker='o', markersize=4, color='red')
ax1.plot(X_LUMIO_Dataset[0, 0], X_LUMIO_Dataset[0, 1], marker='o', markersize=4, color='green')
ax1.set_title('Trajectory in xy-plane')
ax1.set_xlabel('x-direction [m]')
ax1.set_ylabel('y-direction [m]')
ax2.plot(0, 0, marker='o', markersize=10, color='blue')
ax2.plot(states[:, 0], states[:, 2], color='orange')
ax2.plot(X_Moon_Dataset[:, 0], X_Moon_Dataset[:, 2], color='red')
ax2.plot(X_LUMIO_Dataset[:, 0], X_LUMIO_Dataset[:, 2], color='green')
ax2.plot(states[0, 0], states[0, 2], marker='o', markersize=4, color='orange')
ax2.plot(X_Moon_Dataset[0, 0], X_Moon_Dataset[0, 2], marker='o', markersize=4, color='red')
ax2.plot(X_LUMIO_Dataset[0, 0], X_LUMIO_Dataset[0, 2], marker='o', markersize=4, color='green')
ax2.set_title('Trajectory in xz-plane')
ax2.set_xlabel('x-direction [m]')
ax2.set_ylabel('z-direction [m]')
ax3.plot(0, 0, marker='o', markersize=10, color='blue')
ax3.plot(states[:, 1], states[:, 2], color='orange')
ax3.plot(X_Moon_Dataset[:, 1], X_Moon_Dataset[:, 2], color='red')
ax3.plot(X_LUMIO_Dataset[:, 1], X_LUMIO_Dataset[:, 2], color='green')
ax3.plot(states[0, 1], states[0, 2], marker='o', markersize=4, color='orange')
ax3.plot(X_Moon_Dataset[0, 1], X_Moon_Dataset[0, 2], marker='o', markersize=4, color='red')
ax3.plot(X_LUMIO_Dataset[0, 1], X_LUMIO_Dataset[0, 2], marker='o', markersize=4, color='green')
ax3.set_title('Trajectory in yz-plane')
ax3.set_xlabel('y-direction [m]')
ax3.set_ylabel('z-direction [m]')
ax1.legend(['Earth (center)', 'Propagated LUMIO state', 'Moon from dataset', 'LUMIO from dataset'])

fig2, (bx1, bx2, bx3) = plt.subplots(3, 1, constrained_layout=True, sharey=False)
bx1.plot(time, states[:, 0], color='orange')
bx1.plot(time, X_LUMIO_Dataset[:, 0], color='green')
bx1.set_title('Trajectory LUMIO in x-direction')
bx1.set_xlabel('Time [days]')
bx1.set_ylabel('x-direction [m]')
bx2.plot(time, states[:, 1], color='orange')
bx2.plot(time, X_LUMIO_Dataset[:, 1], color='green')
bx2.set_title('Trajectory LUMIO in y-direction')
bx2.set_xlabel('Time [days]')
bx2.set_ylabel('y-direction [m]')
bx3.plot(time, states[:, 2], color='orange')
bx3.plot(time, X_LUMIO_Dataset[:, 2], color='green')
bx3.set_title('Trajectory LUMIO in z-direction')
bx3.set_xlabel('Time [days]')
bx3.set_ylabel('z-direction [m]')
bx1.legend(['Propagated state element', 'Dataset state element'])

fig3, (cx1, cx2, cx3) = plt.subplots(3, 1, constrained_layout=True, sharey=False)
cx1.plot(time, states[:, 3], color='orange')
cx1.plot(time, X_LUMIO_Dataset[:, 3], color='green')
cx1.set_title('Velocity LUMIO in x-direction')
cx1.set_xlabel('Time [days]')
cx1.set_ylabel('Velocity [m/s]')
cx2.plot(time, states[:, 4], color='orange')
cx2.plot(time, X_LUMIO_Dataset[:, 4], color='green')
cx2.set_title('Velocity LUMIO in y-direction')
cx2.set_xlabel('Time [days]')
cx2.set_ylabel('Velocity [m/s]')
cx3.plot(time, states[:, 5], color='orange')
cx3.plot(time, X_LUMIO_Dataset[:, 5], color='green')
cx3.set_title('Velocity LUMIO in x-direction')
cx3.set_xlabel('Time [days]')
cx3.set_ylabel('Velocity [m/s]')
cx1.legend(['Propagated state element', 'Dataset state element'])


fig4, (dx1, dx2) = plt.subplots(2, 1, constrained_layout=True, sharey=False)
dx1.plot(time, delta_LUMIO_state[:, 0])
dx1.plot(time, delta_LUMIO_state[:, 1])
dx1.plot(time, delta_LUMIO_state[:, 2])
dx1.set_title('Difference in position (Dataset minus Propagated)')
dx1.set_xlabel('Time [days]')
dx1.set_ylabel('Distance [m]')
dx1.legend(['x-direction', 'y-direction', 'z-direction'])
dx2.plot(time, delta_LUMIO_state[:, 3])
dx2.plot(time, delta_LUMIO_state[:, 4])
dx2.plot(time, delta_LUMIO_state[:, 5])
dx2.set_title('Difference in velocity (Dataset minus Propagated)')
dx2.set_xlabel('Time [days]')
dx2.set_ylabel('Velocity [m/s]')
dx2.legend(['x-direction', 'y-direction', 'z-direction'])

plt.figure()
plt.plot(time, np.linalg.norm(output[:, 0:3], axis=1))
plt.plot(time, output[:, 3])
plt.plot(time, output[:, 4])
plt.plot(time, output[:, 5])
plt.plot(time, output[:, 6])
plt.plot(time, output[:, 7])
plt.plot(time, output[:, 8])
plt.plot(time, output[:, 9])
plt.plot(time, output[:, 10])
plt.plot(time, output[:, 11])
plt.plot(time, output[:, 12])
plt.plot(time, output[:, 13])
plt.legend(['Total acceleration', 'Solar Radiation Pressure', 'Spherical harmonic gravity by Earth',
            'Spherical harmonic gravity by Moon', 'Point mass gravity by Sun', 'Point mass gravity by Mercury',
            'Point mass gravity by Venus', 'Point mass gravity by Mars', 'Point mass gravity by Jupiter',
            'Point mass gravity by Saturn', 'Point mass gravity by Uranus', 'Point mass gravity by Neptune'])
plt.title('Acceleration distribution')
plt.xlabel('Time [days]')
plt.ylabel('Acceleration [m/$s^{2}$]')

plt.figure()
plt.plot(time, output[:, 6])
plt.plot(time, output[:, 7])
plt.plot(time, output[:, 8])
plt.plot(time, output[:, 9])
plt.plot(time, output[:, 10])
plt.plot(time, output[:, 11])
plt.plot(time, output[:, 12])
plt.plot(time, output[:, 13])
plt.legend(['Point mass gravity by Sun', 'Point mass gravity by Mercury',
            'Point mass gravity by Venus', 'Point mass gravity by Mars', 'Point mass gravity by Jupiter',
            'Point mass gravity by Saturn', 'Point mass gravity by Uranus', 'Point mass gravity by Neptune'])
plt.title('Detailed acceleration distribution of only the point masses')
plt.xlabel('Time [days]')
plt.ylabel('Acceleration [m/$s^{2}$]')

plt.figure()
plot = plt.axes(projection='3d')
plot.plot(0, 0, 0, marker='o', markersize=10, color='blue')
plot.plot(states[:, 0], states[:, 1], states[:, 2], color='orange')
plot.plot(X_Moon_Dataset[:, 0], X_Moon_Dataset[:, 1], X_Moon_Dataset[:, 2], color='red')
plot.plot(X_LUMIO_Dataset[:, 0], X_LUMIO_Dataset[:, 1], X_LUMIO_Dataset[:, 2], color='green')
plot.plot(states[0, 0], states[0, 1], states[0, 2], marker='o', markersize=4, color='orange')
plot.plot(X_Moon_Dataset[0, 0], X_Moon_Dataset[0, 1], X_Moon_Dataset[0, 2], marker='o', markersize=4, color='red')
plot.plot(X_LUMIO_Dataset[0, 0], X_LUMIO_Dataset[0, 1], X_LUMIO_Dataset[0, 2], marker='o', markersize=4, color='green')
plt.title('Earth-Moon-LUMIO System overview')
plt.legend(['Earth', 'LUMIO (propagated)', 'Moon (dataset)', 'LUMIO (dataset)'])
plot.set_xlabel('x-direction [m]')
plot.set_ylabel('y-direction [m]')
plot.set_zlabel('z-direction [m]')

plt.show()


print("[FullSolarSystemLUMIOpropagation.py] ran successfully")


