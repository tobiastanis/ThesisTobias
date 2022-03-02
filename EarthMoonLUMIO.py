"""
This is a propagation of LUMIO, Earth and the Moon only. It is a building block to obtain a full propagation of the solar
system
"""

import Dataset_reader
import Simulation_setup
import numpy as np
from tudatpy.kernel import constants
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
import matplotlib.pyplot as plt
spice_interface.load_standard_kernels()
print("Running [EarthMoonLUMIO.py]")
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

# Environment #
bodies_to_create = ["Earth", "Moon"]
global_frame_origin = "Earth"
global_frame_orientation = "J2000"
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation
)

bodies = environment_setup.create_system_of_bodies(body_settings)

##### System of bodies #####
bodies.create_empty_body("LUMIO")
bodies.get("LUMIO").mass = 22.3
#bodies.get("Moon").mass = spice_interface.get_body_gravitational_parameter("Moon")/constants.GRAVITATIONAL_CONSTANT

bodies_to_propagate = ["Moon", "LUMIO"]
central_bodies = ["Earth", "Earth"]

acceleration_settings_Moon = dict(
    Earth=[propagation_setup.acceleration.spherical_harmonic_gravity(5,5)]
)
acceleration_settings_LUMIO = dict(
    Earth=[propagation_setup.acceleration.spherical_harmonic_gravity(5,5)],
    Moon=[propagation_setup.acceleration.spherical_harmonic_gravity(5,5)]
)
acceleration_settings = {
    "Moon": acceleration_settings_Moon,
    "LUMIO": acceleration_settings_LUMIO
}
acceleration_models = propagation_setup.create_acceleration_models(
    bodies, acceleration_settings, bodies_to_propagate, central_bodies
)

# Initial states
initial_states = np.transpose([np.concatenate((X_Moon_ini, X_LUMIO_ini), axis=0)])

# Savings
Moon_dependent_variables_to_save = [
    propagation_setup.dependent_variable.total_acceleration("Moon"),
    propagation_setup.dependent_variable.central_body_fixed_cartesian_position("Moon", "Earth")
]
LUMIO_dependent_variables_to_save = [
    propagation_setup.dependent_variable.total_acceleration("LUMIO"),
    propagation_setup.dependent_variable.central_body_fixed_cartesian_position("LUMIO", "Earth")
]
Saved_variables = Moon_dependent_variables_to_save + LUMIO_dependent_variables_to_save

# Propagating
termination_condition = propagation_setup.propagator.time_termination(simulation_end_epoch)
propagation_setup = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models,
    bodies_to_propagate,
    initial_states,
    termination_condition,
    output_variables=Saved_variables
)
# Integrating
integrator_settings = numerical_simulation.propagation_setup.integrator.runge_kutta_4(
    simulation_start_epoch, fixed_time_step
)
# Simulating
dynamic_simulator = numerical_simulation.SingleArcSimulator(
    bodies, integrator_settings, propagation_setup
)

states = dynamic_simulator.state_history

######### Rewritting Data #############
width = states[simulation_start_epoch].size            # This is the length of np array
height = len(states)                # This is the amount of key/value pairs fo dict

states_array = np.empty(shape=(height,width))        # Ini 2d matrix
# Loop over entries in dictionair getting both key and value
for x, (key, np_array) in enumerate(states.items()):
    # Looping over elements in the np array
    for y, np_value in enumerate(np_array):
        #print("i {}: key: {}, np.array {}".format(x, key, np_value))
        states_array[x, y] = np_value

states_Moon = states_array[:, 0:6]
states_LUMIO = states_array[:, 6:12]

if all(states_Moon[0, :]) == all(X_Moon_ini) and all(states_LUMIO[0, :]) == all(X_LUMIO_ini):
    print('jeej')
else:
    print('error')

plt.figure()
plt.plot(states_Moon[:,0], states_Moon[:,1])
plt.plot(states_LUMIO[:,0], states_LUMIO[:, 1])
plt.plot(X_Moon_Dataset[:, 0], X_LUMIO_Dataset[:, 1])
plt.plot(X_LUMIO_Dataset[:, 0], X_LUMIO_Dataset[:, 1])
plt.plot(0, 0, marker='o', markersize=10, color='blue')
plt.legend(['Moon', 'LUMIO', 'Moon_dataset', 'LUMIO dataset', 'Earth'])
plt.title('xy')
plt.figure()
plt.plot(states_Moon[:,0], states_Moon[:,2])
plt.plot(states_LUMIO[:,0], states_LUMIO[:, 2])
plt.plot(X_Moon_Dataset[:, 0], X_LUMIO_Dataset[:, 2])
plt.plot(X_LUMIO_Dataset[:, 0], X_LUMIO_Dataset[:, 2])
plt.plot(0, 0, marker='o', markersize=10, color='blue')
plt.legend(['Moon', 'LUMIO', 'Moon_dataset', 'LUMIO dataset', 'Earth'])
plt.title('xz')
plt.figure()
plt.plot(states_Moon[:,1], states_Moon[:,2])
plt.plot(states_LUMIO[:,1], states_LUMIO[:, 2])
plt.plot(X_Moon_Dataset[:, 1], X_LUMIO_Dataset[:, 2])
plt.plot(X_LUMIO_Dataset[:, 1], X_LUMIO_Dataset[:, 2])
plt.plot(0, 0, marker='o', markersize=10, color='blue')
plt.legend(['Moon', 'LUMIO', 'Moon_dataset', 'LUMIO dataset', 'Earth'])
plt.title('xy')
plt.show()

print("[EarthMoonLUMIO.py] ran successfully \n")