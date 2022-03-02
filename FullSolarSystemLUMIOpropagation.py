"""
Propagation of LUMIO using tudatpy libraries. Propagation starts at the same initial condition as the dataset, but over
it is expected that the propagated by tudat trajectory will deviate due to no stationkeeping and or dynamic errors not
taken into account.
Errors that are taken into account: Spherical harmonic gravity Earth and Moon, pointmass Sun, Venus, Mars, Jupiter and
Solar Radiation Pressure

Note that the numbers used for SRP, are an estimate and not factual yet...
"""
import Dataset_reader
import Simulation_setup
import numpy as np
from tudatpy.kernel import constants
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import propagation
import matplotlib.pyplot as plt
spice_interface.load_standard_kernels()
print("Running [FullSolarSystemLUMIOpropagation.py]")

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
body_settings = environment_setup.get_default_body_settings(bodies_to_create)
body_system = environment_setup.create_system_of_bodies(body_settings)

# Adding LUMIO to the fray
body_system.create_empty_body("LUMIO")
body_system.get("LUMIO").mass = 22.3

bodies_to_propagate = ["LUMIO"] + bodies_to_create

central_bodies = []
for body_name in bodies_to_propagate:
    if body_name == "Moon":
        central_bodies.append("Earth")
    elif body_name == "LUMIO":
        central_bodies.append("Earth")
    elif body_name == "Sun":
        central_bodies.append("SSB")
    else:
        central_bodies.append("Sun")

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
acceleration_settings_Earth = dict(
    Moon=[propagation_setup.acceleration.spherical_harmonic_gravity(5,5)],
    Sun=[propagation_setup.acceleration.point_mass_gravity()],
    Mercury=[propagation_setup.acceleration.point_mass_gravity()],
    Venus=[propagation_setup.acceleration.point_mass_gravity()],
    Mars=[propagation_setup.acceleration.point_mass_gravity()],
    Jupiter=[propagation_setup.acceleration.point_mass_gravity()],
    Saturn=[propagation_setup.acceleration.point_mass_gravity()],
    Uranus=[propagation_setup.acceleration.point_mass_gravity()],
    Neptune=[propagation_setup.acceleration.point_mass_gravity()]
)
acceleration_settings_Moon = dict(
    Earth=[propagation_setup.acceleration.spherical_harmonic_gravity(5,5)],
    Sun=[propagation_setup.acceleration.point_mass_gravity()],
    Mercury=[propagation_setup.acceleration.point_mass_gravity()],
    Venus=[propagation_setup.acceleration.point_mass_gravity()],
    Mars=[propagation_setup.acceleration.point_mass_gravity()],
    Jupiter=[propagation_setup.acceleration.point_mass_gravity()],
    Saturn=[propagation_setup.acceleration.point_mass_gravity()],
    Uranus=[propagation_setup.acceleration.point_mass_gravity()],
    Neptune=[propagation_setup.acceleration.point_mass_gravity()]
)

acceleration_settings_Sun = dict(
    Earth=[propagation_setup.acceleration.point_mass_gravity()],
    Moon=[propagation_setup.acceleration.point_mass_gravity()],
    Mercury=[propagation_setup.acceleration.point_mass_gravity()],
    Venus=[propagation_setup.acceleration.point_mass_gravity()],
    Mars=[propagation_setup.acceleration.point_mass_gravity()],
    Jupiter=[propagation_setup.acceleration.point_mass_gravity()],
    Saturn=[propagation_setup.acceleration.point_mass_gravity()],
    Uranus=[propagation_setup.acceleration.point_mass_gravity()],
    Neptune=[propagation_setup.acceleration.point_mass_gravity()]
)

acceleration_settings_Mercury = dict(
    Earth=[propagation_setup.acceleration.point_mass_gravity()],
    Moon=[propagation_setup.acceleration.point_mass_gravity()],
    Sun=[propagation_setup.acceleration.point_mass_gravity()],
    Venus=[propagation_setup.acceleration.point_mass_gravity()],
    Mars=[propagation_setup.acceleration.point_mass_gravity()],
    Jupiter=[propagation_setup.acceleration.point_mass_gravity()],
    Saturn=[propagation_setup.acceleration.point_mass_gravity()],
    Uranus=[propagation_setup.acceleration.point_mass_gravity()],
    Neptune=[propagation_setup.acceleration.point_mass_gravity()]
)
acceleration_settings_Venus = dict(
    Earth=[propagation_setup.acceleration.point_mass_gravity()],
    Moon=[propagation_setup.acceleration.point_mass_gravity()],
    Sun=[propagation_setup.acceleration.point_mass_gravity()],
    Mercury=[propagation_setup.acceleration.point_mass_gravity()],
    Mars=[propagation_setup.acceleration.point_mass_gravity()],
    Jupiter=[propagation_setup.acceleration.point_mass_gravity()],
    Saturn=[propagation_setup.acceleration.point_mass_gravity()],
    Uranus=[propagation_setup.acceleration.point_mass_gravity()],
    Neptune=[propagation_setup.acceleration.point_mass_gravity()]
)
acceleration_settings_Mars = dict(
    Earth=[propagation_setup.acceleration.point_mass_gravity()],
    Moon=[propagation_setup.acceleration.point_mass_gravity()],
    Sun=[propagation_setup.acceleration.point_mass_gravity()],
    Mercury=[propagation_setup.acceleration.point_mass_gravity()],
    Venus=[propagation_setup.acceleration.point_mass_gravity()],
    Jupiter=[propagation_setup.acceleration.point_mass_gravity()],
    Saturn=[propagation_setup.acceleration.point_mass_gravity()],
    Uranus=[propagation_setup.acceleration.point_mass_gravity()],
    Neptune=[propagation_setup.acceleration.point_mass_gravity()]
)
acceleration_settings_Jupiter = dict(
    Earth=[propagation_setup.acceleration.point_mass_gravity()],
    Moon=[propagation_setup.acceleration.point_mass_gravity()],
    Sun=[propagation_setup.acceleration.point_mass_gravity()],
    Mercury=[propagation_setup.acceleration.point_mass_gravity()],
    Venus=[propagation_setup.acceleration.point_mass_gravity()],
    Mars=[propagation_setup.acceleration.point_mass_gravity()],
    Saturn=[propagation_setup.acceleration.point_mass_gravity()],
    Uranus=[propagation_setup.acceleration.point_mass_gravity()],
    Neptune=[propagation_setup.acceleration.point_mass_gravity()]
)
acceleration_settings_Saturn = dict(
    Earth=[propagation_setup.acceleration.point_mass_gravity()],
    Moon=[propagation_setup.acceleration.point_mass_gravity()],
    Sun=[propagation_setup.acceleration.point_mass_gravity()],
    Mercury=[propagation_setup.acceleration.point_mass_gravity()],
    Venus=[propagation_setup.acceleration.point_mass_gravity()],
    Mars=[propagation_setup.acceleration.point_mass_gravity()],
    Jupiter=[propagation_setup.acceleration.point_mass_gravity()],
    Uranus=[propagation_setup.acceleration.point_mass_gravity()],
    Neptune=[propagation_setup.acceleration.point_mass_gravity()]
)
acceleration_settings_Uranus = dict(
    Earth=[propagation_setup.acceleration.point_mass_gravity()],
    Moon=[propagation_setup.acceleration.point_mass_gravity()],
    Sun=[propagation_setup.acceleration.point_mass_gravity()],
    Mercury=[propagation_setup.acceleration.point_mass_gravity()],
    Venus=[propagation_setup.acceleration.point_mass_gravity()],
    Mars=[propagation_setup.acceleration.point_mass_gravity()],
    Jupiter=[propagation_setup.acceleration.point_mass_gravity()],
    Saturn=[propagation_setup.acceleration.point_mass_gravity()],
    Neptune=[propagation_setup.acceleration.point_mass_gravity()]
)
acceleration_settings_Neptune = dict(
    Earth=[propagation_setup.acceleration.point_mass_gravity()],
    Moon=[propagation_setup.acceleration.point_mass_gravity()],
    Sun=[propagation_setup.acceleration.point_mass_gravity()],
    Mercury=[propagation_setup.acceleration.point_mass_gravity()],
    Venus=[propagation_setup.acceleration.point_mass_gravity()],
    Mars=[propagation_setup.acceleration.point_mass_gravity()],
    Jupiter=[propagation_setup.acceleration.point_mass_gravity()],
    Saturn=[propagation_setup.acceleration.point_mass_gravity()],
    Uranus=[propagation_setup.acceleration.point_mass_gravity()],
)
acceleration_settings = {
    "LUMIO": acceleration_settings_LUMIO,
    "Earth": acceleration_settings_Earth,
    "Moon": acceleration_settings_Moon,
    "Sun": acceleration_settings_Sun,
    "Mercury": acceleration_settings_Mercury,
    "Venus": acceleration_settings_Venus,
    "Mars": acceleration_settings_Mars,
    "Jupiter": acceleration_settings_Jupiter,
    "Saturn": acceleration_settings_Saturn,
    "Uranus": acceleration_settings_Uranus,
    "Neptune": acceleration_settings_Neptune
}

acceleration_models = propagation_setup.create_acceleration_models(
    body_system, acceleration_settings, bodies_to_propagate, central_bodies)

LUMIO_initial_state = X_LUMIO_ini
celestial_initial_states = propagation.get_initial_state_of_bodies(
    bodies_to_propagate=bodies_to_propagate[1::],
    central_bodies=central_bodies[1::],
    body_system=body_system,
    initial_time=simulation_start_epoch)

#initial_states = np.transpose([np.concatenate((LUMIO_initial_state, celestial_initial_states), axis=0)])
initial_states = np.concatenate((LUMIO_initial_state, celestial_initial_states), axis=0)
### Savings ###
# Is adjustable
dependent_variables_to_save = [
    propagation_setup.dependent_variable.total_acceleration("LUMIO"),
    propagation_setup.dependent_variable.total_acceleration("Moon"),
    propagation_setup.dependent_variable.total_acceleration("Earth"),
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
output_dict = dynamic_simulator.dependent_variable_history
states_dict = dynamic_simulator.state_history
output = np.vstack(list(output_dict.values()))
states = np.vstack(list(states_dict.values()))

print(output[0, :])

LUMIO_wrt_Earth = states[:, 0:6]
Earth_wrt_Sun = states[:, 6:12]
Moon_wrt_Earth = states[:, 12:18]


plt.figure()
plt.plot(0, 0, marker='o', markersize=10, color='blue')
plt.plot(Moon_wrt_Earth[:, 0], Moon_wrt_Earth[:, 1])
plt.plot(LUMIO_wrt_Earth[:, 0], LUMIO_wrt_Earth[:, 1])
plt.figure()
plt.plot(0, 0, marker='o', markersize=10, color='blue')
plt.plot(Moon_wrt_Earth[:, 0], Moon_wrt_Earth[:, 2])
plt.plot(LUMIO_wrt_Earth[:, 0], LUMIO_wrt_Earth[:, 2])
plt.figure()
plt.plot(0, 0, marker='o', markersize=10, color='blue')
plt.plot(Moon_wrt_Earth[:, 1], Moon_wrt_Earth[:, 2])
plt.plot(LUMIO_wrt_Earth[:, 1], LUMIO_wrt_Earth[:, 2])

plt.show()


print("[FullSolarSystemLUMIOpropagation.py] ran successfully")


