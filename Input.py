"""
This file contains the normalized barycentric initial conditions of LUMIO and initial values for a two body problem Low
Lunar orbiter. Both values are used for a theoretical state simulation as described in TheoreticalSimulation.py.

Besides the initial states, the simulation set-up regarding time is provided in order that all simulation run the same
timesteps and time interval.
"""

import numpy as np
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
spice_interface.load_standard_kernels()

#################### LUMIO initial states at 21 March 2024 ########################
# Normalized initial states of LUMIO
x_norm = np.array([1.1435, 0, -0.1579, 0, -0.2220, 0])
# State Moon wrt Earth
t0 = 764251269.1826417446136475     # Julian seconds to 21 MArch 2024
X_P1P2 = spice_interface.get_body_cartesian_state_at_epoch("Moon", "Earth", "J2000", "NONE", t0)
# Characterisic units based on Moons position wrt Earth
L_char = np.linalg.norm(X_P1P2[0:3])
m_char = spice_interface.get_body_gravitational_parameter("Earth")/constants.GRAVITATIONAL_CONSTANT+spice_interface.get_body_gravitational_parameter("Moon")/constants.GRAVITATIONAL_CONSTANT
t_char = np.sqrt(L_char**3/(constants.GRAVITATIONAL_CONSTANT*m_char))
v_char = L_char/t_char
mu = spice_interface.get_body_gravitational_parameter("Moon")/(spice_interface.get_body_gravitational_parameter("Earth")+spice_interface.get_body_gravitational_parameter("Moon"))

########################################################################################################################
### Low Lunar Orbiter ###
# Circular Restricted Two-Body Problem
h_orbit = 60E3      # Orbital height above surface [m]
r_LLO = spice_interface.get_average_radius("Moon")          # [m]
v_LLO = np.sqrt(spice_interface.get_body_gravitational_parameter("Moon")/r_LLO)
x_LLO_theoretical_i = np.array([0, r_LLO, 0, 0, 0, v_LLO])

############# Simulation Set up #####################
simulation_time = 10                    # [days] Adjustable
n_steps = 30000                         # Number of steps, adjustable
simulation_start_epoch = 0.0            # Noon 21 March 2024
simulation_end_epoch = simulation_start_epoch+simulation_time*constants.JULIAN_DAY
simulation_end_epoch_norm = simulation_end_epoch/(t_char)
simulation_span = np.linspace(simulation_start_epoch, simulation_end_epoch, n_steps)
simulation_span_norm = np.linspace(simulation_start_epoch, simulation_end_epoch_norm, n_steps)
simulation_time_days = simulation_span/constants.JULIAN_DAY
fixed_step_size = 1000
