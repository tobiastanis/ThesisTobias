"""
In this file the simulation set-up is presented, so that all simulations or propagations use the same time interval
Adjust the simulation_time and fixed_time_steps only.
Simulation_time for the duration and fixed_time_step for the interval between two epochs within the simulation_time.
"""
import math
import numpy as np
from tudatpy.kernel import constants
from Dataset_reader import simulation_start_epoch
print("Running [Simulation_setup.py]")
### MJD times for datareading ###
t0_mjd = 60390.00           # Start time 21-03-2024 (next few days no stationkeeping
t1_mjd = 60418.00           # 18-04-2024 Next few days no stationkeeping
tend_mjd = 60755.00         # End of life time 21-03-2025


simulation_time = 10            ####### Simulation time in days
simulation_start_epoch = simulation_start_epoch(t0_mjd)
simulation_end_epoch = simulation_start_epoch+simulation_time*constants.JULIAN_DAY

fixed_time_step = 0.25*constants.JULIAN_DAY             # in seconds
#fixed_time_step = 5000
n_steps = math.floor((simulation_end_epoch-simulation_start_epoch)/fixed_time_step)+1
#simulation_span = np.linspace(simulation_start_epoch, simulation_end_epoch, n_steps)
simulation_span = np.linspace(0, simulation_time, n_steps)
#simulation_span_days = simulation_span/constants.JULIAN_DAY



print("[Simulation_setup.py] ran successfully \n")
