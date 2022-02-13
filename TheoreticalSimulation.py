"""
Simulation of two satellites. One Low Lunar orbit and one EarthMoon L2 orbit. Theoretical equations of motion are used.
The theoretical simulation is a success!
The visibility analysis is NOT a succes yet
The conversion to ICRF coordinates is NOT a success yet.
"""
printje = 1     # Toggle 1 for print max and min values distance L2 and LLO and L2 and Moon
norm_plots = 1      # Toggle 1 for plot several figures of the simulation of one nominal orbit

import numpy as np
from scipy.integrate import odeint
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
import matplotlib.pyplot as plt
import Input as I
spice_interface.load_standard_kernels()

# Initial State LUMIO at 21 MArch 2024
x_L2_i = I.x_norm
x_LLO_i = I.x_LLO_theoretical_i
##### Defining third body acceleration function #####
def crtbp(x, t, mu):
    # Normalized distances
    r1 = np.sqrt((x[0] + mu) ** 2 + x[1] ** 2 + x[2] ** 2)
    r2 = np.sqrt((x[0] + mu - 1) ** 2 + x[1] ** 2 + x[2] ** 2)
    # Normalized masses of the primaries
    mu = I.mu

    xdot = [x[3],
            x[4],
             x[5],
            x[0] + 2 * x[4] - (1 - mu) * (x[0] + mu) / r1 ** 3 - mu * (x[0] + mu - 1) / r2 ** 3,
            -2 * x[3] + (1 - (1 - mu) / r1 ** 3 - mu / r2 ** 3) * x[1],
            ((mu - 1) / r1 ** 3 - mu / r2 ** 3) * x[2]
            ]
    return xdot
def twobody_dyn(y, t, mu):
    rx, ry, rz, vx, vy, vz = y
    r = np.array([rx, ry, rz])
    r_abs = np.linalg.norm(r)
    # Acceleration
    ax, ay, az = -r * mu/r_abs**3

    return [vx, vy, vz, ax, ay, az]
# Integration
LLO_states_wrt_Moon = odeint(twobody_dyn,
                             x_LLO_i,
                             I.simulation_span,
                             args=(spice_interface.get_body_gravitational_parameter("Moon"),),
                             rtol=1e-12,
                             atol=1e-12
                             )
L2_states_norm = odeint(crtbp,
                        x_L2_i,
                        I.simulation_span_norm,
                        args=(I.mu,),
                        rtol=1e-12,
                        atol=1e-12
                        )

# Low Lunar Orbiter states wrt
states_LLO = np.array([(1-I.mu)*I.L_char, 0, 0, 0, 0, 0]) + LLO_states_wrt_Moon
# Dimensional L2 states
states_L2 = np.array([I.L_char, I.L_char, I.L_char, I.v_char, I.v_char, I.v_char]) * L2_states_norm
pos_LLO = states_LLO[:, :3]
pos_L2 = states_L2[:, :3]

L2_wrt_LLO_vector = pos_L2 - pos_LLO
intersatellite_distance = np.linalg.norm(L2_wrt_LLO_vector, axis=1)
L2_wrt_Moon_vector = pos_L2 - np.array([(1-I.mu)*I.L_char, 0, 0])
L2_Moon_distance = np.linalg.norm(L2_wrt_Moon_vector, axis=1)


if printje == 1:
    print('Maximum inter-satellite distance [m]:\n', max(intersatellite_distance))
    print('Minimum inter-satellite distance [m]:\n', min(intersatellite_distance))
    print('Maximum distance L2 to Moon [m]:\n', max(L2_Moon_distance))
    print('Minimum distance L2 to Moon [m]:\n', min(L2_Moon_distance))


### Plots ###
# Cartesian element over time LUMIO
P1_pos = np.array([-I.mu * I.L_char, 0, 0])
P2_pos = np.array([(1 - I.mu) * I.L_char, 0, 0])
if norm_plots == 1:
    fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, constrained_layout=True, sharey=False)
    ax1.plot(I.simulation_time_days, states_L2[:, 0]*10**-3)
    ax1.set_title('Distance of the LUMIO wrt the barycenter in x-direction')
    ax1.set_xlabel('Time [days]')
    ax1.set_ylabel('Distance [km]')
    ax2.plot(I.simulation_time_days, states_L2[:, 1]*10**-3)
    ax2.set_title('Distance of the LUMIO wrt the barycenter in y-direction')
    ax2.set_xlabel('Time [days]')
    ax2.set_ylabel('Distance [km]')
    ax3.plot(I.simulation_time_days, states_L2[:, 2]*10**-3)
    ax3.set_title('Distance of the LUMIO wrt the barycenter in z-direction')
    ax3.set_xlabel('Time [days]')
    ax3.set_ylabel('Distance [km]')

    # Cartesian elements over time LLO
    fig2, (ax1, ax2, ax3) = plt.subplots(3, 1, constrained_layout=True, sharey=False)
    ax1.plot(I.simulation_time_days, states_LLO[:, 0]*10**-3)
    ax1.set_title('Distance of the LLO orbiter wrt the barycenter in x-direction')
    ax1.set_xlabel('Time [days]')
    ax1.set_ylabel('Distance [km]')
    ax2.plot(I.simulation_time_days, states_LLO[:, 1]*10**-3)
    ax2.set_title('Distance of the LLO orbiter wrt the barycenter in y-direction')
    ax2.set_xlabel('Time [days]')
    ax2.set_ylabel('Distance [km]')
    ax3.plot(I.simulation_time_days, states_LLO[:, 2]*10**-3)
    ax3.set_title('Distance of the LLO orbiter wrt the barycenter in z-direction')
    ax3.set_xlabel('Time [days]')
    ax3.set_ylabel('Distance [km]')

    # Cartesian elements wrt each other LUMIO
    fig3, (ax1, ax2, ax3) = plt.subplots(1, 3, constrained_layout=True, sharey=False)
    ax1.plot(states_L2[:, 0]*10**-3, states_L2[:, 1]*10**-3)
    ax1.set_title('LUMIO states in xy-plane')
    ax1.set_xlabel('x-direction [km]')
    ax1.set_ylabel('y-direction [km]')
    ax2.plot(states_L2[:, 0]*10**-3, states_L2[:, 2]*10**-3)
    ax2.set_title('LUMIO states in xz-plane')
    ax2.set_xlabel('x-direction [km]')
    ax2.set_ylabel('z-direction [km]')
    ax3.plot(states_L2[:, 1]*10**-3, states_L2[:, 2])
    ax3.set_title('LUMIO states in yz-plane')
    ax3.set_xlabel('y-direction [km]')
    ax3.set_ylabel('z-direction [km]')

    # 2D Three body system including both satellites and primaries
    fig4, (ax1, ax2, ax3) = plt.subplots(3, 1, constrained_layout=True, sharey=False)
    ax1.plot(states_L2[:, 0]*10**-3, states_L2[:, 1]*10**-3)
    ax1.plot(states_LLO[:, 0]*10**-3, states_LLO[:, 1]*10**-3)
    ax1.plot(P1_pos[0]*10**-3, 0, marker='o', markersize=10, color='blue')
    ax1.plot(P2_pos[0]*10**-3, 0, marker='o', markersize=3, color='grey')
    ax1.set_title('LUMIO states in xy-plane (top-view)')
    ax1.set_xlabel('x-direction [km]')
    ax1.set_ylabel('y-direction [km]')
    ax2.plot(states_L2[:, 0]*10**-3, states_L2[:, 2]*10**-3)
    ax2.plot(states_LLO[:, 0]*10**-3, states_LLO[:, 2]*10**-3)
    ax2.plot(P1_pos[0]*10**-3, 0, marker='o', markersize=10, color='blue')
    ax2.plot(P2_pos[0]*10**-3, 0, marker='o', markersize=3, color='grey')
    ax2.set_title('LUMIO states in xz-plane (side-view)')
    ax2.set_xlabel('x-direction [km]')
    ax2.set_ylabel('z-direction [km]')
    ax3.plot(states_L2[:, 1]*10**-3, states_L2[:, 2]*10**-3)
    ax3.plot(states_LLO[:, 1]*10**-3, states_LLO[:, 2]*10**-3)
    ax3.plot(0, 0, marker='o', markersize=10, color='blue')
    ax3.plot(0, 0, marker='o', markersize=3, color='grey')
    ax3.set_title('LUMIO states in yz-plane')
    ax3.set_xlabel('y-direction [km]')
    ax3.set_ylabel('z-direction [km]')

    # 3D Three body system including both satellites and primaries
    plt.figure()
    plot = plt.axes(projection='3d')
    plot.plot3D(states_L2[:, 0] * 10 ** -3, states_L2[:, 1] * 10 ** -3, states_L2[:, 2] * 10 ** -3)
    plot.plot3D(states_LLO[:, 0] * 10 ** -3, states_LLO[:, 1] * 10 ** -3, states_LLO[:, 2] * 10 ** -3)
    plot.plot(P1_pos[0]*10**-3, 0, marker='o', markersize=10, color='blue')
    plot.plot(P2_pos[0]*10**-3, 0, marker='o', markersize=3, color='grey')
    plot.set_xlabel('Primary distance with the origin at barycenter [km]')
    plot.set_ylabel('In orbital plane direction [km]')
    plot.set_zlabel('Orbital height [km]')
    plot.set_title('3D-plot of LUMIO and the LLO orbiter including the primaries')
    plt.legend(['LUMIO states', 'LLO orbiter states', 'Earth', 'Moon'])
    # Distance curve
    plt.figure()
    plt.plot(I.simulation_time_days, intersatellite_distance)
    plt.plot(I.simulation_time_days, L2_Moon_distance)
    plt.xlabel('Time [days]')
    plt.ylabel('Relative distance [km]')
    plt.legend(['Inter-satellite distance', 'LUMIO-Moon distance'])
    plt.title('Relative distance in one orbit LUMIO')

plt.show()

    ########################################################################################################################
    ##### Visbility analysis ##### Klopt nog geen hout van
    ########################################################################################################################
"""
r = L2_wrt_Moon_vector
x = L2_wrt_LLO_vector
#xx = []
#for x_vector, x_vector in zip(x,x):
#    dotje = np.dot(x_vector,x_vector)
#    xx.append(dotje)
#xx = np.array(xx)
x_abs = intersatellite_distance
r_abs = L2_Moon_distance
rx = []
for r_values, x_values in zip(r,x):
    dot = np.dot(r_values,x_values)
    rx.append(dot)
rx = np.array(rx)

factor = rx/x_abs**2
p = np.vstack(factor)*x
p_abs = np.linalg.norm(p, axis=1)

h_abs = np.sqrt(r_abs**2-p_abs**2)

Moon_radius_incl_marge = spice_interface.get_average_radius("Moon")
StringMoonRadius = [Moon_radius_incl_marge]*len(I.simulation_time_days)

plt.figure()
plt.plot(I.simulation_time_days, h_abs)
plt.plot(I.simulation_time_days, StringMoonRadius)
plt.legend(['h', 'Radius Moon'])
plt.show()

factor = np.vstack(rx/xx)
p = factor*x
p_abs = np.linalg.norm(p, axis=1)
dot_pr = []
for p_vector, r_vector in zip(p, r):
    dottie = np.dot(p_vector,r_vector)
    dot_pr.append(dottie)
# Vector center of the Moon perpendicular to p

h = r_abs*np.sin(np.sqrt(1-(dot_pr/(p_abs*r_abs))**2))
#h1 = r_abs*np.sin(np.arccos(dot_pr/(p_abs*r_abs)))
"""