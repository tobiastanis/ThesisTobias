import numpy as np
from scipy.integrate import odeint
import Datapack_to_initial_CRTBP as data
import Input as I
from tudatpy.kernel.interface import spice_interface
import matplotlib.pyplot as plt
spice_interface.load_standard_kernels()

x_L2 = data.x_norm_initial
#x_L2 = np.array([1.1435, 0, -0.1579, 0, -0.2220, 0])
print(x_L2)

def crtbp(x, t, mu):
    # Normalized distances
    #r1 = np.sqrt((x[0] + mu) ** 2 + x[1] ** 2 + x[2] ** 2)
    #r2 = np.sqrt((x[0] + mu - 1) ** 2 + x[1] ** 2 + x[2] ** 2)
    r1 = np.sqrt((mu+x[0])**2 + x[1]**2 + x[2]**2)
    r2 = np.sqrt((1-mu-x[0])**2 + x[1]**2 + x[2]**2)
    # Normalized masses of the primaries
    mu = data.mu

    xdot = [x[3],
            x[4],
            x[5],
            x[0] + 2 * x[4] - (1 - mu) * (x[0] + mu) / r1 ** 3 - mu * (x[0] + mu - 1) / r2 ** 3,
            -2 * x[3] + (1 - (1 - mu) / r1 ** 3 - mu / r2 ** 3) * x[1],
            ((mu - 1) / r1 ** 3 - mu / r2 ** 3) * x[2]
            ]
    return xdot

L2_states_norm = odeint(crtbp,
                        x_L2,
                        I.simulation_span_norm,
                        args=(data.mu,),
                        rtol=1e-12,
                        atol=1e-12)

pos_Moon = (1-data.mu)*data.l_char*10**-3
pos_Earth = -data.mu*data.l_char*10**-3
P1_pos = np.array([pos_Earth, 0, 0])
P2_pos = np.array([pos_Moon, 0, 0])
states_L2 = np.array([data.l_char, data.l_char, data.l_char, data.v_char, data.v_char, data.v_char]) * L2_states_norm


fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, constrained_layout=True, sharey=False)
ax1.plot(I.simulation_time_days, states_L2[:, 0] * 10 ** -3)
ax1.set_title('Distance of the LUMIO wrt the barycenter in x-direction')
ax1.set_xlabel('Time [days]')
ax1.set_ylabel('Distance [km]')
ax2.plot(I.simulation_time_days, states_L2[:, 1] * 10 ** -3)
ax2.set_title('Distance of the LUMIO wrt the barycenter in y-direction')
ax2.set_xlabel('Time [days]')
ax2.set_ylabel('Distance [km]')
ax3.plot(I.simulation_time_days, states_L2[:, 2] * 10 ** -3)
ax3.set_title('Distance of the LUMIO wrt the barycenter in z-direction')
ax3.set_xlabel('Time [days]')
ax3.set_ylabel('Distance [km]')

# Cartesian elements wrt each other LUMIO
fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, constrained_layout=True, sharey=False)
ax1.plot(states_L2[:, 0] * 10 ** -3, states_L2[:, 1] * 10 ** -3)
ax1.plot(states_L2[0,0]*10**-3, states_L2[0,1]*10**-3, marker='o', markersize=10, color='orange')
ax1.set_title('LUMIO states in xy-plane')
ax1.set_xlabel('x-direction [km]')
ax1.set_ylabel('y-direction [km]')
ax2.plot(states_L2[:, 0] * 10 ** -3, states_L2[:, 2] * 10 ** -3)
ax2.set_title('LUMIO states in xz-plane')
ax2.set_xlabel('x-direction [km]')
ax2.set_ylabel('z-direction [km]')
ax3.plot(states_L2[:, 1] * 10 ** -3, states_L2[:, 2])
ax3.set_title('LUMIO states in yz-plane')
ax3.set_xlabel('y-direction [km]')
ax3.set_ylabel('z-direction [km]')

fig3, (ax1, ax2, ax3) = plt.subplots(3, 1, constrained_layout=True, sharey=False)
ax1.plot(states_L2[:, 0] * 10 ** -3, states_L2[:, 1] * 10 ** -3)
ax1.plot(P1_pos[0] * 10 ** -3, 0, marker='o', markersize=10, color='blue')
ax1.plot(P2_pos[0] * 10 ** -3, 0, marker='o', markersize=3, color='grey')
ax1.plot(states_L2[0,0]*10**-3, states_L2[0,1]*10**-3, marker='o', markersize=10, color='orange')
ax1.set_title('LUMIO states in xy-plane (top-view)')
ax1.set_xlabel('x-direction [km]')
ax1.set_ylabel('y-direction [km]')
ax2.plot(states_L2[:, 0] * 10 ** -3, states_L2[:, 2] * 10 ** -3)
ax2.plot(P1_pos[0] * 10 ** -3, 0, marker='o', markersize=10, color='blue')
ax2.plot(P2_pos[0] * 10 ** -3, 0, marker='o', markersize=3, color='grey')
ax2.set_title('LUMIO states in xz-plane (side-view)')
ax2.set_xlabel('x-direction [km]')
ax2.set_ylabel('z-direction [km]')
ax3.plot(states_L2[:, 1] * 10 ** -3, states_L2[:, 2] * 10 ** -3)
ax3.plot(0, 0, marker='o', markersize=10, color='blue')
ax3.plot(0, 0, marker='o', markersize=3, color='grey')
ax3.set_title('LUMIO states in yz-plane')
ax3.set_xlabel('y-direction [km]')
ax3.set_ylabel('z-direction [km]')





plt.show()
