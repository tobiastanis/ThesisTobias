import Input as I
import numpy as np
import TheoreticalSimulation as TS
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
import matplotlib.pyplot as plt
tudat = 1
if tudat == 1:
    import tuatpy_propagation as tudat
    x_LUMIO_tudat = tudat.states_LUMIO
    x_Moon_tudat = tudat.states_Moon
datapack = 1
if datapack == 1:
    import LUMIO_States_reader as lsr
    # States from datapack
    x_LUMIO_data = lsr.state_LUMIO
    x_Moon_data = lsr.state_Moon

spice_interface.load_standard_kernels()
# Use saved states
data_norm_states = TS.L2_states_norm

# Position and velocity vector of the non-dimensional rotational frame wrt barycenter
r_norm = data_norm_states[:, 0:3]
v_norm = data_norm_states[:, 3:6]

t0 = I.t0
period = I.simulation_span

m_char = I.m_char       # Characteristic mass (M_P1+M_P2)
G = constants.GRAVITATIONAL_CONSTANT
mu = I.mu

# Storage
x_Moon = []
x_LUMIO = []
for i in range(len(period)):
    # Time steps
    dt = period[i]
    # Moon state vector
    X_P1P2 = spice_interface.get_body_cartesian_state_at_epoch("Moon", "Earth", "J2000", "NONE", t0+dt)
    x_Moon.append(X_P1P2)
    r_Moon = X_P1P2[0:3]
    v_Moon = X_P1P2[3:6]
    # Characteristic values
    l_char = np.linalg.norm(r_Moon)
    t_char = np.sqrt(l_char**3/(G*m_char))
    v_char = l_char/t_char
    # Primary position in rotating frame (non-dim)
    r_p1 = np.array([-mu, 0, 0])
    v_p1 = np.array([0, 0, 0])
    # Non-dimensional rotational state vector
    r_rot_nd = r_norm[i,:]
    v_rot_nd = v_norm[i,:]
    # Non-dimensional primary centered
    r_pc_nd = r_rot_nd+r_p1
    v_pc_nd = v_rot_nd+v_p1
    # Primary centered dimensional
    r_pc = r_pc_nd*l_char
    v_pc = v_pc_nd*v_char
    sv_pc = np.transpose([np.concatenate((r_pc,v_pc), axis=0)])
    #print(sv_pc)
    # Attitude matrix between inertial and rotating frame
    X_ref = r_Moon / np.linalg.norm(r_Moon)
    Z_ref = np.cross(r_Moon,v_Moon)/np.linalg.norm(np.cross(r_Moon,v_Moon))
    Y_ref = np.cross(Z_ref,X_ref)
    A_ref = np.transpose([X_ref, Y_ref, Z_ref])
    # Instantaneous angular velocity
    omega = np.linalg.norm(np.cross(r_Moon,v_Moon))/(np.linalg.norm(r_Moon)**2)
    # The creation of B
    C11 = A_ref[0,0]; C12 = A_ref[0,1]; C13 = A_ref[0,2]
    C21 = A_ref[1,0]; C22 = A_ref[1,1]; C23 = A_ref[1,2]
    C31 = A_ref[2,0]; C32 = A_ref[2,1]; C33 = A_ref[2,2]
    B_ref = np.array([[omega*C12, -omega*C11, 0],
                      [omega*C22, -omega*C21, 0],
                      [omega*C32, -omega*C31, 0]])
    O_ref = np.zeros((3,3))
    # Full transformation matrix
    A_top = np.concatenate((A_ref, O_ref), axis=1)
    A_bot = np.concatenate((B_ref, A_ref), axis=1)
    A_full = np.concatenate((A_top, A_bot), axis=0)
    # State vector in the ephemeris frame
    x_ephem = np.transpose(np.matmul(A_full, sv_pc))[0]
    #print(x_ephem)
    x_LUMIO.append(x_ephem)
# States from conversion
x_LUMIO = np.array(x_LUMIO)
x_Moon = np.array(x_Moon)


########################################################################################################################
############################################Plots Plots Plots ##########################################################

plt.figure()
plot = plt.axes(projection='3d')
plot.plot3D(x_LUMIO[:,0], x_LUMIO[:,1], x_LUMIO[:,2], color='purple')
plot.plot3D(x_Moon[:,0], x_Moon[:,1], x_Moon[:,2], color='grey')
plot.plot3D(0, 0, 0, marker='o', markersize=10, color='blue')
plot.plot3D(x_LUMIO[0,0], x_LUMIO[0,1], x_LUMIO[0,2], marker='o', markersize=4, color='purple')
plot.plot3D(x_Moon[0,0], x_Moon[0,1], x_Moon[0,2], marker='o', markersize=5, color='grey')
if datapack==1:
    plot.plot3D(x_LUMIO_data[:, 0]*10**3, x_LUMIO_data[:, 1]*10**3, x_LUMIO_data[:, 2]*10**3, color='orange')
    plot.plot3D(x_LUMIO_data[0, 0]*10**3, x_LUMIO_data[0, 1]*10**3, x_LUMIO_data[0, 2]*10**3, marker='o', markersize=4, color='orange')
    plot.plot3D(x_Moon_data[:, 0]*10**3, x_Moon_data[:, 1]*10**3, x_Moon_data[:, 2]*10**3, color='black')
    plot.plot3D(x_Moon_data[0, 0]*10**3, x_Moon_data[0, 1]*10**3, x_Moon_data[0, 2]*10**3, color='black')
    plot.plot3D(x_LUMIO_tudat[:, 0], x_LUMIO_tudat[:, 1], x_LUMIO_tudat[:, 2], color='red')
    plot.plot3D(x_LUMIO_tudat[0, 0], x_LUMIO_tudat[0, 1], x_LUMIO_tudat[0, 2], marker='o', markersize=4, color='red')
    plot.plot3D(x_Moon_tudat[:, 0], x_Moon_tudat[:, 1], x_Moon_tudat[:, 2], color='brown')
    plot.plot3D(x_Moon_tudat[0, 0], x_Moon_tudat[0, 1], x_Moon_tudat[0, 2], marker='o', markersize=4, color='brown')
    plt.legend(['Trajectory LUMIO', 'Trajectory Moon', 'Earth', 'Initial state LUMIO', 'Initial state Moon',
            'Trajectory LUMIO datapack', 'Initial state LUMIO datapack', 'Trajectory Moon datapack',
                'Initial state Moon datapack', 'Trajectory LUMIO tudat', 'Initial State LUMIO tudat',
                'Trajectory Moon tudat', 'Initial state Moon tudat'])
else:
    plt.legend(['Trajectory LUMIO', 'Trajectory Moon', 'Earth', 'Initial state LUMIO', 'Initial state Moon'])
plt.title('State of LUMIO after CRTBP to ICRF conversion')
plt.xlabel('X-direction [m]'); plt.ylabel('Y-direction [m]'); plot.set_zlabel('Z-direction [m]')

plt.figure()
plt.plot(x_LUMIO[:,0], x_LUMIO[:,1], color='purple')
plt.plot(x_Moon[:,0], x_Moon[:,1], color='grey')
plt.plot(0, 0, marker='o', markersize=10, color='blue')
plt.plot(x_LUMIO[0,0], x_LUMIO[0,1], marker='o', markersize=4, color='purple')
plt.plot(x_Moon[0,0], x_Moon[0,1], marker='o', markersize=5, color='grey')
if datapack==1:
    plt.plot(x_LUMIO_data[:, 0]*10**3, x_LUMIO_data[:, 1]*10**3, color='orange')
    plt.plot(x_LUMIO_data[0, 0]*10**3, x_LUMIO_data[0, 1]*10**3, marker='o', markersize=4, color='orange')
    plt.plot(x_Moon_data[:, 0]*10**3, x_Moon_data[:, 1]*10**3, color='black')
    plt.plot(x_Moon_data[0, 0]*10**3, x_Moon_data[0, 1]*10**3, color='black')
    plt.legend(['Trajectory LUMIO', 'Trajectory Moon', 'Earth', 'Initial state LUMIO', 'Initial state Moon',
            'Trajectory LUMIO datapack', 'Initial state LUMIO datapack', 'Trajectory Moon datapack', 'Initial state Moon datapack'])
else:
    plt.legend(['Trajectory LUMIO', 'Trajectory Moon', 'Earth', 'Initial state LUMIO', 'Initial state Moon'])
plt.title('State of LUMIO after CRTBP to ICRF conversion (xy-plane')
plt.xlabel('X-direction [m]'); plt.ylabel('Y-direction [m]')

plt.figure()
plt.plot(x_LUMIO[:,0], x_LUMIO[:,2], color='purple')
plt.plot(x_Moon[:,0], x_Moon[:,2], color='grey')
plt.plot(0, 0, marker='o', markersize=10, color='blue')
plt.plot(x_LUMIO[0,0], x_LUMIO[0,2], marker='o', markersize=4, color='purple')
plt.plot(x_Moon[0,0], x_Moon[0,2], marker='o', markersize=5, color='grey')
if datapack==1:
    plt.plot(x_LUMIO_data[:, 0]*10**3, x_LUMIO_data[:, 2]*10**3, color='orange')
    plt.plot(x_LUMIO_data[0, 0]*10**3, x_LUMIO_data[0, 2]*10**3, marker='o', markersize=4, color='orange')
    plt.plot(x_Moon_data[:, 0]*10**3, x_Moon_data[:, 2]*10**3, color='black')
    plt.plot(x_Moon_data[0, 0]*10**3, x_Moon_data[0, 2]*10**3, color='black')
    plt.legend(['Trajectory LUMIO', 'Trajectory Moon', 'Earth', 'Initial state LUMIO', 'Initial state Moon',
            'Trajectory LUMIO datapack', 'Initial state LUMIO datapack', 'Trajectory Moon datapack', 'Initial state Moon datapack'])
else:
    plt.legend(['Trajectory LUMIO', 'Trajectory Moon', 'Earth', 'Initial state LUMIO', 'Initial state Moon'])
plt.title('State of LUMIO after CRTBP to ICRF conversion (xz-plane')
plt.xlabel('X-direction [m]'); plt.ylabel('Z-direction [m]')

plt.figure()
plt.plot(x_LUMIO[:,1], x_LUMIO[:,2], color='purple')
plt.plot(x_Moon[:,1], x_Moon[:,2], color='grey')
plt.plot(0, 0, marker='o', markersize=10, color='blue')
plt.plot(x_LUMIO[0,1], x_LUMIO[0,2], marker='o', markersize=4, color='purple')
plt.plot(x_Moon[0,1], x_Moon[0,2], marker='o', markersize=5, color='grey')
if datapack==1:
    plt.plot(x_LUMIO_data[:, 1]*10**3, x_LUMIO_data[:, 2]*10**3, color='orange')
    plt.plot(x_LUMIO_data[0, 1]*10**3, x_LUMIO_data[0, 2]*10**3, marker='o', markersize=4, color='orange')
    plt.plot(x_Moon_data[:, 1]*10**3, x_Moon_data[:, 2]*10**3, color='black')
    plt.plot(x_Moon_data[0, 1]*10**3, x_Moon_data[0, 2]*10**3, color='black')
    plt.legend(['Trajectory LUMIO', 'Trajectory Moon', 'Earth', 'Initial state LUMIO', 'Initial state Moon',
            'Trajectory LUMIO datapack', 'Initial state LUMIO datapack', 'Trajectory Moon datapack', 'Initial state Moon datapack'])
else:
    plt.legend(['Trajectory LUMIO', 'Trajectory Moon', 'Earth', 'Initial state LUMIO', 'Initial state Moon'])
plt.title('State of LUMIO after CRTBP to ICRF conversion (yz-plane')
plt.xlabel('Y-direction [m]'); plt.ylabel('Z-direction [m]')

if datapack == 1:
    fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, constrained_layout=True, sharey=False)
    ax1.plot(x_LUMIO[:, 0], x_LUMIO[:, 1])
    ax1.plot(x_LUMIO_data[:, 0]*10**3, x_LUMIO_data[:, 1]*10**3)
    ax1.set_title('Trajectory in xy-plane')
    ax1.set_xlabel('x-direction [m]')
    ax1.set_ylabel('y-direction [m]')
    ax2.plot(x_LUMIO[:, 0], x_LUMIO[:, 2])
    ax2.plot(x_LUMIO_data[:, 0]*10**3, x_LUMIO_data[:, 2]*10**3)
    ax2.set_title('Trajectory in xz-plane')
    ax2.set_xlabel('x-direction [m]')
    ax2.set_ylabel('z-direction [m]')
    ax3.plot(x_LUMIO[:, 1], x_LUMIO[:, 2])
    ax3.plot(x_LUMIO_data[:, 1]*10**3, x_LUMIO_data[:, 2]*10**3)
    ax3.set_title('Trajectory in yz-plane')
    ax3.set_xlabel('y-direction [m]')
    ax3.set_ylabel('z-direction [m]')
    plt.legend(['LUMIO ICRF Trajectory', 'LUMIO  Datapack Trajectory'])
plt.show()

