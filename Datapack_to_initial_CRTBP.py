"""
Converting the initial value of LUMIO at 21 MArch 2024 to CRTBP
"""
import numpy as np
import LUMIO_States_reader as lsr
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
spice_interface.load_standard_kernels()

# Modified Julian Date at 24 March 2024
t_MJD = '60390.00000'
# Initial Ephemeris Time
t_ET = float(lsr.LUMIOdata_timespan[0, 1])
# LUMIO initial state ICRF P1 based
LUMIO_initial_value_data = lsr.state_LUMIO[0,:]*10**3
# Initial state Moon
X_Moon = spice_interface.get_body_cartesian_state_at_epoch("Moon", "Earth", "J2000", "NONE", t_ET)
r_Moon = X_Moon[0:3]
v_Moon = X_Moon[3:6]

####### Conversion to CRTBP #######
# Characteristic units
l_char = np.linalg.norm(r_Moon)
m_char = spice_interface.get_body_gravitational_parameter("Earth")/constants.GRAVITATIONAL_CONSTANT+spice_interface.get_body_gravitational_parameter("Moon")/constants.GRAVITATIONAL_CONSTANT
t_char = np.sqrt(l_char**3/(constants.GRAVITATIONAL_CONSTANT*m_char))
v_char = l_char/t_char
mu = spice_interface.get_body_gravitational_parameter("Moon")/(spice_interface.get_body_gravitational_parameter("Moon")+spice_interface.get_body_gravitational_parameter("Earth"))
# Matrix A
X_ref = r_Moon / l_char
Z_ref = np.cross(r_Moon, v_Moon)/np.linalg.norm(np.cross(r_Moon, v_Moon))
Y_ref = np.cross(Z_ref,X_ref)
A_ref = np.transpose([X_ref, Y_ref, Z_ref])
# Instantaneous angular velocity
omega = np.linalg.norm(np.cross(r_Moon, v_Moon)) / (np.linalg.norm(r_Moon) ** 2)
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

# Back to primary centered dimensional (statevector_primarycentered)
x_ephem = np.transpose([LUMIO_initial_value_data])
sv_pc = np.matmul(np.transpose(A_full), x_ephem)
r_pc = np.transpose(sv_pc[0:3])[0]
v_pc = np.transpose(sv_pc[3:6])[0]
# Back to normalized primary centered
r_pc_nd = r_pc/l_char
v_pc_nd = v_pc/v_char
#print(r_pc_nd, v_pc_nd)
# State primary wrt barycenter
r_p1 = np.array([-mu, 0, 0])
v_p1 = np.array([0, 0, 0])
# Non-dimensional state in rotating reference frame wrt barycenter
r_rot_nd = r_pc_nd-r_p1
v_rot_nd = v_pc_nd-v_p1
x_norm_initial = np.concatenate((r_rot_nd, v_rot_nd), axis=0)
#print(x_norm_initial)