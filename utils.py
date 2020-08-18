import numpy as np

from scipy.spatial.transform import Rotation as R


def eulerAnglesToRotationMatrix(theta) :
    """
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
                    
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                    
                    
    R = R_z @ R_y @ R_x
    """
    return R.from_euler('xyz', theta).as_matrix()


def d_quat(q1, q2):
    quat1 = R.from_quat(q1)
    quat2 = R.from_quat(q2)
    q3 = quat2 * quat1.inv()
    velo = q3.as_euler('xyz')/1
    Omega = 0.5 * np.array([[0, -velo[0], -velo[1], -velo[2]], [velo[0], 0, velo[2], -velo[1]], [velo[1], -velo[2], 0, velo[0]], [velo[2], velo[1], -velo[0], 0]])
    return Omega * 0

# Lissage des signaux
def movingaverage(values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.zeros(values.shape)
    sma[:window-1] = values[:window-1]
    sma[window-1:] = np.convolve(values, weights, 'valid')
    return sma
    

def quaternion_to_matrix(q):
    return R.from_quat(q).as_matrix()


def quaternion_to_euler(q):
    return R.from_quat(q).as_euler(seq='xyz')


def update_orientation_quaternion(prev_q, angular_rate, dt):
    q1 = R.from_quat(prev_q)
    q2 = R.from_euler(seq='xyz', angles=angular_rate*dt)
    q_updated = q1 * q2
    
    return q_updated.as_quat()


def nominal_state_predict(prev_state, input_data, dt):
    # Receive information
    p, v, q, a_b, omega_b, g = prev_state[:3], prev_state[3:6], prev_state[6:10], prev_state[10:13], prev_state[13:16], prev_state[16:19]
    a_m, omega_m = input_data[0:3], input_data[3:6]

    # Prediction
    p_predict = p + v * dt + 0.5 * (quaternion_to_matrix(q) @ (a_m - a_b) + g) * dt**2
    v_predict = v + (quaternion_to_matrix(q) @ (a_m - a_b) + g) * dt
    q_predict = update_orientation_quaternion(q, omega_m - omega_b, dt)
    a_b_predict = a_b
    omega_b_predict = omega_b
    g_predict = g

    return np.concatenate([p_predict, v_predict, q_predict, a_b_predict, omega_b_predict, g_predict])


def skew(omega):
    assert omega.shape == (3,)
    return np.array([[  0,          -omega[2],  omega[1]    ],
                     [  omega[2],   0,          -omega[0]   ],
                     [  -omega[1],  omega[0],   0           ]])

def error_state_predict(prev_delta_x, prev_P_delta_x, x, input_data, dt, V_i, Theta_i, A_i, Omega_i):
    # Receive information
    p, v, q, a_b, omega_b, g = x[:3], x[3:6], x[6:10], x[10:13], x[13:16], x[16:19]
    a_m, omega_m = input_data[0:3], input_data[3:6]

    # Fx, Fi, Qi matrices - equation 270, page 61, https://arxiv.org/pdf/1711.02508.pdf
    Fx = np.identity(18)
    R_matrix = quaternion_to_matrix(q)
    Fx[:3, 3:6] = np.eye(3) * dt
    Fx[3:6, 6:9] = -skew(R_matrix @ (a_m - a_b)) * dt
    Fx[3:6, 9:12] = -R_matrix * dt
    Fx[6:9, 12:15] = -R_matrix * dt
    Fx[3:6, 15:] = np.eye(3) * dt

    Fi = np.zeros((18, 12))
    Fi[3:6, :3] = np.eye(3) 
    Fi[6:9, 3:6] = np.eye(3) 
    Fi[9:12, 6:9] = np.eye(3)
    Fi[12:15, 9:12] = np.eye(3)

    Qi = np.zeros((12, 12))
    Qi[:3, :3] = V_i
    Qi[3:6, 3:6] = Theta_i
    Qi[6:9, 6:9] = A_i
    Qi[9:12, 9:12] = Omega_i

    delta_x_predict = Fx @ prev_delta_x
    P_delta_x_predict = Fx @ prev_P_delta_x @ Fx.T + Fi @ Qi @ Fi.T

    return delta_x_predict, P_delta_x_predict


def zero_velocity_update(x, P_delta_x, V, velo):
    # Receive information
    p, v, q, a_b, omega_b, g = x[:3], x[3:6], x[6:10], x[10:13], x[13:16], x[16:19]

    # Section 6.1, https://arxiv.org/pdf/1711.02508.pdf

    Q_delta_theta = 0.5 * np.array([[   -q[0],  -q[1],  -q[2]   ],
                                    [   q[3],   q[2],  -q[1]    ],
                                    [   -q[2],   q[3],   q[0]   ],
                                    [   q[1],  -q[0],   q[3]    ]])
    
    X_delta_x = np.zeros((19, 18))
    X_delta_x[:6, :6] = np.eye(6)
    X_delta_x[6:10, 6:9] = Q_delta_theta
    X_delta_x[10:, 9:] = np.eye(9)

    H_x = np.zeros((3, 19))
    H_x[:3, 3:6] = np.eye(3)
    
    H = H_x @ X_delta_x

    K = P_delta_x @ H.T @ np.linalg.inv((H @ P_delta_x @ H.T + V))
    delta_x_update = K @ (np.array([*velo]) - H_x @ x)
    
    P_delta_x_update = (np.eye(18) - K @ H) @ P_delta_x @ (np.eye(18) - K @ H).T + K @ V @ K.T

    return delta_x_update, P_delta_x_update


def momentary_velocity_update(x, P_delta_x, V, velo):
    # Receive information
    p, v, q, a_b, omega_b, g = x[:3], x[3:6], x[6:10], x[10:13], x[13:16], x[16:19]

    # Section 6.1, https://arxiv.org/pdf/1711.02508.pdf

    Q_delta_theta = 0.5 * np.array([[   -q[0],  -q[1],  -q[2]   ],
                                    [   q[3],   q[2],  -q[1]    ],
                                    [   -q[2],   q[3],   q[0]   ],
                                    [   q[1],  -q[0],   q[3]    ]])
    
    X_delta_x = np.zeros((19, 18))
    X_delta_x[:6, :6] = np.eye(6)
    X_delta_x[6:10, 6:9] = Q_delta_theta
    X_delta_x[10:, 9:] = np.eye(9)

    H_x = np.zeros((1, 19))
    H_x[0, 3:6] = v/np.linalg.norm(v)
    print(v)
    
    H = H_x @ X_delta_x

    K = P_delta_x @ H.T @ np.linalg.inv((H @ P_delta_x @ H.T + V))
    delta_x_update = K @ np.array([velo - np.linalg.norm(v)])
    
    P_delta_x_update = (np.eye(18) - K @ H) @ P_delta_x @ (np.eye(18) - K @ H).T + K @ V @ K.T

    return delta_x_update, P_delta_x_update


def injection_obs_err_to_nominal_state(x, delta_x):
    # Receive information
    p, v, q, a_b, omega_b, g = x[:3], x[3:6], x[6:10], x[10:13], x[13:16], x[16:19]
    delta_p, delta_v, delta_theta, delta_a_b, delta_omega_b, delta_g = delta_x[:3], delta_x[3:6], delta_x[6:9], delta_x[9:12], delta_x[12:15], delta_x[15:18]

    # Section 6.2, https://arxiv.org/pdf/1711.02508.pdf
    p_update = p + delta_p
    v_update = v + delta_v
    # q_update = update_orientation_quaternion(q, delta_theta, 1)
    R_matrix = quaternion_to_matrix(q)
    omega = np.array([[0,-delta_x[8], delta_x[7]],[delta_x[8],0,-delta_x[6]],[-delta_x[7],delta_x[6],0]])
    R_matrix = (np.identity(3) + omega) @ R_matrix
    q_update = R.from_matrix(R_matrix).as_quat()
    a_b_update = a_b + delta_a_b
    omega_b_update = omega_b + delta_omega_b
    g_update = g + delta_g

    return np.concatenate([p_update, v_update, q_update, a_b_update, omega_b_update, g_update])



def ESKF_reset(delta_x, P_delta_x):
    # Receive information
    delta_theta = delta_x[6:9]

    # Section 6.3, https://arxiv.org/pdf/1711.02508.pdf
    delta_x_update = np.zeros((18,))
    
    G = np.identity(18)
    G[6:9, 6:9] = np.eye(3) + skew(0.5 * delta_theta)

    
    P_delta_x_update = G @ P_delta_x @ G.T

    return delta_x_update, P_delta_x_update


def SHOE(imudata, g, W=5, G=4.1e8, sigma_a=0.00098**2, sigma_w=(8.7266463e-5)**2):
    T = np.zeros(np.int(np.floor(imudata.shape[0]/W)+1))
    zupt = np.zeros(imudata.shape[0])
    a = np.zeros((1,3))
    w = np.zeros((1,3))
    inv_a = 1/sigma_a
    inv_w = 1/sigma_w
    acc = imudata[:,0:3]
    gyro = imudata[:,3:6]

    i=0
    for k in range(0,imudata.shape[0]-W+1,W): #filter through all imu readings
        smean_a = np.mean(acc[k:k+W,:],axis=0)
        for s in range(k,k+W):
            a.put([0,1,2],acc[s,:])
            w.put([0,1,2],gyro[s,:])
            T[i] += inv_a*( (a - g * smean_a/np.linalg.norm(smean_a)).dot(( a - g * smean_a/np.linalg.norm(smean_a)).T)) #acc terms
            T[i] += inv_w*( (w).dot(w.T) )
        zupt[k:k+W].fill(T[i])
        i+=1
    zupt = zupt/W
    return zupt < G