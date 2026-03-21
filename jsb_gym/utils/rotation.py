import numpy as np

def Rz(yaw):
    return np.array([
        [ np.cos(yaw), -np.sin(yaw), 0],
        [ np.sin(yaw),  np.cos(yaw), 0],
        [ 0,            0,           1]
    ])

def Ry(pitch):
    return np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [ 0,             1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

def Rx(roll):
    return np.array([
        [1, 0,            0           ],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])

def enu_to_body(v_enu, agent):
    roll = agent.simObj.get_phi()
    pitch = agent.simObj.get_theta()
    yaw = agent.simObj.get_psi()
    
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians((yaw - 90 )%360)
    R = Rx(roll) @ Ry(pitch) @ Rz(yaw)
    return R @ v_enu


if __name__ == "__main__":
    v_enu = [0 , 0 , 1]
    roll = np.radians(0)
    pitch = np.radians(0)
    yaw = np.radians((0 - 90 )%360)
    R = Rx(roll) @ Ry(pitch) @ Rz(yaw)
    v = R @ v_enu
    print(v)