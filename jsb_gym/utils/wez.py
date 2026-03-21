from jsb_gym.utils.units import f2m
from jsb_gym.utils.geospatial import dinstance_between_agents, enu_between_agents
from jsb_gym.utils.rotation import enu_to_body
from jsb_gym.utils.geometry import angle_between_vectors
import numpy as np

def gunsnap(r_wez, max_range=f2m(3_000), min_range=f2m(500)):
    """Convert a distance to target to damage.

    Args:
        r_wez (float): distance to target.

    Returns:
        float: damage.
    """
    if r_wez > max_range:
        return 0.0
    elif r_wez < max_range and r_wez >= min_range:
        return (max_range - r_wez)/(max_range - min_range)
    else:
        return 0.0

def angle_to_target_in_body_frame_deg(own_agent, target_agent):
    e, n, u = enu_between_agents(own_agent, target_agent)
    v_enu = np.array([e,n,u])
    v_enu_body = enu_to_body(v_enu, own_agent)
    aircraft_nose_axis = np.array([1,0,0])
    angle_to_target = angle_between_vectors(aircraft_nose_axis, v_enu_body)
    return np.degrees(angle_to_target)


def in_gun_wez(own_agent, target_agent, angle_limit = 2, max_range = 3_000, min_range = 500):
    d = dinstance_between_agents(own_agent, target_agent)
    ang = angle_to_target_in_body_frame_deg(own_agent, target_agent)
    if abs(ang) < angle_limit:
        return gunsnap(d, max_range=f2m(max_range), min_range=f2m(min_range))
    return 0
