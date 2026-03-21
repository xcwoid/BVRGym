import numpy as np
import pymap3d as pm

def to_360(angle):
    # from -180 to 180 deg
    # to 0 to 360 deg
    return (angle + 360) % 360

def dinstance_between_agents(own_agent, target_agent):
    # doesnt really matter which is own and which is target here, since distance is symmetric
    lat0 = own_agent.simObj.get_lat_gc_deg()
    lon0 = own_agent.simObj.get_long_gc_deg()
    h0   = own_agent.simObj.get_altitude()

    lat = target_agent.simObj.get_lat_gc_deg()
    lon = target_agent.simObj.get_long_gc_deg()
    h   = target_agent.simObj.get_altitude()
    e, n, u = pm.geodetic2enu(lat, lon, h, lat0, lon0, h0, ell=None, deg=True)
    
    return np.linalg.norm(np.array([e, n, u]))


def bearing_between_agents(own_agent, target_agent):
    # returns bearing from own_agent to target_agent in degrees
    lat0 = own_agent.simObj.get_lat_gc_deg()
    lon0 = own_agent.simObj.get_long_gc_deg()
    h0   = own_agent.simObj.get_altitude()

    lat = target_agent.simObj.get_lat_gc_deg()
    lon = target_agent.simObj.get_long_gc_deg()
    h   = target_agent.simObj.get_altitude()
    e, n, u = pm.geodetic2enu(lat, lon, h, lat0, lon0, h0, ell=None, deg=True)
    
    bearing_rad = np.arctan2(e, n)  # arctan2 returns angle between -pi and pi
    #bearing_rad = translate_semi_to_full_circle(bearing_rad)
    bearing_deg = np.degrees(bearing_rad)
    return bearing_deg


def relative_bearing_between_agents(own_agent, target_agent):
    bearing_to_enemy = bearing_between_agents(own_agent, target_agent)
    own_heading = own_agent.simObj.get_psi()
    angle = (bearing_to_enemy - own_heading + 180) % 360 - 180
    return angle



def unit_vector(vector):
        return vector / np.linalg.norm(vector)       

def angle_between(v1, v2, in_deg = False):
        ''' Return angle between 0 to 180 deg, or 0 to pi '''
        ''' [1,0,0] and [-1,  1, 0] = 135 deg'''
        ''' [1,0,0] and [-1, -1, 0] = 135 deg'''
        v1_u = unit_vector(v1)
        v2_u = unit_vector(v2)
        angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        if in_deg:
            return np.degrees(angle)
        return angle

def dinstance_between_simObj_agent(own_simObj, target_agent):
    # doesnt really matter which is own and which is target here, since distance is symmetric
    lat0 = own_simObj.get_lat_gc_deg()
    lon0 = own_simObj.get_long_gc_deg()
    h0   = own_simObj.get_altitude()
    try:
        lat = target_agent.simObj.get_lat_gc_deg()
        lon = target_agent.simObj.get_long_gc_deg()
        h   = target_agent.simObj.get_altitude()
    except AttributeError:
        # this part is for tests.test_missile where target is a simObj directly
        lat = target_agent.get_lat_gc_deg()
        lon = target_agent.get_long_gc_deg()
        h   = target_agent.get_altitude()
    e, n, u = pm.geodetic2enu(lat, lon, h, lat0, lon0, h0, ell=None, deg=True)
    
    return np.linalg.norm(np.array([e, n, u]))


def enu_between_agents(own_agent, target_agent):
    # returns bearing from own_agent to target_agent in degrees
    lat0 = own_agent.simObj.get_lat_gc_deg()
    lon0 = own_agent.simObj.get_long_gc_deg()
    h0   = own_agent.simObj.get_altitude()

    lat = target_agent.simObj.get_lat_gc_deg()
    lon = target_agent.simObj.get_long_gc_deg()
    h   = target_agent.simObj.get_altitude()
    e, n, u = pm.geodetic2enu(lat, lon, h, lat0, lon0, h0, ell=None, deg=True)    
    return e, n , u