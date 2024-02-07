import numpy as np
import geopy
from geopy.distance import geodesic
from geopy import distance
import pyproj
import pymap3d as pm

class Geo():
    def __init__(self):
        self.geodesic = pyproj.Geod(ellps='WGS84')

    def get_random_position_in_circle(self, lat0 , long0, d = [40e3, 100e3], b = [0, 360]):
        # input in [km] 
        # get a random lat long for aircraft/missile position
        # lat0, long0 [deg]
        # d [km] distance 
        # b [deg] bearing 
        origin = geopy.Point(lat0, long0)        
        d = np.sqrt(np.random.uniform(d[0]**2, d[1]**2))
        b = np.random.uniform(b[0], b[1])
        destination = geodesic(meters=d).destination(origin, b)
        lat, long = destination.latitude, destination.longitude      
        return lat, long, d, b
    
    def db2latlong(sef, lat0, long0, d, b):
        #from distance and bearing to lat and long 
        # lat0, long0 [deg]
        # d [km] distance 
        # b [deg] bearing 
        origin = geopy.Point(lat0, long0)        
        destination = geodesic(meters=d).destination(origin, b)
        lat, long = destination.latitude, destination.longitude      
        return lat, long

    def get_bearing(self, lat_tgt, long_tgt, lat, long):
        fwd_azimuth, back_azimuth, distance = self.geodesic.inv(long_tgt, lat_tgt, long, lat)
        if fwd_azimuth < 0:
            fwd_azimuth += 360
        return fwd_azimuth


    def get_relative_unit_position_NED(self, lat0, lon0, h0, lat, lon, h):
        # The local coordinate origin
        #lat0 = TAU.get_lat_gc_deg() # deg
        #lon0 = TAU.get_long_gc_deg()  # deg
        #h0 = TAU.get_altitude()     # meters

        # The point of interest
        #lat = Tgt.get_lat_gc_deg() # deg
        #lon = Tgt.get_long_gc_deg()  # deg
        #h = Tgt.get_altitude()     # meters

        east, north , up = pm.geodetic2enu(lat, lon, h, lat0, lon0, h0)
        down = -up
        self.d_tgt_east = east
        self.d_tgt_north = north
        self.d_tgt_down = down
        try:
            self.position_tgt_NED_norm = round(np.linalg.norm(np.array([east, north, down])))            
        except ValueError:          
            print('position_tgt_NED_norm value error')
            print(east, north, down)
            print(lat, lon, h, lat0, lon0, h0, self.position_tgt_NED_norm_old)
            self.position_tgt_NED_norm = self.position_tgt_NED_norm_old
            print('Sim, time: ', self.get_sim_time_sec())
            
        self.position_tgt_NED_norm_old = self.position_tgt_NED_norm
        
        return east, north, down

    
    def geo2km(self, lat, long, lat_ref, long_ref, time_stamped = False):
        X = []
        #print(lat)
        for i in lat:
            if time_stamped:
                if i[1] < lat_ref:
                    X.append((i[0],-geodesic((i[1], long_ref), (lat_ref, long_ref)).kilometers) )
                else:
                    X.append( (i[0],geodesic((i[1], long_ref), (lat_ref, long_ref)).kilometers))
            else:

                if i < lat_ref:
                    X.append(-geodesic((i, long_ref), (lat_ref, long_ref)).kilometers)
                else:
                    X.append( geodesic((i, long_ref), (lat_ref, long_ref)).kilometers)
        
        
        Y = []
        for i in long:
            if time_stamped:
                if i[1] < long_ref:
                    Y.append( (i[0],-geodesic((lat_ref, i[1]), (lat_ref, long_ref)).kilometers))
                else:
                    Y.append(  (i[0],geodesic((lat_ref, i[1]), (lat_ref, long_ref)).kilometers)) 
            
            else:
                if i < long_ref:
                    Y.append( -geodesic((lat_ref, i), (lat_ref, long_ref)).kilometers)
                else:
                    Y.append(  geodesic((lat_ref, i), (lat_ref, long_ref)).kilometers) 
        return X, Y

class toolkit(object):
    def __init__(self):
        self.diff_heading = np.array([None, None, None])
        self.diff_heading_abs = np.array([None, None, None])

    def translate_semi_to_full_circle(self, angle):
        '''
        From
        [-pi , pi]
        to 
        [0 , 2pi] 
        '''
        #print(np.degrees(angle))
        if angle < 0.0:
            return 2*np.pi + angle
        else:
            return angle
    
    def get_heading_difference(self, psi_ref, psi_deg):
        diff_cw = psi_ref - psi_deg
        diff_ccw = psi_ref - (psi_deg+ 360)
        diff_ccw_r = psi_ref + 360 - psi_deg
        self.diff_heading[0] = diff_cw
        self.diff_heading[1] = diff_ccw
        self.diff_heading[2] = diff_ccw_r
        self.diff_heading_abs[0] = abs(diff_cw)
        self.diff_heading_abs[1] = abs(diff_ccw)
        self.diff_heading_abs[2] = abs(diff_ccw_r)

        return self.diff_heading[np.argmin(self.diff_heading_abs)]

    def f2m(self, x):
        # feet to meters 
        return x*0.3048

    def m2f(self, x):
        '''meters to feet'''
        return x/0.3048

    def lbs2kg(self, x):
        # pounds to kg 
        return x*0.4535924

    def unit_vector(self, vector):
        return vector / np.linalg.norm(vector)

    def angle_between(self, v1, v2, in_deg = False):
        ''' Return angle between 0 to 180 deg, or 0 to pi '''
        ''' [1,0,0] and [-1,  1, 0] = 135 deg'''
        ''' [1,0,0] and [-1, -1, 0] = 135 deg'''
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        if in_deg:
            return np.degrees(angle)
        else:
            return angle

    def truncate_heading(self, heading):
        # rest when a full circle rotation has been done
        if heading < 0:
            return 360 + heading 
        elif heading > 360:
            return heading - 360
        else:
            return heading

    def scale_between(self, a, a_min, a_max):
        # input between min and max
        # return scaled value between -1 and 1  
        return (2 *(a - a_min)/(a_max-a_min)) - 1
    
    def scale_between_inv(self, a, a_min, a_max):
        # input -1 to 1 
        # return scaled value between -min and max  
        return (a + 1)*(a_max-a_min)*0.5 + a_min
    
    def get_relative_target_position_NED(self, fdm_tgt, fdm):

        lat_tgt  = fdm_tgt.fdm['position/lat-gc-deg']  
        long_tgt = fdm_tgt.fdm['position/long-gc-deg']
        alt_tgt  = fdm_tgt.fdm['position/h-sl-meters']
        
        lat  = fdm.fdm['position/lat-gc-deg']  
        long = fdm.fdm['position/long-gc-deg']
        alt  = fdm.fdm['position/h-sl-meters']
        # distance to north North 
        coords_1 = (lat_tgt, long)
        coords_2 = (lat, long)
        if lat_tgt > lat:
            dist_to_north = geodesic(coords_1, coords_2).meters
        else:
            dist_to_north = -geodesic(coords_1, coords_2).meters    
        # distance east 
        coords_1 = (lat,long_tgt)
        coords_2 = (lat,long)
        if long_tgt > long:
            dist_to_east = geodesic(coords_1, coords_2).meters
        else:
            dist_to_east = -geodesic(coords_1, coords_2).meters    

        diff_alt = alt_tgt - alt
        return dist_to_north, dist_to_east, diff_alt

    def x_rotation(self, vector,theta):
        """Rotates 3-D vector around x-axis"""
        R = np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0, np.sin(theta), np.cos(theta)]])
        return np.dot(R,vector)

    def y_rotation(self, vector,theta):
        """Rotates 3-D vector around y-axis"""
        R = np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta), 0, np.cos(theta)]])
        return np.dot(R,vector)

    def z_rotation(self, vector,theta):
        """Rotates 3-D vector around z-axis"""
        R = np.array([[np.cos(theta), -np.sin(theta),0],[np.sin(theta), np.cos(theta),0],[0,0,1]])
        return np.dot(R,vector)


class CircleClip():
    def __init__(self):
        pass

    def reset(self, ref_init, b = 70):
        
        '''
        Cliping function for circle. If missile is fired in 0 deg direction, operational bound is limited between -b to b 
        etc 290 deg to 70 

        '''
        self.ref_init_op = ref_init + 180
        if self.ref_init_op > 360:
            self.ref_init_op -= 360
            
        if (ref_init - b) > 0 and (ref_init + b < 360):
            self.breakpoint = False
            self.heading_ref_min = ref_init - b
            self.heading_ref_max = ref_init + b
        else:
            self.breakpoint = True
            
            if (ref_init - b) < 0:
                self.heading_ref_min = 360 + (ref_init - b)
                self.heading_ref_max = ref_init + b
            
            elif (ref_init + b) > 360:
                self.heading_ref_min = ref_init - b
                self.heading_ref_max = (ref_init + b) - 360
    
    def clip(self, heading_ref):
        if self.breakpoint:
            if (heading_ref > self.ref_init_op):
                if heading_ref < self.heading_ref_min:
                    return self.heading_ref_min
                else:
                    return heading_ref
            else:
                if heading_ref > self.heading_ref_max:
                    return self.heading_ref_max
                else:
                    return heading_ref
        
        else:
            return np.clip(heading_ref, a_min = self.heading_ref_min, a_max = self.heading_ref_max)