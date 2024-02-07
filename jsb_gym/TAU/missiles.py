import jsbsim
import numpy as np
from jsb_gym.utils.utils import toolkit
from jsb_gym.utils.controllers import PID
import pymap3d as pm

class AIM(object):
    def __init__(self, conf, FlightGear = False, fg_out_directive = None, logs = None):
        # set general parameters related to vizualization and jsbsim
        self.conf = conf
        self.FlightGear = FlightGear
        self.tk = toolkit()
        #self.cp = CircleClip()
        self.fdm = jsbsim.FGFDMExec('.', None)
        self.fdm.load_script(self.conf.general['fdm_xml'])
        self.name = 'None'
        self.state = None 
        self.BT = None  

        self.logs = logs
        
        self.tgt = None
        # send data to flightgear 
        if self.FlightGear:
            self.r_pid = range(self.conf.general['fg_r_pid'])
            self.r_tick = range(self.conf.general['fg_r_tick'])
            self.fg_sleep = self.conf.general['fg_sleep']
        else:
            # 120 -> 1 sec 
            self.r_pid = range(self.conf.general['r_pid'])
            self.r_tick = range(self.conf.general['r_tick'])

        self.load_PID()
        self.load_control_parameters()
        self.load_PN_parameters()
        self.reset_errors()
        self.reset_health()
        
        if self.FlightGear:
            if fg_out_directive == None:
                self.fdm.set_output_directive('data_output/flightgear.xml')
            else:
                self.fdm.set_output_directive(fg_out_directive)

        self.state_at_launch = np.zeros(6)
        
    def cache_initial_state(self, lat0, long0, vel0, alt0):
        # save initial conditions 
        self.lat0 = lat0
        self.long0 = long0
        self.alt0 = alt0
        self.vel0 = vel0
        self.v_e0 = self.get_v_east()
        self.v_n0 = self.get_v_north()
        self.v_d0 = self.get_v_down()
        self.vel0_vec = np.array([ self.v_n0, self.v_e0, self.v_d0])
        
    def reset(self, lat0, long0, alt0, vel0, heading0):
        
        self.reset_errors()

        self.reset_health()

        # input  lat long in deg 
        self.fdm.set_property_value("ic/lat-gc-deg", lat0)
        self.fdm.set_property_value("ic/long-gc-deg", long0)
        # input  altitude in meters 
        self.fdm.set_property_value("ic/h-sl-ft", self.tk.m2f(alt0))
        # input vel in m/s      
        self.fdm['ic/u-fps'] = self.tk.m2f(vel0)
        # input  heading in deg    
        self.fdm['ic/psi-true-rad'] = np.radians(heading0)

        self.fdm.set_property_value('propulsion/set-running', -1)

        # apply and reset jsbsim 
        self.fdm.reset_to_initial_conditions(0)

        self.count = 0

        self.fdm.run()

        self.cache_initial_state(lat0, long0, vel0, alt0)

        self.update_initial_direction()


    def record_logs(self):
        if self.logs != None:
            self.logs.record(TAU= self)

    def reset_errors(self):
        self.value_error = False

    def reset_health(self):
        self.alive = True

    def reset_target(self, tgt, set_active = True):
        self.active = set_active
        self.tgt = tgt
        self.target_lost = False
        self.target_hit = False
        self.position_tgt_NED_norm_old = None
        self.count_lost = 0
        self.position_tgt_NED_norm_min = None

    def set_target_hit(self):
        self.active = False
        self.alive = False
        self.target_lost = False
        self.target_hit = True
        
    def set_target_lost(self):
        self.active = False
        self.alive = False
        self.target_lost = True
        self.target_hit = False

    def is_ready_to_launch(self):
        return not self.active and self.alive and not self.target_lost and not self.target_hit

    def is_tracking_target(self):
        return self.active and self.alive and not self.target_lost and not self.target_hit

    def is_target_hit(self):
        return not self.active and not self.alive and not self.target_lost and self.target_hit

    def is_traget_lost(self):
        return not self.active and not self.alive and self.target_lost and not self.target_hit

    def log_closest_encounter(self):
        if self.position_tgt_NED_norm_min == None:
            self.position_tgt_NED_norm_min = self.position_tgt_NED_norm
        elif self.position_tgt_NED_norm_min > self.position_tgt_NED_norm:
            self.position_tgt_NED_norm_min = self.position_tgt_NED_norm

    def update_initial_direction(self):
        self.theta0 = self.get_theta()
        self.psi0 = self.get_psi()
        self.lat0 = self.get_lat_gc_deg()
        self.long0 = self.get_long_gc_deg()
        self.heading_ref = self.get_psi()
        self.theta_ref = self.get_theta()
        
    def update_relative_target_position_NED(self, scaled = False):
        # The local coordinate origin
        lat0 = self.get_lat_gc_deg() # deg
        lon0 = self.get_long_gc_deg()  # deg
        h0 = self.get_altitude()     # meters

        # The point of interest
        lat = self.tgt.get_lat_gc_deg() # deg
        lon = self.tgt.get_long_gc_deg()  # deg
        h = self.tgt.get_altitude()     # meters

        east, north , up = pm.geodetic2enu(lat, lon, h, lat0, lon0, h0)
        down = -up
        self.d_tgt_east = east
        self.d_tgt_north = north
        self.d_tgt_down = down
        try:
            #self.position_tgt_NED_norm_old = self.position_tgt_NED_norm
            self.position_tgt_NED_norm = round(np.linalg.norm(np.array([east, north, down])))
        except ValueError:          
            print('position_tgt_NED_norm value error')
            print(east, north, down)
            print(lat, lon, h, lat0, lon0, h0, self.position_tgt_NED_norm_old)
            self.position_tgt_NED_norm = self.position_tgt_NED_norm_old
            print('Sim, time: ', self.get_sim_time_sec())
            self.target_lost = True
            self.value_error = True

        if scaled:
            north = self.tk.scale_between( north, a_min= 0.0, a_max=self.conf.PN['rage_NE'])
            east = self.tk.scale_between( east, a_min= 0.0, a_max=self.conf.PN['rage_NE'])
            down = self.tk.scale_between( down, a_min= 0.0, a_max=self.conf.PN['rage_D'])
        
        self.tgt_east = east
        self.tgt_north= north
        self.tgt_down = down 

    def load_PID(self):
        P = self.conf.pid_roll['P']
        I = self.conf.pid_roll['I']
        D = self.conf.pid_roll['D']
        I_min = self.conf.pid_roll['I_min']
        I_max = self.conf.pid_roll['I_max']
        self.pid_roll    = PID(P=P, I=I, D= D, Derivator=0, Integrator=0, Integrator_max=I_max, Integrator_min=I_min)
        P = self.conf.pid_pitch['P']
        I = self.conf.pid_pitch['I']
        D = self.conf.pid_pitch['D']      
        I_min = self.conf.pid_pitch['I_min']
        I_max = self.conf.pid_pitch['I_max']

        self.pid_pitch   = PID(P=P, I=I, D=D, Derivator=0, Integrator=0, Integrator_max=I_max, Integrator_min=I_min)
        P = self.conf.pid_heading['P']
        I = self.conf.pid_heading['I']
        D = self.conf.pid_heading['D']
        I_min = self.conf.pid_heading['I_min']
        I_max = self.conf.pid_heading['I_max']
        self.pid_heading = PID(P=P, I=I, D=D, Derivator=0, Integrator=0, Integrator_max=I_max, Integrator_min=I_min)        

    def load_control_parameters(self):
        self.diff_heading = np.array([None, None, None])
        self.diff_heading_abs = np.array([None, None, None])
        
    def load_PN_parameters(self):

        # targets flight dynamics model 
        self.tgt = None
        # position 
        self.lat_tgt = None
        self.long_tgt = None 
        self.alt_tgt = None

        self.dist_to_tgt = None
        self.dist_to_tgt_old = None
        self.altitude_to_tgt = None
        self.altitude_to_tgt_old = None

        self.sim_time_old = None
        
        self.position_tgt_NED = np.zeros(3)
        self.velocity_tgt_NED = np.zeros(3)
        self.velocity_NED = np.zeros(3)

        self.velocity_relative_NED = np.zeros(3)
        self.rotation_vector = np.zeros(3)
        self.acceleration_cmd_NED = np.zeros(3)

        self.velocity_NED_PN = np.zeros(3)

        self.velocity_NE = np.zeros(3)
        self.velocity_NE_PN = np.zeros(3)

        self.velocity_HD = np.zeros(3)
        self.velocity_HD_PN = np.zeros(3)

        self.velocity_SD = np.zeros(3)
        self.velocity_SD_PN = np.zeros(3)

    def get_lat_gc_deg(self, val0 = False):
        if val0:
            return 
        else:
            return self.fdm['position/lat-gc-deg']

    def get_long_gc_deg(self):
        return self.fdm['position/long-gc-deg']

    def get_altitude(self, scaled = False):
        alt = self.fdm['position/h-sl-meters']
        if scaled:
            alt = self.tk.scale_between(alt, a_min = self.conf.ctrl['alt_min'], a_max = self.conf.ctrl['alt_max'])
        return alt

    def get_v_east(self):
        # returns vector component for velocity in east direction in meters per second
        return self.tk.f2m(self.fdm['velocities/v-east-fps'])

    def get_v_north(self):
        return self.tk.f2m(self.fdm['velocities/v-north-fps'])

    def get_v_down(self):
        return self.tk.f2m(self.fdm['velocities/v-down-fps'])

    def get_sim_time_sec(self):
        return self.fdm['simulation/sim-time-sec']

    def get_Mach(self, scaled = False):
        vel = self.fdm['velocities/mach']
        if scaled:
            vel = self.tk.scale_between(vel, a_min = self.vel_mach_min, a_max = self.vel_mach_max)
        return vel

    def get_phi(self, scaled = False, in_deg = True):
        if in_deg:
            phi =  self.fdm['attitude/phi-deg']
            if scaled:
                phi = self.tk.scale_between(phi, a_min = self.conf.ctrl['phi_min'], a_max = self.conf.ctrl['phi_max'])
        else:
            phi =  self.fdm['attitude/phi-rad']
            if scaled:
                phi = self.tk.scale_between(phi, a_min = np.radians(self.conf.ctrl['phi_min']), a_max = np.radians(self.conf.ctrl['phi_max']))
        return phi

    def get_theta(self, scaled = False, in_deg = True):
        if in_deg:
            theta =  self.fdm['attitude/theta-deg']
            if scaled:
                theta = self.tk.scale_between(theta, a_min = self.conf.ctrl['theta_min'], a_max = self.conf.ctrl['theta_max'])
        else:
            theta =  self.fdm['attitude/theta-rad']
            if scaled:
                theta = self.tk.scale_between(theta, a_min = np.radians(self.conf.ctrl['theta_min']), a_max = np.radians(self.conf.ctrl['theta_max']))
        return theta
    
    def get_true_airspeed(self):
        return self.tk.f2m(self.fdm['velocities/vt-fps'])

    def get_psi(self, scaled = False, in_deg = True):
        if in_deg:
            psi =  self.fdm['attitude/psi-deg']
            if scaled:
                psi = self.tk.scale_between(psi, a_min = self.conf.ctrl['psi_min'], a_max = self.conf.ctrl['psi_max'])
        else:
            psi =  self.fdm['attitude/psi-rad']
            if scaled:
                psi = self.tk.scale_between(psi, a_min = np.radians(self.conf.ctrl['psi_min']), a_max = np.radians(self.conf.ctrl['psi_max']))
        return psi

    def get_roll_delta(self, ref):
        roll_delta = ref - self.get_phi()
        if roll_delta > 180.0:
            roll_delta -= 360.0
        elif roll_delta <= -180.0:
            roll_delta += 360.0
        return roll_delta

    def get_heading_difference(self, reference):
        #psi = np.degrees(self.fdm['attitude/heading-true-rad'])     
        psi = self.get_psi()   
        diff_cw = reference - psi
        diff_ccw = reference - (psi+ 360)
        diff_ccw_r = reference + 360 - psi
        self.diff_heading[0] = diff_cw
        self.diff_heading[1] = diff_ccw
        self.diff_heading[2] = diff_ccw_r
        self.diff_heading_abs[0] = abs(diff_cw)
        self.diff_heading_abs[1] = abs(diff_ccw)
        self.diff_heading_abs[2] = abs(diff_ccw_r)

        return self.diff_heading[np.argmin(self.diff_heading_abs)]

    def set_roll_PID(self, ref):
        
        self.roll_ref = ref    
        # get reference value for control input 
        diff = self.get_roll_delta(ref)
        #print('roll : ', self.roll_ref, round(self.get_phi()), round(diff))
        # set aileron
        cmd = self.pid_roll.update(current_value=diff)
        cmd =  np.clip(cmd, -self.conf.ctrl['aileron_clip'], self.conf.ctrl['aileron_clip'])
        self.fdm['fcs/aileron-cmd-norm'] = - cmd 

    def set_pitch_PID(self, ref):
        self.theta_ref = ref
        diff = self.theta_ref - self.get_theta()
        cmd = self.pid_pitch.update(current_value= diff)
        cmd =  np.clip(cmd, -self.conf.ctrl['elevator_clip'], self.conf.ctrl['elevator_clip'])
        self.fdm['fcs/elevator-cmd-norm'] = -cmd

    def set_yaw_PID(self, ref):
        #self.psi_ref = self.cp.clip(ref)
        self.psi_ref = ref
        diff = self.get_heading_difference(self.psi_ref)
        cmd = self.pid_heading.update(current_value= diff)
        self.fdm['fcs/rudder-cmd-norm'] = np.clip(cmd, -self.conf.ctrl['rudder_clip'], self.conf.ctrl['rudder_clip'] )
    
    def set_throttle(self,cmd = 0.7):
        self.fdm['fcs/throttle-cmd-norm[0]'] = cmd

    def set_altitude_PID(self, ref, theta_min, theta_max):
        # angle between different altitudes 
        diff_atl = ref - self.get_altitude()
        self.theta_ref = np.degrees(np.arctan2(diff_atl, self.conf.PN['tan_ref']))
        self.theta_ref = np.clip(a = self.theta_ref, a_min = theta_min, a_max = theta_max)
        self.set_pitch_PID(self.theta_ref)
        
    def PN(self):
        # set heading and pitch 
        self.position_tgt_NED[0] = self.tgt_east
        self.position_tgt_NED[1] = self.tgt_north
        self.position_tgt_NED[2] = self.tgt_down

        self.velocity_tgt_NED[0] = self.tgt.get_v_east()
        self.velocity_tgt_NED[1] = self.tgt.get_v_north()
        self.velocity_tgt_NED[2] = self.tgt.get_v_down()

        #self.position_tgt_NED = self.position_tgt_NED + self.velocity_tgt_NED * self.dt_PN

        self.velocity_NED[0] = self.get_v_east()
        self.velocity_NED[1] = self.get_v_north()
        self.velocity_NED[2] = self.get_v_down()
        
        self.velocity_relative_NED = self.velocity_tgt_NED - self.velocity_NED

        self.rotation_vector = (np.cross(self.position_tgt_NED,self.velocity_relative_NED)) / (self.position_tgt_NED @ self.position_tgt_NED)
        
        self.acceleration_cmd_NED = self.conf.PN['N'] * np.cross(self.velocity_relative_NED , self.rotation_vector)

        self.velocity_NED_PN = self.velocity_NED + self.acceleration_cmd_NED*self.conf.PN['dt']

        # get heading 
        self.velocity_NE[0] = self.velocity_NED[0].copy()
        self.velocity_NE[1] = self.velocity_NED[1].copy()

        self.velocity_NE_PN[0] = self.velocity_NED_PN[0].copy()
        self.velocity_NE_PN[1] = self.velocity_NED_PN[1].copy()

        heading = self.tk.angle_between(v1=self.velocity_NE, v2= self.velocity_NE_PN, in_deg= True)

        if np.cross(self.velocity_NE, self.velocity_NE_PN)[2] < 0:
            self.heading_ref = self.heading_ref + heading
        else:
            self.heading_ref = self.heading_ref - heading
        
        self.heading_ref = self.tk.truncate_heading(self.heading_ref)

        self.theta_ref = -np.clip(a=self.acceleration_cmd_NED[2], a_min = self.conf.PN['theta_min'], a_max = self.conf.PN['theta_max'])
        
        self.velocity_relative_NED_norm = np.linalg.norm(self.velocity_relative_NED)
        self.time_to_impact = self.position_tgt_NED_norm/self.velocity_relative_NED_norm
        self.altitude_ref = self.tgt.get_altitude() - self.velocity_tgt_NED[2]*self.time_to_impact

    def is_mach_low(self):
        return self.get_Mach() < self.conf.PN['target_lost_below_mach'] and self.acceleration_stage_done()

    def is_alt_low(self):
        return self.get_altitude() < self.conf.PN['target_lost_below_alt']

    def is_target_lost(self):
        # check if missile missed         
        if (self.position_tgt_NED_norm_old != None) \
            and (self.position_tgt_NED_norm_old < self.position_tgt_NED_norm) \
            and (self.position_tgt_NED_norm < 3e3):
            self.count_lost += 1
            if self.count_lost > self.conf.PN['count_lost']:
                #self.target_lost = True
                return True
            else:
                return False
        else:
            self.count_lost = 0
            return False

    def is_within_warhead_range(self):
        return self.position_tgt_NED_norm <= self.conf.PN['warhead']

    def is_alive(self):
        return self.alive

    def is_done(self):
        # check is missile too slow 
        if self.is_mach_low():
            print('Lost: Mach low')
            self.set_target_lost()

        if self.is_alt_low():
            print('Hit ground')
            self.set_target_lost()

        if self.is_target_lost():
            print('Lost target')
            self.set_target_lost()

        if self.is_within_warhead_range():
            self.set_target_hit()
        
        self.position_tgt_NED_norm_old = self.position_tgt_NED_norm
        
    def step_evasive(self):
        #
        for _ in self.r_tick:
            #print(round(self.get_sim_time_sec()), round(self.get_Mach(), 2))
            self.step_PN()
            self.is_done()
        self.count += 1

    def step_PN(self):
        
        self.update_relative_target_position_NED()
        self.log_closest_encounter()
        self.PN()
        for _ in self.r_pid:
            self.step_PID()

    def acceleration_stage_done(self):
        if self.get_sim_time_sec() < self.conf.PN['steady_flight_sec']:
            # still accelerating
            return False
        else:
            return True

    def step_PID(self):
        # steady flight before control
        if not self.acceleration_stage_done():
            #print('Acc')
            self.set_throttle()
            self.set_roll_PID(ref= 0.0)
            self.set_pitch_PID(ref= self.theta0)
            self.set_yaw_PID(ref= self.psi0)     
        else:
        # turn off engine and glide    
            #print('PN')
            self.set_throttle(cmd = 0.0)
            self.set_roll_PID(ref= 0.0)
            self.set_yaw_PID(ref= self.heading_ref)
            
            if self.position_tgt_NED_norm > self.conf.PN['dive_at']:
                self.altitude_ref = self.conf.PN['alt_cruise']
                self.set_altitude_PID(ref= self.altitude_ref, theta_min= self.conf.PN['theta_min_cruise'], theta_max=self.conf.PN['theta_max_cruise'])
            else:
                self.set_pitch_PID(self.theta_ref)    
            
        #if self.conf.general['rec']:
        #    self.logs.record(aim= self)
        self.record_logs()
         
        self.fdm.run()
