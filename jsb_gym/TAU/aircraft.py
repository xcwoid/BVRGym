import jsbsim
from jsb_gym.utils.utils import toolkit
from jsb_gym.utils.controllers import PID
import time
import numpy as np

class F16(object):
    def __init__(self, conf, FlightGear = False, fg_out_directive = None, logs = None):
        self.conf = conf
        self.FlightGear = FlightGear
        self.tk = toolkit()
        self.fdm = jsbsim.FGFDMExec('.', None)
        self.fdm.load_script(self.conf.general['fdm_xml'])    
        self.name = 'None'
        self.state = None
        self.state_block = {}
        self.BT = None  

        self.logs = logs
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
        
        if self.FlightGear:
            if fg_out_directive == None:
                self.fdm.set_output_directive('data_output/flightgear.xml')
            else:
                self.fdm.set_output_directive(fg_out_directive)
        
    def load_PID(self):
        P = self.conf.pid_roll['P']
        I = self.conf.pid_roll['I']
        D = self.conf.pid_roll['D']
        self.pid_roll    = PID(P=P, I=I, D= D, Derivator=0, Integrator=0, Integrator_max=1, Integrator_min=-1)

        P = self.conf.pid_roll_sec['P']
        I = self.conf.pid_roll_sec['I']
        D = self.conf.pid_roll_sec['D']
        self.pid_roll_sec    = PID(P=P, I=I, D= D, Derivator=0, Integrator=0, Integrator_max=1, Integrator_min=-1)

        P = self.conf.pid_pitch['P']
        I = self.conf.pid_pitch['I']
        D = self.conf.pid_pitch['D']      
        self.pid_pitch   = PID(P=P, I=I, D=D, Derivator=0, Integrator=0, Integrator_max=1, Integrator_min=-1)
        
        #P = self.conf.pid_heading['P']
        #I = self.conf.pid_heading['I']
        #D = self.conf.pid_heading['D']      
        #self.pid_heading = PID(P=P, I=I, D=D, Derivator=0, Integrator=0, Integrator_max=10, Integrator_min=-10)        

        P = self.conf.pid_rudder_theta['P']
        I = self.conf.pid_rudder_theta['I']
        D = self.conf.pid_rudder_theta['D']      
        self.pid_rudder_theta = PID(P=P, I=I, D=D, Derivator=0, Integrator=0, Integrator_max=10, Integrator_min=-10)        

        P = self.conf.pid_elevator_psi['P']
        I = self.conf.pid_elevator_psi['I']
        D = self.conf.pid_elevator_psi['D']      
        self.pid_elevator_psi = PID(P=P, I=I, D=D, Derivator=0, Integrator=0, Integrator_max=10, Integrator_min=-10)        

    def reset(self, lat, long, alt, vel, heading):
        

        # input  lat long in deg 
        self.fdm.set_property_value("ic/lat-gc-deg", lat)
        self.fdm.set_property_value("ic/long-gc-deg", long)
        # input  altitude in meters 
        self.fdm.set_property_value("ic/h-sl-ft", self.tk.m2f(alt))
        # input vel in m/s      
        self.fdm['ic/u-fps'] = self.tk.m2f(vel)
        # input  heading in deg    
        self.fdm['ic/psi-true-rad'] = np.radians(heading)

        self.fdm.set_property_value('propulsion/set-running', -1)

        self.fdm.reset_to_initial_conditions(0)
        
        self.fdm.run()

        self.reset_ctrl()

        self.count = 0
    
    def record_logs(self):
        if self.logs != None:
            self.logs.record(TAU= self)

    def reset_ctrl(self):
        self.diff_heading = np.array([None, None, None])
        self.diff_heading_abs = np.array([None, None, None])

        self.heading_clip = self.conf.ctrl['heading_clip']
        self.alt_ref = self.get_altitude()
        self.theta_ref = self.get_theta()
        self.psi_ref = self.get_psi()
        self.rudder_cmd = 0.0
        self.elevator_cmd = 0.0
        self.aileron_cmd = 0.0
        self.crank_left = self.conf.crank['left']
        self.crank_time = self.conf.crank['time']
        
    def get_lat_gc_deg(self):
        return self.fdm['position/lat-gc-deg']

    def get_long_gc_deg(self):
        return self.fdm['position/long-gc-deg']

    def get_sim_time_sec(self):
        return self.fdm['simulation/sim-time-sec']

    def get_Mach(self, scaled = False):
        vel = self.fdm['velocities/mach']
        if scaled:
            vel = self.tk.scale_between(vel, a_min = self.conf.ctrl['vel_mach_min'], a_max = self.conf.ctrl['vel_mach_max'])
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

    def get_true_airspeed(self):
        return self.tk.f2m(self.fdm['velocities/vt-fps'])

    def get_u(self):
        return self.tk.f2m(self.fdm['u-fps'])
    
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

    def get_v_down(self, scaled = False):
        v_down = self.tk.f2m(self.fdm['velocities/v-down-fps'])
        if scaled:
            v_down = self.tk.scale_between(scaled, a_min = self.conf.ctrl['v_down_min'], a_max = self.conf.ctrl['v_down_max'])
        return v_down

    def set_gear(self, cmd = 0):
        self.fdm['gear/gear-pos-norm'] = cmd

    def set_aileron(self, cmd):
        # cmd range -1 to 1 
        self.fdm['fcs/aileron-cmd-norm'] = cmd

    def set_elevator(self, cmd):
        # cmd range -1 to 1 
        self.fdm['fcs/elevator-cmd-norm'] = cmd

    def set_rudder(self, cmd):
        # cmd range -1 to 1 
        self.fdm['fcs/rudder-cmd-norm'] = cmd

    def set_throttle(self, cmd = 0.49):
        # cmd 0.49 (no afterburner)
        # cmd 0.7  (active afterburner)
        self.fdm['fcs/throttle-cmd-norm'] = np.clip(cmd, self.conf.ctrl['thr_min'], self.conf.ctrl['thr_max'])

    def get_roll_delta(self, ref):
        roll_delta = ref - self.fdm['attitude/phi-deg']
        if roll_delta > 180.0:
            roll_delta -= 360.0
        elif roll_delta <= -180.0:
            roll_delta += 360.0
        return roll_delta

    def get_heading_difference(self, reference):
        diff_cw = reference - self.fdm['attitude/psi-deg']
        diff_ccw = reference - (self.fdm['attitude/psi-deg']+ 360)
        diff_ccw_r = reference + 360 - self.fdm['attitude/psi-deg']
        self.diff_heading[0] = diff_cw
        self.diff_heading[1] = diff_ccw
        self.diff_heading[2] = diff_ccw_r
        self.diff_heading_abs[0] = abs(diff_cw)
        self.diff_heading_abs[1] = abs(diff_ccw)
        self.diff_heading_abs[2] = abs(diff_ccw_r)

        return self.diff_heading[np.argmin(self.diff_heading_abs)]

    def set_altitude_PID(self, ref_altitude):

        self.alt_ref = ref_altitude
        diff_atl = ref_altitude - self.get_altitude()
        theta_ref = np.degrees(np.arctan2(diff_atl, self.conf.ctrl['tan_ref']))
        theta_ref = np.clip(a= theta_ref, a_min = -60, a_max = 60)
        self.set_pitch_PID(theta_ref)
        self.set_roll_PID(roll_ref= 0.0)

    def set_roll_PID(self, roll_ref, secondary_pid = False):
        # make sure within range
        roll_ref = np.clip(roll_ref, self.conf.ctrl['phi_min'], self.conf.ctrl['phi_max'])
        
        # save for recording and other stuff
        self.roll_ref = roll_ref
        
        # get reference value for control input 
        diff = self.get_roll_delta(roll_ref)
        
        if secondary_pid:
           cmd = -self.pid_roll_sec.update(current_value=diff)
        else:
           cmd = -self.pid_roll.update(current_value=diff)
        cmd = np.clip(a = cmd, a_min = -1, a_max= 1)
        self.aileron_cmd = cmd
        self.set_aileron(cmd)
        #self.fdm['fcs/aileron-cmd-norm'] = cmd
        
    def set_pitch_PID(self, theta_ref):
        self.theta_ref = np.clip(theta_ref, self.conf.ctrl['theta_min'], self.conf.ctrl['theta_max'])
        diff = self.theta_ref - self.get_theta()
        cmd = self.pid_pitch.update(current_value= diff)
        cmd = np.clip(a = cmd, a_min = -1, a_max= 1)
        self.elevator_cmd = cmd
        self.fdm['fcs/elevator-cmd-norm'] = cmd

    def set_rudder_PID(self, theta_ref = None, psi_diff = None):
        if theta_ref != None:
            diff = theta_ref - self.get_theta()
            cmd = self.pid_rudder_theta.update(diff)
            self.rudder_cmd = cmd
            self.fdm['fcs/rudder-cmd-norm'] = cmd

        if psi_diff != None:
            cmd = self.pid_rudder_theta.update(psi_diff)
            self.rudder_cmd = cmd
            self.fdm['fcs/rudder-cmd-norm'] = cmd
    
    def get_total_fuel(self):
        return self.tk.lbs2kg(self.fdm['propulsion/total-fuel-lbs'])

    def set_elevator_PID(self, psi_diff = None):
        if psi_diff != None:
            cmd = self.pid_elevator_psi.update(psi_diff)
            cmd = np.clip(a = cmd, a_min = -1, a_max= 1)
            self.elevator_cmd = cmd
            self.fdm['fcs/elevator-cmd-norm'] = -cmd

    def set_heading_PID(self, ref_heading, dive_alt):
        # future note:
        # probably would be nice to actually implement a smoother multivariable controller.
        # or maybe a master student can waste time on it 
        # if you are the lucky master student working on it, thank you :)  
        self.psi_ref = ref_heading
        diff = self.get_heading_difference(ref_heading)
        diff_alt = dive_alt- self.get_altitude()
        #print(self.conf.ctrl['alt_act_space'])
        # if the altitude is ok, but not the heading 
        if abs(diff) >= self.heading_clip and abs(diff_alt) <= self.conf.ctrl['alt_act_space']:
            # theta is not tracked
            self.theta_ref = self.get_theta()
            # create larger altitude action space
            self.conf.ctrl['alt_act_space'] = self.conf.ctrl['alt_act_space_max']
            # decide which direction to roll 
            roll_rot_dir = 1 if diff >= 0 else -1
            self.set_roll_PID(roll_ref= roll_rot_dir * (self.conf.set_heading_PID['roll_max']))
            # if roll is close to the limit 
            if self.conf.set_heading_PID['roll_max'] - abs(self.get_phi()) < 30:
                #print('close')
                self.fdm['fcs/elevator-cmd-norm'] = -0.3
            else:
                self.fdm['fcs/elevator-cmd-norm'] = -0.9

            self.heading_clip = 10

        else:
            # decrease margine adjust heading a bit 
            self.conf.ctrl['alt_act_space'] = self.conf.ctrl['alt_act_space_min']
            self.alt_ref = dive_alt
            diff_atl = dive_alt - self.get_altitude()
            theta_ref = np.degrees(np.arctan2(diff_atl, self.conf.ctrl['tan_ref']))
            theta_ref = np.clip(a= theta_ref, a_min = -45, a_max = 45)
            if abs(diff_atl) > 1.5e3:
                #print('alt 1')
                self.conf.ctrl['theta_act_space'] = self.conf.ctrl['theta_act_space_max']
                self.set_roll_PID(roll_ref= 0.0)
            elif abs(diff_atl) < 1.5e3 and abs(self.get_theta()) > self.conf.ctrl['theta_act_space']:
                #print('alt 2')
                self.conf.ctrl['theta_act_space'] = self.conf.ctrl['theta_act_space_min']
                self.set_roll_PID(roll_ref= 0.0)
            else:
                #print('alt 3')
                self.conf.ctrl['theta_act_space'] = self.conf.ctrl['theta_act_space_max']
                diff = self.get_heading_difference(ref_heading)
                #print(diff)
                roll_rot_dir = 1 if diff >= 0 else -1
                
                if abs(diff) < 10:
                    #print('diff < 10', abs(diff)*0.01)
                    self.set_roll_PID(roll_ref= roll_rot_dir * abs(diff), secondary_pid=True)
                else:
                    self.set_roll_PID(roll_ref= roll_rot_dir * abs(diff)*3)
            
            self.set_pitch_PID(theta_ref= theta_ref)
            self.heading_clip = 35
            self.fdm['fcs/rudder-cmd-norm'] = 0

    def step_BVR(self, action, action_type):
        for _ in self.r_tick:
            self.set_gear()
            if action_type == 0:
                self.cmd_BVR(action)
            elif action_type == 1:
                self.cmd_crank(action)
            else:
                print('Action type does not exist.')
                exit()
        self.count += 1

    def cmd_BVR(self, action):
        # create a step that uses input array [heading, altitude, thrust]'
        # 
        # set some constant in the main file only 
        # heading altitude 
        # set throthle at a lower frequency 
        #print(action[0], action[1], action[2])

        act_head = self.tk.scale_between_inv(a = action[0], a_min=self.conf.ctrl['psi_min'], a_max=self.conf.ctrl['psi_max'])
        act_alt  = self.tk.scale_between_inv(a = action[1], a_min=self.conf.ctrl['alt_min'], a_max=self.conf.ctrl['alt_max'])
        act_thr  = self.tk.scale_between_inv(a = action[2], a_min=self.conf.ctrl['thr_min'], a_max=self.conf.ctrl['thr_max'])
        #print(self.name, act_head, act_alt, act_thr)
        #print(act_thr, round(self.get_Mach(), 2))
        self.set_throttle(act_thr)
        for _ in self.r_pid:
            if self.FlightGear:
                time.sleep(self.fg_sleep)
            self.set_heading_PID(ref_heading = act_head, dive_alt= act_alt)
            #
            self.record_logs()
            self.fdm.run()

    def cmd_crank(self, action):
        
        act_thr  = self.tk.scale_between_inv(a = action[2], a_min=self.conf.ctrl['thr_min'], a_max=self.conf.ctrl['thr_max'])
        self.set_throttle(act_thr)
        
        act_head = self.tk.scale_between_inv(a = action[0], a_min=self.conf.ctrl['psi_min'], a_max=self.conf.ctrl['psi_max'])
        act_alt  = self.tk.scale_between_inv(a = action[1], a_min=self.conf.ctrl['alt_min'], a_max=self.conf.ctrl['alt_max'])

        # utilities, truncate_heading 
        if self.crank_left:
            act_head -= self.conf.crank['angle']
        else:
            act_head += self.conf.crank['angle']


        t0 = self.get_sim_time_sec()
        for _ in self.r_pid:
            if self.FlightGear:
                time.sleep(self.fg_sleep)
                
            self.set_heading_PID(ref_heading = act_head, dive_alt= act_alt)
            self.record_logs()
            self.fdm.run()
        
        dt = self.get_sim_time_sec() - t0
        self.crank_time += dt

        if self.crank_time >= self.conf.crank['time_max']:
            self.crank_time = 0
            if self.crank_left:
                self.crank_left = False
            else:
                self.crank_left = True

    def step_cruise(self, action):
        # heading altitude 
        for _ in self.r_pid:
            if self.FlightGear:
                time.sleep(self.fg_sleep)
                
            if action[1] >= self.conf.act['split_at']:
                act = self.tk.scale_between_inv(a = action[0], a_min=self.conf.ctrl['psi_min'], a_max=self.conf.ctrl['psi_max'])
                self.set_heading_PID(ref_heading = act)
            
            else:
                act = self.tk.scale_between_inv(a = action[0], a_min=self.conf.ctrl['alt_min'], a_max=self.conf.ctrl['alt_max'])
                self.set_altitude_PID(ref_altitude = act)
            
            self.record_logs()
            self.fdm.run()

    def step_cruise1(self, action):
        # heading altitude 
        for _ in self.r_pid:
            if self.FlightGear:
                time.sleep(self.fg_sleep)
                
            act = self.tk.scale_between_inv(a = action[0], a_min=self.conf.ctrl['psi_min'], a_max=self.conf.ctrl['psi_max'])
            #print('act:', act)
            self.set_heading_PID(ref_heading = act, dive_alt= 6e3)
            
            self.record_logs()
            self.fdm.run()

    def step_evasive1(self, action, throttle):
        for _ in self.r_tick:
            self.set_gear()         
            self.set_throttle(throttle)
            self.step_cruise1(action)
            # is_done()
        self.count += 1

    def step_evasive1(self, action, throttle):
        for _ in self.r_tick:
            self.set_gear()         
            self.set_throttle(throttle)
            self.step_cruise1(action)
            # is_done()
        self.count += 1

    def step_evasive(self, action, throttle):
        for _ in self.r_tick:
            self.set_gear()         
            self.set_throttle(throttle)
            self.step_cruise(action)
            # is_done()
        self.count += 1

    def step_dogFight(self, action, randy = False):
        
        for _ in self.r_tick:
            self.step_BT(action, randy)
        self.count += 1

    def step_BT(self, action, randy):
        
        self.set_gear()
        
        self.BT.tick()
        
        if self.BT.BTState == 'Alt_act':
            self.step_pull_up()   
        else:
            if randy:
                self.step_randy(throttle= 0.35)
            else:
                self.step_PID(action)

    def step_pull_up(self):
        self.set_throttle(cmd = 0.49)
            
        for _ in self.r_pid:
            if self.FlightGear:
                time.sleep(self.fg_sleep)
                print('Time Delay')

            if self.count %2 ==0:
                self.set_altitude_PID(ref_altitude = 3000)
            else:
                self.set_roll_PID(roll_ref=0.0)
            self.fdm.run()

    def step_PID(self, action, throttle_cmd = 0.49, relative_heading = True):
        
        self.set_throttle(throttle_cmd)
        
        for _ in self.r_pid:
            if self.FlightGear:
                time.sleep(self.fg_sleep)     
            if action[1] >= 0.0:
                # set heading
                if relative_heading:
                    ref_heading = self.tk.scale_between_inv(a = action[0], a_min= -179, a_max=179)
                    self.psi_ref = self.tk.truncate_heading( self.get_psi() + ref_heading)                
                else:
                    self.psi_ref = self.tk.scale_between_inv(a = action[0], a_min=self.conf.ctrl['psi_min'], a_max=self.conf.ctrl['psi_max'])
                
                self.set_heading_PID(ref_heading= self.psi_ref)
                
            else:
                theta_ref = self.tk.scale_between_inv(a = action[0], a_min=self.conf.ctrl['theta_min'], a_max=self.conf.ctrl['theta_max'])
                head_diff = self.get_heading_difference(self.psi_ref)
                self.set_pitch_PID(theta_ref)
                self.set_roll_PID(roll_ref=0.0) 
                self.set_rudder_PID(ref_heading = head_diff)
                   
            self.fdm.run()

    def step_randy(self, throttle = 0.49):
         
        self.set_throttle(throttle)
        for _ in self.r_pid:
            if self.FlightGear:
                time.sleep(self.fg_sleep)

            if self.randy_action_slack >= 0.0:
                act = self.tk.scale_between_inv(a = self.randy_action, a_min=self.conf.ctrl['psi_min'], a_max=self.conf.ctrl['psi_max'])
                self.set_heading_PID(ref_heading = act)
            else:
                act = self.tk.scale_between_inv(a = self.randy_action, a_min=self.conf.randy['alt_min'], a_max=self.conf.randy['alt_max'])
                self.set_altitude_PID(ref_altitude = act)
            self.fdm.run()

    def update_randy_action(self):
        self.randy_action_slack = np.random.uniform( self.conf.randy['slack_min'], self.conf.randy['slack_max'])
        self.randy_action = np.random.uniform( self.conf.randy['act_min'] , self.conf.randy['act_max'])

    def step_AERT(self, aileron_cmd, elevator_cmd, rudder_cmd, throttle_cmd):
        '''All inputs are between -1 to 1 '''
        thr_cmd = self.tk.scale_between_inv(a= throttle_cmd, a_min= self.conf.ctrl['thr_min'], a_max= self.conf.ctrl['thr_max'])
        for _ in self.r10:

            self.set_aileron(aileron_cmd)
            self.set_elevator(elevator_cmd)
            self.set_rudder(rudder_cmd)
            self.set_throttle(thr_cmd)
            self.fdm.run()

        self.count += 1
