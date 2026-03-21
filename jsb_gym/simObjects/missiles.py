import numpy as np
from jsb_gym.simObjects.FDMObject import FDMObject
from jsb_gym.utils.guidance_laws import PN
import time
from jsb_gym.utils.autopilot import MissilePIDAutopilot
from jsb_gym.utils.geospatial import dinstance_between_simObj_agent

class AAMBVR(FDMObject):
    def __init__(self, conf):
        super().__init__(conf)
        self.conf = conf
        self.missile_control = MissilePIDAutopilot(self)
        self.PN = PN(conf)
        self.missile_simulation_config = self.conf.missile_simulation
        self.missile_performance = self.conf.missile_performance
        self.reset_operational_state()

    def reset_operational_state(self):
        self.target = None

        self.target_lost = False
        self.target_hit = False
        self.active = False
        self.alive = True
        
        self.lost_count = 0
        self.target_norm = None
        self.target_norm_old = None


    def reset(self, lat, long, alt, vel, heading):
        super().reset(lat, long, alt, vel, heading)
        

    def is_alive(self):
        return self.alive

    def is_ready_to_launch(self):
        return not self.active and self.alive and not self.target_lost and not self.target_hit

    def is_active(self):
        return self.active and self.alive and not self.target_lost and not self.target_hit

    def is_target_hit(self):
        return not self.active and not self.alive and not self.target_lost and self.target_hit

    def is_traget_lost(self):
        return not self.active and not self.alive and self.target_lost and not self.target_hit


    def set_target(self, target):
        ''' 
        Set target aircraft object 
        Input: simObject
        '''
        self.target = target

    def launch(self, carrier_aircraft):
        if self.is_alive():
            self.active = True
            lat = carrier_aircraft.simObj.get_lat_gc_deg()
            long = carrier_aircraft.simObj.get_long_gc_deg()
            alt = carrier_aircraft.simObj.get_altitude()
            vel = carrier_aircraft.simObj.get_mach()
            heading = carrier_aircraft.simObj.get_psi()

            self.reset(lat, long, alt, vel, heading)


    def is_target_within_effective_range(self):
        
        if self.target_norm <= self.missile_performance.effective_radius:
            self.target_hit = True
            self.active = False
            self.alive = False
            
            # set hit to the target
            self.target.healthPoints -= 1
            return True
        return False

    def is_mach_low(self):
        if self.missile_control.acceleration_stage_done() and self.get_mach() < self.missile_performance.target_lost_below_mach:
            self.target_lost = True
            self.target_hit = False
            self.active = False
            self.alive = False
            return True
        return False

    def is_alt_low(self):
        if self.get_altitude() < self.missile_performance.target_lost_below_alt:
            self.target_lost = True
            self.target_hit = False
            self.active = False
            self.alive = False
            return True
        return False

    def is_target_lost(self):
        # check if missile missed         
        if self.target_norm_old > self.target_norm:
            self.lost_count += 1
            if self.lost_count > self.missile_performance.lost_count:
                self.target_lost = True
                self.target_hit = False
                self.active = False
                self.alive = False
                return True
            else:
                return False
        else:
            self.count_lost = 0
            return False

    def step(self):   
        
        self.target_norm_old = dinstance_between_simObj_agent(self, self.target)     
        
        for _ in range(self.missile_simulation_config.Sim_time_step):
            self.target_norm = dinstance_between_simObj_agent(self, self.target)
            self.is_mach_low()
            self.is_alt_low()
            #self.is_target_lost()
            self.is_target_within_effective_range()

            self._step()

    def _step(self):

        heading_cmd, altitude_cmd = self.PN.get_guidance(self)
        
        if self.target_norm > self.conf.missile_navigation.dive_at:
            altitude_cmd = self.conf.missile_navigation.alt_cruise
            self.conf.missile_navigation.tan_ref = self.conf.missile_navigation.tan_ref_max
        else:
            self.conf.missile_navigation.tan_ref = self.conf.missile_navigation.tan_ref_min
        
        for _ in range(self.missile_simulation_config.Control_time_step):
                
            aileron_cmd, elevator_cmd, rudder_cmd, throttle_cmd	= self.missile_control.get_control_input(heading_cmd, altitude_cmd)
            self.command_missile(aileron_cmd, elevator_cmd, rudder_cmd, throttle_cmd)  
            self.fdm.run()
            if self.conf.fg_sleep_time is not None:
                time.sleep(self.conf.fg_sleep_time)



    def command_missile(self, aileron_cmd, elevator_cmd, rudder_cmd, throttle_cmd):

        self.set_aileron(aileron_cmd)
        self.set_elevator(elevator_cmd)
        self.set_rudder(rudder_cmd)
        # todo velocity control. for now set 0.49 throttle at the imput 
        self.set_throttle(throttle_cmd)






