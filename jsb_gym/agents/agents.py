from jsb_gym.simObjects.aircraft import F16BVR
from jsb_gym.simObjects.missiles import AAMBVR
import numpy as np
from jsb_gym.utils.geospatial import dinstance_between_agents, relative_bearing_between_agents


class BaseBVRAgent():
    def __init__(self, conf, env):
        self.conf = conf
        # agent level configurations
        self.agent_parameters = conf.agent_parameters
        self.aircraft_simObj_conf =  conf.aircraft_simObj_conf
        self.missile_simObj_conf  =  conf.missile_simObj_conf
        
        # flight dynamics model level configurations
        self.simObj = F16BVR(self.aircraft_simObj_conf)
        self.reset_agent()
        self.reset_ammo()

        # environment reference to access other agents etc.
        self.env = env

        self.healthPoints = 1.0
        self.target = None

    def reset_agent(self):
        self.simObj.reset(
            lat=self.agent_parameters.lat,
            long=self.agent_parameters.long,
            alt=self.agent_parameters.alt,
            vel=self.agent_parameters.vel,
            heading=self.agent_parameters.heading)

    def set_target(self, target_agent):
        self.target = target_agent        

    def is_own_missile_active(self):
        return any(self.ammo[i].active for i in self.ammo.keys()) 

    def apply_action(self, action):
        self.simObj.step(action)
        self.apply_missile_action()

    def apply_missile_action(self):
        for i in self.ammo:
            if self.ammo[i].is_active():
                self.ammo[i].step()

    def launch_missile(self):
        if not self.is_own_missile_active():
            for i in self.ammo:
                if self.ammo[i].is_ready_to_launch():
                    self.ammo[i].set_target(self.target)
                    self.ammo[i].launch(self)
                    break

    
    def reset_ammo(self):    
        self.ammo = {}
        if self.agent_parameters.ammo > 0:
            for i in range(self.agent_parameters.ammo):
                self.ammo[str(i)] = AAMBVR(self.missile_simObj_conf)


class BTBVRAgent(BaseBVRAgent):

    def load_BT(self, BT_model):
        self.BT = BT_model(self)

    def apply_action(self):
        self.BT.tick()
        heading = self.BT.heading
        altitude = self.BT.altitude
        throttle = 0.49
        if self.BT.launch_missile:
            self.launch_missile()
        action = np.array([heading, altitude, throttle])
        super().apply_action(action)


class RLBVRAgent(BaseBVRAgent):
    
    def launch_conditions_met(self):
        self.launch_distance = 60e3
        self.relative_bearing_scope = 40
        dist = dinstance_between_agents(self, self.target)
        relative_bearing = relative_bearing_between_agents(self, self.target)        
        
        if dist < self.launch_distance and abs(relative_bearing) < self.relative_bearing_scope:
            return True
        else:
            return False

    def apply_action(self, action):
        if self.launch_conditions_met():
            self.launch_missile()
        super().apply_action(action)




class BTWVRAgent(BaseBVRAgent):
    
    def load_BT(self, BT_model):
        self.BT = BT_model(self)

    def apply_action(self):
        self.BT.tick()
        heading = self.BT.heading
        altitude = self.BT.altitude
        throttle = 0.49
        action = np.array([heading, altitude, throttle])
        super().apply_action(action)


class RLWVRAgent(BaseBVRAgent):   
    def apply_action(self, action):
        super().apply_action(action)

