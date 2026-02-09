import py_trees as pt 
from jsb_gym.utils.geospatial import bearing_between_agents, to_360
import random

class WVR_Pursue_action(pt.behaviour.Behaviour):
    def __init__(self, name, agent):
        super(WVR_Pursue_action, self).__init__(name)
        self.agent = agent
        self.offset = 0
        self.altitude = None
        self.launch_missile = False

    def update(self):

        self.altitude = self.agent.target.simObj.get_altitude()
        heading = to_360(bearing_between_agents(self.agent, self.agent.target))
        self.heading = (heading + self.offset ) %360        
        return pt.common.Status.RUNNING



class WVR_Random_action(pt.behaviour.Behaviour):
    def __init__(self, name, agent):
        super(WVR_Random_action, self).__init__(name)
        self.agent = agent
        self.altitude_list = range(2000, 10001, 1000)
        self.heading_list = range(0, 360, 20)
        self.altitude = random.choice(list(self.altitude_list))
        self.heading = random.choice(list(self.heading_list))
        
    def update(self):
        if random.random() < 0.01:
            self.altitude = random.choice(list(self.altitude_list))
        if random.random() < 0.09:
            self.heading = random.choice(list(self.heading_list))
        return pt.common.Status.RUNNING

