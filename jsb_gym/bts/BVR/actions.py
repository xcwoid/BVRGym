import py_trees as pt 
from jsb_gym.utils.geospatial import bearing_between_agents, to_360

class MAW_guide_evade_action(pt.behaviour.Behaviour):
    def __init__(self, name, agent):
        super(MAW_guide_evade_action, self).__init__(name)
        self.agent = agent
        self.offset = 80
        self.altitude = 7e3
        
    def update(self):
        # evade in flank 
        heading = to_360(bearing_between_agents(self.agent, self.agent.target))
        self.heading = (heading + self.offset ) %360
        self.launch_missile = False
        return pt.common.Status.RUNNING

class MAW_evade_action(pt.behaviour.Behaviour):
    def __init__(self, name, agent):
        super(MAW_evade_action, self).__init__(name)
        self.agent = agent
        self.offset = 180
        self.altitude = 5e3

        
    def update(self):
        heading = to_360(bearing_between_agents(self.agent, self.agent.target))
        self.heading = (heading + self.offset ) %360
        self.launch_missile = False

        return pt.common.Status.RUNNING

class Guide_own_action(pt.behaviour.Behaviour):
    def __init__(self, name, agent):
        super(Guide_own_action, self).__init__(name)
        self.agent = agent
        self.offset = 45
        self.altitude = 10e3

    
    def update(self):
        
        heading = to_360(bearing_between_agents(self.agent, self.agent.target))
        self.heading = (heading + self.offset ) %360
        self.launch_missile = False
        return pt.common.Status.RUNNING

class Pursue_action(pt.behaviour.Behaviour):
    def __init__(self, name, agent):
        super(Pursue_action, self).__init__(name)
        self.agent = agent
        self.offset = 0
        self.altitude = 10e3
        self.launch_missile = False

    def update(self):
        
        heading = to_360(bearing_between_agents(self.agent, self.agent.target))
        self.heading = (heading + self.offset ) %360        
        return pt.common.Status.RUNNING


class Launch_action(pt.behaviour.Behaviour):
    def __init__(self, name, agent):
        super(Launch_action, self).__init__(name)
        self.agent = agent
        self.altitude = 10e3
        self.launch_missile = True
        self.offset = 0

    def update(self):
        heading = to_360(bearing_between_agents(self.agent, self.agent.target))
        self.heading = (heading + self.offset ) %360
        return pt.common.Status.RUNNING
