import py_trees as pt 
from jsb_gym.utils.geospatial import dinstance_between_agents, relative_bearing_between_agents

class MAW_own_condition(pt.behaviour.Behaviour):
    def __init__(self, name, agent):
        super(MAW_own_condition, self).__init__(name)
        self.agent = agent

    def no_own_active_missile(self):
        active = self.agent.is_own_missile_active()
        if active:
            return False
        else:
            return True

    def update(self):
        if self.no_own_active_missile():
            return pt.common.Status.SUCCESS
        else:
            return pt.common.Status.FAILURE


class MAW_condition(pt.behaviour.Behaviour):
    def __init__(self, name, agent):
        super(MAW_condition, self).__init__(name)
        self.agent = agent
        
    def no_incomming_missile(self):
        active = self.agent.target.is_own_missile_active()
        if active:
            return False
        else:
            return True

    def update(self):
        #print('Tick 11C')
        if self.no_incomming_missile():
            return pt.common.Status.SUCCESS
        else:
            return pt.common.Status.FAILURE


class Pursue_condition(pt.behaviour.Behaviour):
    def __init__(self, name, agent):
        super(Pursue_condition, self).__init__(name)
        self.agent = agent

    def enemy_not_alive(self):
        return False

    def update(self):
        #self.feedback_message = "MAW_condition"
        if self.enemy_not_alive():
            return pt.common.Status.SUCCESS
        else:
            return pt.common.Status.FAILURE


class Launch_condition(pt.behaviour.Behaviour):
    def __init__(self, name, agent):
        super(Launch_condition, self).__init__(name)
        self.agent = agent
        self.launch_distance = 60e3
        self.relative_bearing_scope = 40

    def is_in_launch_range(self):
        dist = dinstance_between_agents(self.agent, self.agent.target)
        relative_bearing = relative_bearing_between_agents(self.agent, self.agent.target)        
        
        if dist < self.launch_distance and abs(relative_bearing) < self.relative_bearing_scope:
            return True
        else:
            return False

    def not_in_launch_position(self):
        # return status about the blue teams missile 
        if self.is_in_launch_range():
            return False
        else:
            return True

    def update(self):
        self.feedback_message = "MAW_condition"
        if self.not_in_launch_position():
            return pt.common.Status.SUCCESS
        else:
            return pt.common.Status.FAILURE
