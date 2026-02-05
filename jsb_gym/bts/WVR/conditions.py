import py_trees as pt 
from jsb_gym.utils.geospatial import dinstance_between_agents, relative_bearing_between_agents

class WVR_Pursue_condition(pt.behaviour.Behaviour):
    def __init__(self, name, agent):
        super(WVR_Pursue_condition, self).__init__(name)
        self.agent = agent

    def enemy_not_alive(self):
        return False

    def update(self):
        #self.feedback_message = "MAW_condition"
        if self.enemy_not_alive():
            return pt.common.Status.SUCCESS
        else:
            return pt.common.Status.FAILURE


