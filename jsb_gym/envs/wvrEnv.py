from jsb_gym.envs.BaseEnv import BVRBase

from jsb_gym.agents.config import blue_agent, red_agent
from jsb_gym.agents.agents import BTBVRAgent, RLWVRAgent, BTWVRAgent
from jsb_gym.bts.bts import BVRBT, WVRBT, RandomBT


class WvrEnv_BTvsBT(BVRBase):
    '''
    BVR Gym environment class with both agents using behavior trees
    '''
    def __init__(self, conf):
        super().__init__(conf)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        blue_agent.agent_name = "BT_blue"
        #blue_agent.aircraft_simObj_conf.data_output_xml= 'data_output/flightgear.xml'
        #blue_agent.aircraft_simObj_conf.fg_sleep_time= 0.005
        #blue_agent.aircraft_simObj_conf.aircraft_simulation.Sim_time_step = 1


        self.blue_agent = BTBVRAgent(blue_agent, self)
        self.blue_agent.load_BT(BVRBT)
        

        red_agent.agent_name = "RandomBT_red"
        #red_agent.aircraft_simObj_conf.data_output_xml= 'data_output/flightgear_red.xml'
        #red_agent.aircraft_simObj_conf.fg_sleep_time= 0.005
        #red_agent.aircraft_simObj_conf.aircraft_simulation.Sim_time_step = 1

        self.red_agent = BTWVRAgent(red_agent, self)
        self.red_agent.load_BT(RandomBT)
        

        self.all_agents = [self.blue_agent, self.red_agent]

        self.blue_agent.set_target(self.red_agent)

        self.red_agent.set_target(self.blue_agent)
        
        self.update_state()

        return self.state, {}
    
    def step(self, action):
        # apply action to agent
        self.blue_agent.apply_action()
        self.red_agent.apply_action()
        # get new observation
        self.update_state()
        
        self.done = self.max_episode_time_passed()
        # calculate reward
        # check done
        
        self.reward = 0
        
        return self.state, self.reward, self.done, self.max_episode_time_passed(), {'done': self.done, 'trunk': self.max_episode_time_passed()}