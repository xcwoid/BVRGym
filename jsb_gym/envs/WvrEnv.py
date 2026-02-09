from jsb_gym.envs.BaseEnv import BVRBase

from jsb_gym.agents.config import blue_wvr_agent, red_wvr_agent
from jsb_gym.agents.agents import RLWVRAgent, BTWVRAgent
from jsb_gym.bts.bts import WVRBT, RandomBT

from jsb_gym.utils.geospatial import bearing_between_agents

class WvrEnv(BVRBase):
    '''
    BVR Gym environment class with both agents using behavior trees
    '''
    def __init__(self, conf):
        super().__init__(conf)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        blue_wvr_agent.agent_name = "BT_blue"
        self.blue_agent = RLWVRAgent(blue_wvr_agent, self)

        red_wvr_agent.agent_name = "RandomBT_red"
        self.red_agent = BTWVRAgent(red_wvr_agent, self)
        self.red_agent.load_BT(RandomBT)
        

        self.all_agents = [self.blue_agent, self.red_agent]

        self.blue_agent.set_target(self.red_agent)

        self.red_agent.set_target(self.blue_agent)
        
        self.update_state()

        return self.state, {}
    
    def step(self, action):
        action = self.from_nn2agent(action, self.blue_agent)
        
        for i in range(self.conf.step_length):

            # apply action to agent
            self.blue_agent.apply_action(action)
            self.red_agent.apply_action()
            # get new observation
            self.update_state()
            
            self.done =  self.is_done()
            # calculate reward
            # check done
            
            self.reward = self.get_reward(self.done)
            if self.done:
                break
                    
        return self.state, self.reward, self.done, self.max_episode_time_passed(), {'done': self.done, 'trunk': self.max_episode_time_passed()}

    def is_done(self):
        return self.max_episode_time_passed()


    def get_reward(self, done):
        return 0

class WvrEnv_fg(WvrEnv):

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        blue_wvr_agent.agent_name = "BT_blue"
        blue_wvr_agent.aircraft_simObj_conf.data_output_xml= 'data_output/flightgear.xml'
        blue_wvr_agent.aircraft_simObj_conf.fg_sleep_time= 0.005
        blue_wvr_agent.aircraft_simObj_conf.aircraft_simulation.Sim_time_step = 1

        self.blue_agent = RLWVRAgent(blue_wvr_agent, self)
        
        red_wvr_agent.agent_name = "RandomBT_red"
        red_wvr_agent.aircraft_simObj_conf.data_output_xml= 'data_output/flightgear_red.xml'
        red_wvr_agent.aircraft_simObj_conf.fg_sleep_time= 0.005
        red_wvr_agent.aircraft_simObj_conf.aircraft_simulation.Sim_time_step = 1

        self.red_agent = BTWVRAgent(red_wvr_agent, self)
        self.red_agent.load_BT(RandomBT)
        
        self.all_agents = [self.blue_agent, self.red_agent]

        self.blue_agent.set_target(self.red_agent)

        self.red_agent.set_target(self.blue_agent)
        
        self.update_state()

        return self.state, {}


class WvrEnv_ControlZone(WvrEnv):
    def __init__(self, conf):
        super().__init__(conf)

    def get_reward(self, done):
        """
        Rtrack θ penalizes the agent for having a non-zero track angle
        """
        bearing_blue2red = bearing_between_agents(self.blue_agent, self.red_agent)
        bearing_red2blue = bearing_between_agents(self.red_agent, self.blue_agent)
        
        r_relative_pos = abs(bearing_red2blue) - abs(bearing_blue2red)

        #r_track = -abs(bearing_blue2red)
        
        # r_closure

        # Rgunsnap(blue)

        # Rgunsnap(red)

        # Rdeck - autopilot takes care of this 

        # Rtoo close
        
        return r_relative_pos
