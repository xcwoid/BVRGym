from jsb_gym.envs.BaseEnv import BVRBase

from jsb_gym.agents.config import blue_wvr_agent, red_wvr_agent
from jsb_gym.agents.agents import RLWVRAgent, BTWVRAgent
from jsb_gym.bts.bts import RandomBT

from jsb_gym.utils.geospatial import dinstance_between_agents

from jsb_gym.utils.wez import in_gun_wez, angle_to_target_in_body_frame_deg

class WvrEnv(BVRBase):
    '''
    BVR Gym environment class with both agents using behavior trees
    '''
    def __init__(self, conf):
        super().__init__(conf)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        blue_wvr_agent.agent_name = "RL_blue"
        self.blue_agent = RLWVRAgent(blue_wvr_agent, self)

        red_wvr_agent.agent_name = "BTRandy_red"
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

class WvrEnv_ControlZone(WvrEnv):
    def __init__(self, conf):
        super().__init__(conf)
    
    def reset(self, seed=None, options=None):
        states = super().reset(seed=seed)
        self.old_distance_between_agents = None
        return states

    def get_reward(self, done):

        r_theta_track = self.get_relative_position_reward()

        r_closure = self.get_closure_reward()

        r_gunsnap = self.get_gunsnap_reward()

        # r_deck - autopilot takes care of this 

        # r_too_close - meh
        
        return r_theta_track + r_closure + r_gunsnap

    def get_relative_position_reward(self):
        r_rb = 0
        r_br = 0
        if abs(angle_to_target_in_body_frame_deg(self.red_agent, self.blue_agent)) > 135:
            r_rb = -0.1
        if abs(angle_to_target_in_body_frame_deg(self.blue_agent, self.red_agent)) > 135:
            r_br = 0.1
        return r_br + r_rb

    def get_gunsnap_reward(self):
        r_gunsnap_blue = in_gun_wez(self.blue_agent, self.red_agent, max_range=3_000, min_range= 2_999)

        r_gunsnap_red = - in_gun_wez(self.red_agent, self.blue_agent, max_range=3_000, min_range= 2_999)

        return r_gunsnap_blue + r_gunsnap_red

    def get_closure_reward(self):
        distance = dinstance_between_agents(self.blue_agent, self.red_agent)
        if self.old_distance_between_agents is not None:
            r_closure = 0.1 if (self.old_distance_between_agents - distance) > 0 else -0.1
        else:
            r_closure = 0.0

        self.old_distance_between_agents = distance
        return r_closure
    
class WvrEnv_AggressiveShooter(WvrEnv_ControlZone):
    def __init__(self, conf):
        super().__init__(conf)
    
    def reset(self, seed=None, options=None):
        states = super().reset(seed=seed)
        return states

    def get_reward(self, done):

        r_theta_track = self.get_theta_track_reward()

        r_closure = self.get_closure_reward()

        r_gunsnap = self.get_gunsnap_reward()

        # r_deck - autopilot takes care of this 

        # r_too_close - meh
        
        return r_theta_track + r_closure + r_gunsnap

    def get_theta_track_reward(self):
        bearing_blue2red = angle_to_target_in_body_frame_deg(self.blue_agent, self.red_agent)
        bearing_red2blue = angle_to_target_in_body_frame_deg(self.red_agent, self.blue_agent)
        
        return abs(bearing_red2blue) - abs(bearing_blue2red)


    def get_gunsnap_reward(self):
        r_gunsnap_blue = in_gun_wez(self.blue_agent, self.red_agent, max_range=3_000, min_range= 500)

        r_gunsnap_red = - in_gun_wez(self.red_agent, self.blue_agent, max_range=3_000, min_range= 500)

        return r_gunsnap_blue + r_gunsnap_red

class WvrEnv_ConservativeShooter(WvrEnv_AggressiveShooter):
    def __init__(self, conf):
        super().__init__(conf)
    
    def reset(self, seed=None, options=None):
        states = super().reset(seed=seed)
        return states

    def get_gunsnap_reward(self):
        r_gunsnap_blue = in_gun_wez(self.blue_agent, self.red_agent, max_range=3_000, min_range= 2_999)

        r_gunsnap_red = - in_gun_wez(self.red_agent, self.blue_agent, max_range=3_000, min_range= 2_999)

        return r_gunsnap_blue + r_gunsnap_red
