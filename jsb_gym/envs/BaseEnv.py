import gymnasium as gym
from gymnasium import spaces

import numpy as np

from jsb_gym.agents.config import blue_agent, red_agent
from jsb_gym.agents.agents import RLBVRAgent, BTBVRAgent

from jsb_gym.utils.geospatial import dinstance_between_agents, bearing_between_agents, to_360
from jsb_gym.utils.loggers import TacviewLogger

from jsb_gym.utils.scale import scale_between_inv, scale_between

from jsb_gym.bts.bts import BVRBT

class BVRBase(gym.Env):
    def __init__(self, conf):
        '''
        Main BVR Gym environment class
        Parameters  
        ----------
        conf : Config object
            Configuration object containing environment parameters
        '''
        # Environment config file 
        super().__init__()
        self.conf = conf
        self.obs_shape = conf.observation_shape
        self.act_shape = conf.action_shape

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=self.obs_shape, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=self.act_shape, dtype=np.float32)        

        self.state = None
        self.done = False
        self.tacview_logger = None
        self.observation = {}

    def reset(self, seed=None, options=None):
        ''''
        Reset the environment to an initial state
        '''
        super().reset(seed=seed)
        
        self.blue_agent = RLBVRAgent(blue_agent, self)
        
        self.red_agent = BTBVRAgent(red_agent, self)
        
        self.red_agent.load_BT(BVRBT)
        
        self.all_agents = [self.blue_agent, self.red_agent]

        self.blue_agent.set_target(self.red_agent)

        self.red_agent.set_target(self.blue_agent)
        
        self.update_state()

        return self.state, {}

    def log_tacview(self):
        '''
        Log flight data for Tacview
        '''
        if self.conf.tacview_output_dir is not None:
            if self.tacview_logger is None:
                self.tacview_logger = TacviewLogger(self)
            elif self.done:
                self.tacview_logger.save_logs()
            else:
                self.tacview_logger.log_flight_data()
              
    def update_state(self):
        self.update_observation()
        obs_nn = self.from_obs2nn(self.blue_agent)
        
        if self.state is None:
            self.state = np.tile(obs_nn, (self.obs_shape[0], 1))
        else:
            self.state = np.roll(self.state, shift=-1, axis=0)
            self.state[-1,:] = obs_nn

    def update_observation(self):
        
        self.observation['bearing'] = to_360(bearing_between_agents(self.blue_agent, self.blue_agent.target))
        self.observation['heading'] = self.blue_agent.simObj.get_psi()
        self.observation['mach'] = self.blue_agent.simObj.get_mach()
        self.observation['altitude'] = self.blue_agent.simObj.get_altitude()

        self.observation['d'] = dinstance_between_agents(self.blue_agent, self.blue_agent.target)

        self.observation['enemy_bearing'] = to_360(bearing_between_agents(self.red_agent, self.red_agent.target)) 

        self.observation['enemy_heading'] = self.red_agent.simObj.get_psi()
        self.observation['enemy_mach'] = self.red_agent.simObj.get_mach()
        self.observation['enemy_altitude'] = self.red_agent.simObj.get_altitude()

        self.observation['own_missile_active'] = 0
        self.observation['enemy_missile_active'] = 0
        

    def step(self, action):
        # apply action to agent
        action = self.from_nn2agent(action, self.blue_agent)
        
        for i in range(self.conf.step_length):
            # If step_length is 10, this should result aplllying action for 10 sim seconds, unless you changed the sim step time in FDM config
            self.blue_agent.apply_action(action)
            self.red_agent.apply_action()
            # get new observation
            self.update_state()
            
            self.done = self.is_done()
            # calculate reward
            # check done
            
            self.reward = self.get_reward(self.done)
            if self.done:
                break
        return self.state, self.reward, self.done, self.max_episode_time_passed(), {'done': self.done, 'trunk': self.max_episode_time_passed()}

    def from_obs2nn(self, agent):
        '''
        Convert observation dictionary to neural network input array
        '''
        bearing_sin = np.sin(np.radians(self.observation['bearing']))
        bearing_cos = np.cos(np.radians(self.observation['bearing']))
        heading_sin = np.sin(np.radians(self.observation['heading']))
        heading_cos = np.cos(np.radians(self.observation['heading']))
        
        mach = scale_between(self.observation['mach'], a_min = 0.1, a_max = 1.5)
        altitude = scale_between(self.observation['altitude'], a_min = agent.simObj.conf.aircraft_limits.alt_min,
                               a_max = agent.simObj.conf.aircraft_limits.alt_max )
        d = scale_between(self.observation['d'], a_min = 0.0, a_max = 120e3)
        
        enemy_bearing_sin= np.sin(np.radians(self.observation['enemy_bearing']))
        enemy_bearing_cos= np.cos(np.radians(self.observation['enemy_bearing']))
        enemy_heading_sin = np.sin(np.radians(self.observation['enemy_heading']))
        enemy_heading_cos = np.cos(np.radians(self.observation['enemy_heading']))
        
        enemy_mach = scale_between(self.observation['enemy_mach'], a_min = 0.1, a_max = 1.5)
        enemy_altitude = scale_between(self.observation['enemy_altitude'], a_min = agent.simObj.conf.aircraft_limits.alt_min,
                               a_max = agent.simObj.conf.aircraft_limits.alt_max )
        
        return np.array([bearing_sin, bearing_cos, heading_sin, heading_cos, mach, altitude, d, enemy_bearing_sin, enemy_bearing_cos, 
                         enemy_heading_sin, enemy_heading_cos, enemy_mach, enemy_altitude, 
                         self.observation['own_missile_active'], self.observation['enemy_missile_active']])
        
        
        

    
    def from_nn2agent(self, action, agent):
        # heading 
        action[0] = scale_between_inv(action[0],
                                      a_min= agent.simObj.conf.aircraft_limits.psi_min,
                                        a_max= agent.simObj.conf.aircraft_limits.psi_max)        
        # altitude 
        action[1] = scale_between_inv(action[1],
                                      a_min= agent.simObj.conf.aircraft_limits.alt_min,
                                        a_max= agent.simObj.conf.aircraft_limits.alt_max)
        # throttle full thrust without or with afterburner 
        action[2] = 0.49 if action[2] <= 0.0 else 0.69
        return action


    def get_reward(self, is_done):

        if is_done:
            if self.red_agent.healthPoints <= 0.0:
                return 1

            elif self.red_agent.healthPoints <= 0.0:
                return -1
            else:
                return -1
        else:
            return 0

    def is_done(self):
        for agent in self.all_agents:
            if agent.healthPoints <= 0.0:
                return True
            
        
        if self.max_episode_time_passed():
            return True 
        return False
    
    def max_episode_time_passed(self):
        if self.blue_agent.simObj.get_sim_time_sec() >= self.conf.max_episode_time:
            return True 
        return False
               

class BVREnv_BTvsBT(BVRBase):
    '''
    BVR Gym environment class with both agents using behavior trees
    '''
    def __init__(self, conf):
        super().__init__(conf)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        blue_agent.agent_name = "BT_blue"
        self.blue_agent = BTBVRAgent(blue_agent, self)
        
        self.red_agent = BTBVRAgent(red_agent, self)
        
        self.blue_agent.load_BT(BVRBT)
        self.red_agent.load_BT(BVRBT)
        
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