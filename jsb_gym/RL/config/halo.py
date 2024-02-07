conf_ppo = {'max_episodes': 100000,
            'update_timestep': 60 * 20,   # max time env_evs 180 sec, 60 -> one episode of actions
            'action_std' : 0.5,
            'K_epochs': 80,               # update policy for K epochs
            'eps_clip': 0.2,              # clip parameter for PPO
            'gamma' : 0.99,                # discount factor
            'lr': 0.003,                 # parameters for Adam optimizer
            'betas' : (0.9, 0.999),   
            'random_seed': 1,
            'lam_a' : 0.0,
            'normalize_rewards': True,
            'nn_type' : 'tanh'}