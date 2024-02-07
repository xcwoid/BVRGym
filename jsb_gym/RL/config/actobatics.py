conf_ppo = {'max_episodes': 500000,
            'update_timestep': 600 * 3,
            'action_std' : 0.5,
            'K_epochs': 80,               # update policy for K epochs
            'eps_clip': 0.2,              # clip parameter for PPO
            'gamma' : 0.99,                # discount factor
            'lr': 0.001,                 # parameters for Adam optimizer
            'betas' : (0.9, 0.999),   
            'random_seed': 5,
            'lam_a' : 0.01,
            'use_norm_rewards': True}