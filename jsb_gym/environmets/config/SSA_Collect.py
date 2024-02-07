import numpy as np

version = 1

aim1_rand = {'bearing':  [0, 360],
             'distance': [50e3, 100e3],
             'vel':      [340, 340],
             'alt':      [10e3, 10e3]}
aim = {'aim1': aim1_rand}
aim_rand = {'aim1': aim1_rand}


general = {
        'env_name': 'evasive',
        'f16_name': 'f16',
        'sim_time_max': 60*4,           
        'r_step' : 5,
        'fg_r_step' : 1,
        'missile_idle': False,
        'scale': True,
        'rec_f16': False,
        'rec_aim': False}

states= {
        'obs_space': 6,
        'act_space': 3,
        'update_states_type': 3
}

sf = {  'd_min': 50e3,
        'd_max': 100e3,
        't': general['sim_time_max'],
        'alt_min': 8e3,
        'alt_max': 12e3,
        'd_max_reward': 20e3,
        'aim_vel0_min': aim1_rand['vel'][0], 
        'aim_vel0_max': aim1_rand['vel'][1], 
        'aim_alt0_min': aim1_rand['alt'][0], 
        'aim_alt0_max': aim1_rand['alt'][1]}

f16_rand = {'lat':      [59.0, 59.0],
            'long':     [18.0, 18.0],
            'vel':      [340,340],
            'alt':      [10e3, 10e3],
            'heading' : [0, 0]}

