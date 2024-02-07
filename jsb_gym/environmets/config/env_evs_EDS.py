import numpy as np

version = 1

aim1 = {'bearing': 0,
        'distance': 70e3,
        'vel': 300,
        'alt': 10e3}

aim = {'aim1': aim1}

aim1_rand = {'bearing':  [0, 360],
             'distance': [50e3, 100e3],
             'vel':      [290, 340],
             'alt':      [9e3, 12e3]}

aim_rand = {'aim1': aim1_rand}

general = {
        'env_name': 'evasive',
        'f16_name': 'f16',
        'sim_time_max': 60*4,           
        'r_step' : 3,
        'fg_r_step' : 1,
        'missile_idle': False,
        'scale': True,
        'rec':False}

states= {
        'obs_space': 10,
        'act_space': 2,
        'NE_scale' : [-160e3, 160e3],
        'D_scale' : [-12e3, 12e3],
        'v_NED_scale': [-600, 600],
        'update_states_type': 1
}

sf = {  'd_min': 20e3,
        'd_max': 150e3,
        't': general['sim_time_max'],
        'mach_max': 2,
        'alt_min': 5e3,
        'alt_max': 14e3,
        'd_max_reward': 20e3,
        'aim_vel0_min': aim1_rand['vel'][0], 
        'aim_vel0_max': aim1_rand['vel'][1], 
        'aim_alt0_min': aim1_rand['alt'][0], 
        'aim_alt0_max': aim1_rand['alt'][1]}

f16 = { 'lat':      59.0,
        'long':     18.0,
        'vel':      300,
        'alt':      10e3,
        'heading' : 0}


f16_rand = {'lat':      [59.0, 59.0],
            'long':     [18.0, 18.0],
            'vel':      [250,330],
            'alt':      [6e3, 12e3],
            'heading' : [0, 360]}

