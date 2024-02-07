import numpy as np

version = 1

aim1 = {'bearing': 0,
        'distance': 110e3,
        'vel': 300,
        'alt': 12e3}

aim2 = {'bearing': 0,
        'distance': 110e3,
        'vel': 300,
        'alt': 12e3}

aim1r = {'bearing': 0,
        'distance': 110e3,
        'vel': 300,
        'alt': 12e3}

aim2r = {'bearing': 0,
        'distance': 110e3,
        'vel': 300,
        'alt': 12e3}

aim = {'aim1': aim1, 'aim2': aim2}

aimr = {'aim1r': aim1r, 'aim2r': aim2r}

general = {
        'env_name': 'PPODog',
        'f16_name': 'f16',
        'f16r_name': 'f16r',
        'sim_time_max': 60*16,           
        'r_step' : 10,
        'fg_r_step' : 1,
        'missile_idle': False,
        'scale': True,
        'rec':False}

states= {
        'obs_space': 8,
        'act_space': 2,
        'update_states_type': 3
}

logs= {'log_path': '/home/edvards/workspace/jsbsim/jsb_gym/logs/BVRGym',
       'save_to':'/home/edvards/workspace/jsbsim/jsb_gym/plots/BVRGym'}

sf = {  'd_min': 20e3,
        'd_max': 120e3,
        't': general['sim_time_max'],
        'mach_max': 2,
        'alt_min': 3e3,
        'alt_max': 12e3,
        'head_min': 0,
        'head_max': 360 }


f16 = { 'lat':      58.3,
        'long':     18.0,
        'vel':      350,
        'alt':      10e3,
        'heading' : 0}

f16r = { 'lat':      59.0,
         'long':     18.0,
         'vel':      350,
         'alt':      10e3,
         'heading' : 180}
