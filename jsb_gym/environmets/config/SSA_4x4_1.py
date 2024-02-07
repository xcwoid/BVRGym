import numpy as np

version = 1

general = {
        'env_name': 'SSA',
        'f16_name': 'f16',
        'sim_time_max': 60*4,           
        'r_step' : 20,
        'fg_r_step' : 1,
        'missile_idle': False,
        'scale': True,
        'rec_f16': False,
        'rec_aim': False}

states= {
        'obs_space': 6,
        'act_space': 2,
        'update_states_type': 1
}

sf = {  'd_min': 40e3,
        'd_max': 120e3,
        't': general['sim_time_max'],
        'mach_max': 2,
        'alt_min': 3e3,
        'alt_max': 14e3,
        'd_max_reward': 20e3,
        'aim_vel0_min': 280, 
        'aim_vel0_max': 320, 
        'aim_alt0_min': 9e3, 
        'aim_alt0_max': 11e3}

psi_b = 0
f16_1 = { 'lat':    59.0,
        'long':     18.0,
        'vel':      300,
        'alt':      10e3,
        'heading' : psi_b}

f16_2 = { 'lat':    59.0,
        'long':     18.05,
        'vel':      300,
        'alt':      10e3,
        'heading' : psi_b}

f16_3 = { 'lat':    59.0,
        'long':     18.1,
        'vel':      300,
        'alt':      10e3,
        'heading' : psi_b}

f16_4 = { 'lat':    59.0,
        'long':     18.15,
        'vel':      300,
        'alt':      10e3,
        'heading' : psi_b}


psi_r = 180
f16r_1 = { 'lat':   60.0,
        'long':     18.075,
        'vel':      300,
        'alt':      10e3,
        'heading' : psi_r}

f16r_2 = { 'lat':   60.1,
        'long':     18.075,
        'vel':      300,
        'alt':      10e3,
        'heading' : psi_r}

f16r_3 = { 'lat':   60.2,
        'long':     18.075,
        'vel':      300,
        'alt':      10e3,
        'heading' : psi_r}

f16r_4 = { 'lat':   60.3,
        'long':     18.075,
        'vel':      300,
        'alt':      10e3,
        'heading' : psi_r}

f16  = {'f16_1':f16_1, 'f16_2':f16_2, 'f16_3':f16_3, 'f16_4':f16_4}
f16r = {'f16r_1':f16r_1, 'f16r_2':f16r_2, 'f16r_3':f16r_3, 'f16r_4':f16r_4}

#f16  = {'f16_1':f16_1}
#f16r = {'f16r_1':f16r_1}
