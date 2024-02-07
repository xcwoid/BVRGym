import numpy as np

# r_tick updates 0.1 sec 
general = {
    'r_pid': 12,
    'r_tick': 10,
    'fg_sleep' : 0.0025,
    'fg_r_pid': 1,
    'fg_r_tick': 1,
    'rec' : False,
    'fdm_xml': 'scripts/AIM_test.xml'}

pid_roll    = {'P':  1,  'I':  0.0,  'D': 4.0, 'I_min': -0.1, 'I_max': 0.1}
pid_pitch   = {'P':  0.1,'I':  0.0,'D': 3, 'I_min': -0.1, 'I_max': 0.1}
pid_heading = {'P':  1,  'I':  0.0,  'D': 4, 'I_min': -0.1, 'I_max': 0.1}

PN = {
    'N':3,
    'dt': 0.1,
    'cp': 360,
    'range_NE': 80e3,
    'range_D':  20e3,
    'steady_flight_sec':2,
    'target_lost_below_mach': 1,
    'target_lost_below_alt': 1e3,
    'dive_at': 30e3,
    'tan_ref' : 10e3,
    'count_lost' : 10,
    'theta_min_cruise': -30.0,
    'theta_max_cruise': 30.0,
    'theta_min': -80.0,
    'theta_max':  80.0,
    'warhead' : 300,
    'alt_cruise':15e3
}


ctrl = {
    'alt_max': 17e3,
    'alt_min': 1e3,
    'phi_min': -180.0,
    'phi_max': 180.0,
    'theta_min': -90.0,
    'theta_max': 90.0,
    'psi_min': 0.0,
    'psi_max': 360.0,
    'rudder_clip' : 1,
    'elevator_clip': 1,
    'aileron_clip': 1,
}