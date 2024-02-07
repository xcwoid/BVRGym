import numpy as np

general = {
    'r_pid': 12,
    'r_tick': 10,
    'fg_sleep' : 0.0025,
    'fg_r_pid': 1,
    'fg_r_tick': 1,
    'fdm_xml': 'scripts/f16_test.xml'}


pid_roll         = {'P':  0.01, 'I': 0.0,  'D':  0.9}
pid_roll_sec     = {'P':  0.2, 'I': 0.0,  'D':  0.2}
pid_pitch        = {'P':  0.3, 'I': 0.0, 'D':   1.0}
#pid_heading      = {'P':  2.0, 'I': 0.01, 'D':   1.0}
pid_rudder_theta = {'P':  0.05, 'I': 0.0, 'D':   1.0}
pid_rudder_psi = {'P':  0.05, 'I': 0.0, 'D':   1.0}
pid_elevator_psi = {'P':  0.1, 'I': 0.0, 'D':   4.0}

randy = {
    'split_at' : 0.0,
    'act_min': -1.0,
    'act_max':  1.0,
    'slack_min': -0.2,
    'slack_max': 1.0}

ctrl = {
    'heading_clip': 5,
    'phi_min': -180.0,
    'phi_max': 180.0,
    'theta_min': -90.0,
    'theta_max': 90.0,
    'psi_min': 0.0,
    'psi_max': 360.0,
    'alt_min': 3e3,
    'alt_max': 12e3,
    'tan_ref': 2e3,    
    'vel_mach_min' : 0.1,
    'vel_mach_max': 1.2,
    'thr_min': 0.2,
    'thr_max': 0.69,
    'alt_act_space': 1e3,
    'alt_act_space_min': 1e3,
    'alt_act_space_max': 2e3,
    'theta_act_space': 10,
    'theta_act_space_min': 10,
    'theta_act_space_max': 30,
    'v_down_min': -850,
    'v_down_max':  850}

act = {
    'a_min': -1, 
    'a_max':   1,
    'split_at': 0.0,
    'relative_heading': False,
    'roll_pull': 80}

crank = {'left': True,
         'time_max': 25,
         'angle': 35,
         'time': 0
         }

set_heading_PID = {'roll_max': 80}

