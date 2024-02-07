import datetime, time, os
import numpy as np

def make_log_dir():
    def __init__(self):
        pass
    def make_dir(self, path):
        time_stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        directory = path + '/'+ time_stamp
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory

class EVS_logs():
    def __init__(self):
        self.load_rec_var()

    def create_logdir(self, env):
        time_stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        directory = 'jsb_gym/trained/' + env.env_name + '/'+ time_stamp
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory

    def load_rec_var(self):
        self.alt     = []
        self.alt_ref     = []
        self.alt_tgt = []
        
        self.vel     = []
        self.vel_tgt = []
        
        self.lat  = []
        self.long = []
        self.lat_tgt    = []
        self.long_tgt   = []

        self.phi  = []
        self.phi_ref   = []
        self.theta= []
        self.theta_ref = []
        self.psi  = []
        self.psi_ref   = []

        self.sim_time  = []
        self.dist_to_tgt= []

        self.PN_accx = []
        self.PN_accy = []
        self.PN_accz = []

        self.tgt_east = []
        self.tgt_north = []
        self.tgt_down = []
        
    def record(self, aim, tgt):
        self.alt.append(aim.get_altitude())
        self.alt_ref.append(aim.altitude_ref)
        self.alt_tgt.append(tgt.get_altitude())

        self.vel.append(aim.get_Mach())
        self.vel_tgt.append(tgt.get_Mach())

        self.lat.append(aim.get_lat_gc_deg())
        self.long.append(aim.get_long_gc_deg())
        self.lat_tgt.append(tgt.get_lat_gc_deg())
        self.long_tgt.append(tgt.get_long_gc_deg())

        self.phi.append(aim.get_phi())
        self.phi_ref.append(aim.roll_ref)
        self.theta.append(aim.get_theta())
        self.theta_ref.append(aim.theta_ref)
        self.psi.append(aim.get_psi())
        self.psi_ref.append(aim.psi_ref)
        
        self.sim_time.append(aim.get_sim_time_sec())
        self.dist_to_tgt.append(aim.position_tgt_NED_norm)

        self.PN_accx.append(aim.acceleration_cmd_NED[0])
        self.PN_accy.append(aim.acceleration_cmd_NED[1])
        self.PN_accz.append(aim.acceleration_cmd_NED[2])

        self.tgt_east.append(aim.tgt_east)
        self.tgt_north.append(aim.tgt_north)
        self.tgt_down.append(aim.tgt_down)

    def rec_idle(self, tgt):
        self.alt_tgt.append(tgt.get_altitude())
        self.vel_tgt.append(tgt.get_Mach())
        self.lat_tgt.append(tgt.get_lat_gc_deg())
        self.long_tgt.append(tgt.get_long_gc_deg())
        self.sim_time.append(tgt.get_sim_time_sec())

class AIM_logs():
    def __init__(self):
        self.load_rec_var()

    def load_rec_var(self):
        self.phi  = []
        self.phi_ref   = []
        self.theta= []
        self.theta_ref = []
        self.psi  = []
        self.psi_ref   = []
        self.lat = []
        self.long = []
        self.sim_time  = []
        self.alt = []

        self.dist = []
        self.vel     = []
        
    def record(self, aim):

        self.phi.append(aim.get_phi())
        self.phi_ref.append(aim.roll_ref)
        self.theta.append(aim.get_theta())
        self.theta_ref.append(aim.theta_ref)
        self.psi.append(aim.get_psi())
        self.psi_ref.append(aim.psi_ref)
        self.lat.append(aim.get_lat_gc_deg())
        self.long.append(aim.get_long_gc_deg())
        self.sim_time.append(aim.get_sim_time_sec())
        self.dist.append(aim.position_tgt_NED_norm)
        self.alt.append(aim.get_altitude())
        self.vel.append(aim.get_Mach())

class F16_logs():
    def __init__(self):
        self.load_rec_var()

    def load_rec_var(self):
        self.phi  = []
        self.phi_ref   = []
        self.theta= []
        self.theta_ref = []
        self.psi  = []
        self.psi_ref   = []
        self.lat = []
        self.long = []
        self.sim_time  = []
        self.rudder_cmd = []
        self.elevator_cmd = []
        self.aileron_cmd = []

        self.alt = []
        self.alt_ref = []
        self.vel     = []
     
    def record(self, f16):

        self.phi.append(f16.get_phi())
        self.phi_ref.append(f16.roll_ref)
        self.theta.append(f16.get_theta())
        self.theta_ref.append(f16.theta_ref)
        self.psi.append(f16.get_psi())
        self.psi_ref.append(f16.psi_ref)
        self.sim_time.append(f16.get_sim_time_sec())
        self.rudder_cmd.append(f16.rudder_cmd)
        self.elevator_cmd.append(f16.elevator_cmd)
        self.aileron_cmd.append(f16.aileron_cmd)


        self.lat.append(f16.get_lat_gc_deg())
        self.long.append(f16.get_long_gc_deg())
        
        self.alt.append(f16.get_altitude())
        self.alt_ref.append(f16.alt_ref)
        self.vel.append(f16.get_Mach())

class State_logs():
    def __init__(self, env):
        
        self.obs_space = env.observation_space
        self.first_run = True
        self.slack = np.zeros((self.obs_space + 1))
    
    def create_logdir(self, directory = 'jsb_gym/logs/States/tmp', add_date= True):
        if add_date:
            time_stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            directory += '/'+ time_stamp
            
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except:
                print('Error with creating directory. Continue with script...')
                
        self.directory = directory

    def stack(self, state):
        self.slack[:-1] = state[:].copy()
        self.state = np.vstack((self.state, self.slack))
    
    def reset(self, state):
        self.state = np.zeros((self.obs_space + 1))
        self.state[:-1] = state[:].copy()
        
    def stack_block(self, reward):
        if len(self.state.shape) == 1:
            self.state[-1] = reward
        else:
            self.state[:,-1] = reward
        if self.first_run:
            self.first_run = False
            self.state_block = self.state.copy()
            print(self.state_block.shape)
        else:
            self.state_block = np.vstack((self.state_block, self.state))
            #print(self.state_block)
            print(self.state_block.shape)
    
    def save(self, name = 'state_block'):
        np.save(self.directory + '/' + name + '.npy', self.state_block)

class Logs():
    def __init__(self):
        self.logs = {}
        self.est = {}
        self.est_min = {}
        self.est_max = {}
    
    def create_logdir(self, env):
        time_stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        directory = 'jsb_gym/trained/' + env.env_name + '/'+ time_stamp
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory
    
    def add_est(self, name):
        self.est[name] = []

    def add_est_fill(self, name):
        self.est[name] = []
        self.est_min[name] = []
        self.est_max[name] = []

    def add_log(self, name, log_type = 'basic'):
        self.logs[name] = {}
        if log_type == 'basic':
            self.rec_var = ['alt', 'vel', 'lat', 'long', 'phi', 'theta', 'psi', 'sim_time']

        for n in self.rec_var:
            self.logs[name][n] = []
        
    def record(self, name, obj, log_type = 'basic'):
        if log_type == 'basic':
            self.logs[name]['alt'].append(obj.get_altitude())
            self.logs[name]['vel'].append(obj.get_Mach())
            self.logs[name]['lat'].append(obj.get_lat_gc_deg())
            self.logs[name]['long'].append(obj.get_long_gc_deg())
            self.logs[name]['phi'].append(obj.get_phi())
            self.logs[name]['theta'].append(obj.get_theta())
            self.logs[name]['psi'].append(obj.get_psi())
            self.logs[name]['sim_time'].append(obj.get_sim_time_sec())

    def rec_est(self, name, est):
        self.est[name].append(est)

    def rec_est_fill(self, name, est, est_min, est_max):
        self.est[name].append(est)
        self.est_min[name].append(est_min)
        self.est_max[name].append(est_max)
    
    def save_logs(self, path= "my_dict.json"):
        import json

        # open a file in write mode
        with open(path, "w") as f:
            # write the dictionary to the file using json.dump()
            json.dump(self.logs, f)