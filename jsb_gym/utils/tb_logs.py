import os, glob
from torch.utils.tensorboard import SummaryWriter


class F16_logs():
    def __init__(self, conf, name = '/F16', reset_logs = True):
        self.log_dir = conf.logs['log_path'] + name
        if reset_logs:
            self.clear_flightdata()
        self.writer = SummaryWriter(log_dir = self.log_dir)
        
    def clear_flightdata(self):
        self.delete_content(self.log_dir+'/*')

    def delete_content(self, path):
        files = glob.glob(path)
        for f in files:
            os.remove(f)

    def record(self, TAU):
        #print('Recording F16 flight data')
        f16 = TAU
        sim_time = f16.get_sim_time_sec()
        #print(sim_time)
        # Attitude
        #print('Log', f16.aileron_cmd)
        self.writer.add_scalar("phi", f16.get_phi(), sim_time)
        self.writer.add_scalar("phi_ref", f16.roll_ref, sim_time)
        self.writer.add_scalar("theta", f16.get_theta(), sim_time)
        self.writer.add_scalar("theta_ref", f16.theta_ref, sim_time)
        self.writer.add_scalar("psi", f16.get_psi(), sim_time)
        self.writer.add_scalar("psi_ref", f16.psi_ref, sim_time)
        
        # Position
        self.writer.add_scalar("get_lat_gc_deg", f16.get_lat_gc_deg(), sim_time)
        self.writer.add_scalar("get_long_gc_deg", f16.get_long_gc_deg(), sim_time)
        self.writer.add_scalar("altitude", f16.get_altitude(), sim_time)
        self.writer.add_scalar("altitude_ref", f16.alt_ref, sim_time)
        
        self.writer.add_scalar("rudder_cmd", f16.rudder_cmd, sim_time)
        self.writer.add_scalar("elevator_cmd", f16.elevator_cmd, sim_time)
        self.writer.add_scalar("aileron_cmd", f16.aileron_cmd, sim_time)

        self.writer.add_scalar("Mach", f16.get_Mach(), sim_time)

        self.writer.add_scalar("sim_time", sim_time, sim_time)

class AIM_logs():
    def __init__(self, conf, name = '/AIM', reset_logs = True): 
        self.log_dir = conf.logs['log_path'] + name 
        if reset_logs:
            self.clear_flightdata()
        self.writer = SummaryWriter(log_dir = self.log_dir)

    def clear_flightdata(self):
        self.delete_content(self.log_dir+'/*')

    def delete_content(self, path):
        files = glob.glob(path)
        for f in files:
            os.remove(f)

    def record(self, TAU):
        aim = TAU
        sim_time = aim.get_sim_time_sec()
        # Attitude
        self.writer.add_scalar("phi", aim.get_phi(), sim_time)
        self.writer.add_scalar("phi_ref", aim.roll_ref, sim_time)
        self.writer.add_scalar("theta", aim.get_theta(), sim_time)
        self.writer.add_scalar("theta_ref", aim.theta_ref, sim_time)
        self.writer.add_scalar("psi", aim.get_psi(), sim_time)
        self.writer.add_scalar("psi_ref", aim.psi_ref, sim_time)


        # Position
        self.writer.add_scalar("get_lat_gc_deg", aim.get_lat_gc_deg(), sim_time)
        self.writer.add_scalar("get_long_gc_deg", aim.get_long_gc_deg(), sim_time)
        self.writer.add_scalar("altitude", aim.get_altitude(), sim_time)
        #self.writer.add_scalar("altitude_ref", aim.alt_ref, sim_time)
        self.writer.add_scalar("Mach", aim.get_Mach(), sim_time)

        self.writer.add_scalar("Target", aim.position_tgt_NED_norm, sim_time)
        self.writer.add_scalar("sim_time", sim_time, sim_time)

class Env_logs():
    def __init__(self, conf, name = '/ENV', reset_logs = True):
        self.log_dir = conf.logs['log_path'] + name
        
        if reset_logs:
            self.clear_flightdata()
        
        self.writer = SummaryWriter(log_dir = self.log_dir)

    def clear_flightdata(self):
        self.delete_content(self.log_dir+'/*')

    def delete_content(self, path):
        files = glob.glob(path)
        for f in files:
            os.remove(f)

    def record(self, env):
        tgt = env.f16
        aim = env.aim_block['aim1']

        tgt_sim_time = tgt.get_sim_time_sec()
        aim_sim_time = aim.get_sim_time_sec()
        self.writer.add_scalar("sim_time", tgt_sim_time, tgt_sim_time)
        # Altitude 
        self.writer.add_scalar("M_alt", aim.get_altitude(), aim_sim_time)
        self.writer.add_scalar("Tgt_alt", tgt.get_altitude(), tgt_sim_time)

        # velocity 
        self.writer.add_scalar("M_Mach", aim.get_Mach(), aim_sim_time)
        self.writer.add_scalar("Tgt_Mach", tgt.get_Mach(), tgt_sim_time)

        # location 
        self.writer.add_scalar("M_get_lat_gc_deg", aim.get_lat_gc_deg(), aim_sim_time)
        self.writer.add_scalar("M_get_long_gc_deg", aim.get_long_gc_deg(), aim_sim_time)
        self.writer.add_scalar("Tgt_get_lat_gc_deg", tgt.get_lat_gc_deg(), tgt_sim_time)
        self.writer.add_scalar("Tgt_get_long_gc_deg", tgt.get_long_gc_deg(), tgt_sim_time)
        

        #self.writer.add_scalar("M_phi", aim.get_phi(), sim_time)
        #self.writer.add_scalar("M_phi_ref", aim.roll_ref, sim_time)
        #self.writer.add_scalar("M_theta", aim.get_theta(), sim_time)
        #self.writer.add_scalar("M_theta_ref", aim.theta_ref, sim_time)
        
        # direction 
        self.writer.add_scalar("M_psi", aim.get_psi(), aim_sim_time)
        self.writer.add_scalar("Tgt_psi", tgt.get_psi(), tgt_sim_time)
        
        self.writer.add_scalar("Target", aim.position_tgt_NED_norm, aim_sim_time)

        #self.writer.add_scalar("PN_accx", aim.acceleration_cmd_NED[0], sim_time)
        #self.writer.add_scalar("PN_accy", aim.acceleration_cmd_NED[1], sim_time)
        #self.writer.add_scalar("PN_accz", aim.acceleration_cmd_NED[2], sim_time)
        
        self.writer.add_scalar("tgt_east",  aim.tgt_east, aim_sim_time)
        self.writer.add_scalar("tgt_north", aim.tgt_north, aim_sim_time)
        self.writer.add_scalar("tgt_down",  aim.tgt_down, aim_sim_time)
   
class Dog_logs():
    def __init__(self, conf, name = '/DOG', reset_logs = True):
        self.log_dir = conf.logs['log_path'] + name
        if reset_logs:
            self.clear_flightdata()
        
        self.writer = SummaryWriter(log_dir = self.log_dir)

    def clear_flightdata(self):
        self.delete_content(self.log_dir+'/*')

    def delete_content(self, path):
        files = glob.glob(path)
        for f in files:
            os.remove(f)

    def record(self, env):


        sim_time_f16 = env.f16.get_sim_time_sec()
        sim_time_f16r = env.f16r.get_sim_time_sec()

        # location 
        self.writer.add_scalar("f16_lat", env.f16.get_lat_gc_deg(), sim_time_f16)
        self.writer.add_scalar("f16_long", env.f16.get_long_gc_deg(), sim_time_f16)
        self.writer.add_scalar("f16_alt", env.f16.get_altitude(), sim_time_f16r)

        self.writer.add_scalar("f16r_lat", env.f16r.get_lat_gc_deg(), sim_time_f16r)
        self.writer.add_scalar("f16r_long", env.f16r.get_long_gc_deg(), sim_time_f16r)
        self.writer.add_scalar("f16r_alt", env.f16r.get_altitude(), sim_time_f16r)

        if env.aimr_block['aim1r'].active:
            self.writer.add_scalar("aim1r_lat", env.aimr_block['aim1r'].get_lat_gc_deg(), env.aimr_block['aim1r'].get_sim_time_sec())
            self.writer.add_scalar("aim1r_long", env.aimr_block['aim1r'].get_long_gc_deg(), env.aimr_block['aim1r'].get_sim_time_sec())
            self.writer.add_scalar("aim1r_alt", env.aimr_block['aim1r'].get_altitude(), env.aimr_block['aim1r'].get_sim_time_sec())
        else:
            pass

        if env.aimr_block['aim2r'].active:
            self.writer.add_scalar("aim2r_lat", env.aimr_block['aim2r'].get_lat_gc_deg(), env.aimr_block['aim2r'].get_sim_time_sec())
            self.writer.add_scalar("aim2r_long", env.aimr_block['aim2r'].get_long_gc_deg(), env.aimr_block['aim2r'].get_sim_time_sec())
            self.writer.add_scalar("aim2r_alt", env.aimr_block['aim2r'].get_altitude(), env.aimr_block['aim2r'].get_sim_time_sec())
        else:
            pass

        if env.aim_block['aim1'].active:
            self.writer.add_scalar("aim1_lat", env.aim_block['aim1'].get_lat_gc_deg(), env.aim_block['aim1'].get_sim_time_sec())
            self.writer.add_scalar("aim1_long", env.aim_block['aim1'].get_long_gc_deg(), env.aim_block['aim1'].get_sim_time_sec())
            self.writer.add_scalar("aim1_alt", env.aim_block['aim1'].get_altitude(), env.aim_block['aim1'].get_sim_time_sec())
        else:
            pass

        if env.aim_block['aim2'].active:
            self.writer.add_scalar("aim2_lat", env.aim_block['aim2'].get_lat_gc_deg(), env.aim_block['aim2'].get_sim_time_sec())
            self.writer.add_scalar("aim2_long", env.aim_block['aim2'].get_long_gc_deg(), env.aim_block['aim2'].get_sim_time_sec())
            self.writer.add_scalar("aim2_alt", env.aim_block['aim2'].get_altitude(), env.aim_block['aim2'].get_sim_time_sec())
        else:
            pass
        