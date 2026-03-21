import os
import pandas as pd
import shutil
from pathlib import Path

class TacviewLogger:
    def __init__(self, env):
        self.tacview_output_dir = env.conf.tacview_output_dir
        #os.makedirs(self.tacview_output_dir, exist_ok=True)
        self.env = env
        self.data_logs = {}
        for i in self.env.all_agents:
            self.data_logs[i.conf.agent_name] = {}
            self.data_logs[i.conf.agent_name] = {'name':i.conf.agent_name, 'aircraft_name': i.conf.aircraft_name, 'team': i.conf.team}
            self.data_logs[i.conf.agent_name][i.conf.agent_name] = []
            self.data_logs[i.conf.agent_name]['ammo'] = {}
            for j in i.ammo.keys():
                self.data_logs[i.conf.agent_name]['ammo'][j] = []

    def log_flight_data(self):

        for i in self.env.all_agents:
            self.data_logs[i.conf.agent_name][i.conf.agent_name].append(self.get_current_data(i))
            for j in i.ammo.keys():
                if i.ammo[j].is_active():
                    self.data_logs[i.conf.agent_name]['ammo'][j].append(self.get_current_missile_data(i.ammo[j], i))


    def get_current_data(self, agent):
        log = {
        "Time": agent.simObj.get_sim_time_sec(),
        "Longitude": agent.simObj.get_long_gc_deg(),
        "Latitude": agent.simObj.get_lat_gc_deg(),
        "Altitude": agent.simObj.get_altitude(),
        "Roll (deg)": agent.simObj.get_phi(),
        "Pitch (deg)": agent.simObj.get_theta(),
        "Yaw (deg)": agent.simObj.get_psi(),
        }
        return log

    def get_current_missile_data(self, missile, agent):
        log = {
        "Time": agent.simObj.get_sim_time_sec(),
        "Longitude": missile.get_long_gc_deg(),
        "Latitude": missile.get_lat_gc_deg(),
        "Altitude": missile.get_altitude(),
        "Roll (deg)": missile.get_phi(),
        "Pitch (deg)": missile.get_theta(),
        "Yaw (deg)": missile.get_psi(),
        }
        return log


    def save_logs(self):
        self.mkdir_logs()
        for i in self.data_logs:
            pd.DataFrame(self.data_logs[i][i]).to_csv(f"{self.tacview_output_dir}/{self.data_logs[i]['aircraft_name']} ({self.data_logs[i]['name']}) [{self.data_logs[i]['team']}].csv", index=False)
            for j in self.data_logs[i]['ammo']:
                pd.DataFrame(self.data_logs[i]['ammo'][j]).to_csv(f"{self.tacview_output_dir}/AIM-120 AMRAAM (AIM{j}) [{self.data_logs[i]['team']}].csv", index=False)
              
    def mkdir_logs(self):
        folder = Path(self.tacview_output_dir)
        shutil.rmtree(folder, ignore_errors=True)  # delete if it exists
        folder.mkdir(parents=True, exist_ok=True)  # recreate it
