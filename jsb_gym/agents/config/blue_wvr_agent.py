from data_classes.agent_dataclass import Agent_parameters
from jsb_gym.simObjects.config import f16_config, AAM_config


agent_parameters = Agent_parameters(
    ammo=4,
    lat=58.0,
    long=18.0,
    alt=7000.0,
    vel=330.0,
    heading=0.0
)

aircraft_simObj_conf = f16_config

missile_simObj_conf = AAM_config

aircraft_name = 'F-16'

agent_name = "RL"

team = "Blue"