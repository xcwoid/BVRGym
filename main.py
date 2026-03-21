from jsb_gym.envs.BaseEnv import BVRBase
from jsb_gym.envs.config import baseEnv_conf
from jsb_gym.envs.WvrEnv import WvrEnv_ControlZone, WvrEnv_AggressiveShooter, WvrEnv_ConservativeShooter

from stable_baselines3.common.env_util import make_vec_env
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO


class GymCallback(BaseCallback):
    def __init__(self, name):
        super().__init__()
        self.writer = SummaryWriter("runs/" + name)

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if info["done"]:
                self.writer.add_scalar("env/trunk", info["trunk"], global_step=self.num_timesteps)
        return True


def train_BVRBase():
    vec_env = make_vec_env(BVRBase, n_envs=14, vec_env_cls=SubprocVecEnv, env_kwargs={'conf': baseEnv_conf})        
    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="runs")
    model.learn(total_timesteps=2_000_000, callback=GymCallback("BVRBase"))

    model.save("trained/BVRBase_PPO_2M", tb_log_name="BVR_air_combat_RLvsBT")

def train_WvrControlZone():
    vec_env = make_vec_env(WvrEnv_ControlZone, n_envs=14, vec_env_cls=SubprocVecEnv, env_kwargs={'conf': baseEnv_conf})        
    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="runs")
    model.learn(total_timesteps=2_000_000, callback=GymCallback("WVR_ControlZone"), tb_log_name="ControlZone_RLvsRandy")

    model.save("trained/WvrControlZone_PPO_2M")

def train_WvrEnv_AggressiveShooter():
    vec_env = make_vec_env(WvrEnv_AggressiveShooter, n_envs=14, vec_env_cls=SubprocVecEnv, env_kwargs={'conf': baseEnv_conf})        
    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="runs")
    model.learn(total_timesteps=2_000_000, callback=GymCallback("WvrEnv_AggressiveShooter"), tb_log_name="AggressiveShooter_RLvsRandy")

    model.save("trained/WvrEnv_AggressiveShooter_PPO_2M")

def train_WvrEnv_ConservativeShooter():
    vec_env = make_vec_env(WvrEnv_ConservativeShooter, n_envs=14, vec_env_cls=SubprocVecEnv, env_kwargs={'conf': baseEnv_conf})        
    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="runs")
    model.learn(total_timesteps=2_000_000, callback=GymCallback("WvrEnv_ConservativeShooter"), tb_log_name="ConservativeShooter_RLvsRandy")

    model.save("trained/WvrEnv_ConservativeShooter_PPO_2M")


def run_trainedWvrModel(model_path, env):
    baseEnv_conf.tacview_output_dir = 'data_output/run_wvr/'
    env=env(baseEnv_conf)
    model = PPO.load(model_path)
    obs, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.log_tacview()
    print(f"View data in Tacview in {baseEnv_conf.tacview_output_dir}")


if __name__ == "__main__":
    train_BVRBase()
    #train_WvrControlZone()
    #train_WvrEnv_AggressiveShooter()
    #train_WvrEnv_ConservativeShooter()
    #run_trainedWvrModel("trained/WvrControlZone_PPO_2M.zip", WvrEnv_ControlZone)