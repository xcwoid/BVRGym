from jsb_gym.envs.WvrEnv import WvrEnv_ControlZone
from jsb_gym.envs.config import baseEnv_conf

baseEnv_conf.tacview_output_dir = 'data_output/test_wvr/'
baseEnv_conf.max_episode_time = 60*5


env = WvrEnv_ControlZone(baseEnv_conf)

obs = env.reset()

done = False
while not done:
    action = env.action_space.sample()  # Sample a random action
    obs, reward, done, trunk, info = env.step(action)
    print(reward)
    env.log_tacview()
print(info)
print(f"View data in Tacview in {baseEnv_conf.tacview_output_dir}")

