from jsb_gym.envs.BaseEnv import BVREnv_BTvsBT
from jsb_gym.envs.config import baseEnv_conf

baseEnv_conf.tacview_output_dir = 'data_output/test_BTvsBT/'
baseEnv_conf.max_episode_time = 60*20
env = BVREnv_BTvsBT(baseEnv_conf)
obs = env.reset()

done = False
while not done:
    action = env.action_space.sample()  # Sample a random action
    obs, reward, done, trunk, info = env.step(action)
    env.log_tacview()
print(info)
print(f"View data in Tacview in {baseEnv_conf.tacview_output_dir}")

