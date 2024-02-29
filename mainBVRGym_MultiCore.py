import argparse, time
from jsb_gym.environmets import evasive, bvrdog
import numpy as np
from enum import Enum
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import multiprocessing
from jsb_gym.RL.ppo import Memory, PPO
from jsb_gym.environmets.config import BVRGym_PPO1, BVRGym_PPO2, BVRGym_PPODog
from numpy.random import seed
from jsb_gym.TAU.config import aim_evs_BVRGym, f16_evs_BVRGym, aim_dog_BVRGym, f16_dog_BVRGym

def init_pool():
    seed()

class Maneuvers(Enum):
    Evasive = 0
    Crank = 1

def runPPO(args):
    if args['track'] == 'M1':
        from jsb_gym.RL.config.ppo_evs_PPO1 import conf_ppo
        
        env = evasive.Evasive(BVRGym_PPO1, args, aim_evs_BVRGym, f16_evs_BVRGym)
        torch_save = 'jsb_gym/logs/RL/M1.pth'
        state_scale = 1
    elif args['track'] == 'M2':
        from jsb_gym.RL.config.ppo_evs_PPO2 import conf_ppo
        env = evasive.Evasive(BVRGym_PPO2, args, aim_evs_BVRGym, f16_evs_BVRGym)
        torch_save = 'jsb_gym/logs/RL/M2.pth'
        state_scale = 2
    elif args['track'] == 'Dog':
        from jsb_gym.RL.config.ppo_evs_PPO_BVRDog import conf_ppo
        env = bvrdog.BVRDog(BVRGym_PPODog, args, aim_dog_BVRGym, f16_dog_BVRGym)
        torch_save = 'jsb_gym/logs/RL/Dog/'
        state_scale = 1
    elif args['track'] == 'DogR':
        from jsb_gym.RL.config.ppo_evs_PPO_BVRDog import conf_ppo
        env = bvrdog.BVRDog(BVRGym_PPODog, args, aim_dog_BVRGym, f16_dog_BVRGym)
        torch_save = 'jsb_gym/logs/RL/DogR.pth'
        state_scale = 1

    
    writer = SummaryWriter('runs/' + args['track'] )
    state_dim = env.observation_space
    action_dim = env.action_space.shape[1]
    memory = Memory()
    ppo = PPO(state_dim*state_scale, action_dim, conf_ppo, use_gpu = False)    
    pool = multiprocessing.Pool(processes=int(args['cpu_cores']), initializer=init_pool)
    
    for i_episode in range(1, args['Eps']+1):
        ppo_policy = ppo.policy.state_dict()    
        ppo_policy_old = ppo.policy_old.state_dict()
        input_data = [(args, ppo_policy, ppo_policy_old, conf_ppo, state_scale )for _ in range(args['cpu_cores'])]
        running_rewards = []
        tb_obs = []
        
        results = pool.map(train, input_data)
        for idx, tmp in enumerate(results):
            #print(tmp[5])
            #writer.add_scalar("running_rewards" + str(idx), tmp[5], i_episode)
            memory.actions.extend(tmp[0])
            memory.states.extend(tmp[1])
            memory.logprobs.extend(tmp[2])
            memory.rewards.extend(tmp[3])
            memory.is_terminals.extend(tmp[4])
            running_rewards.append(tmp[5])
            tb_obs.append(tmp[6])
            
        ppo.set_device(use_gpu=True)
        ppo.update(memory, to_tensor=True, use_gpu= True)
        memory.clear_memory()
        ppo.set_device(use_gpu=False)
        torch.cuda.empty_cache()
        
        writer.add_scalar("running_rewards", sum(running_rewards)/len(running_rewards), i_episode)
        tb_obs0 = None
        for i in tb_obs:
            #print(i)
            if tb_obs0 == None:
                tb_obs0 = i
            else:
                for key in tb_obs0:
                    tb_obs0[key] += i[key]

        nr = len(tb_obs)
        for key in tb_obs0:
            tb_obs0[key] = tb_obs0[key]/nr
            writer.add_scalar(key, tb_obs0[key], i_episode)
        if i_episode % 500 == 0:
            # save 
            torch.save(ppo.policy.state_dict(), torch_save + 'Dog'+str(i_episode) + '.pth')

    pool.close()
    pool.join()
    #torch.save(ppo.policy.state_dict(), torch_save)

def train(args):
    if args[0]['track'] == 'M1':
        env = evasive.Evasive(BVRGym_PPO1, args[0], aim_evs_BVRGym, f16_evs_BVRGym)
    elif args[0]['track'] == 'M2':
        env = evasive.Evasive(BVRGym_PPO2, args[0], aim_evs_BVRGym, f16_evs_BVRGym)
    elif args[0]['track'] == 'Dog':
        env = bvrdog.BVRDog(BVRGym_PPODog, args[0], aim_dog_BVRGym, f16_dog_BVRGym)
    elif args[0]['track'] == 'DogR':
        env = bvrdog.BVRDog(BVRGym_PPODog, args[0], aim_dog_BVRGym, f16_dog_BVRGym)


    maneuver = Maneuvers.Evasive
    memory = Memory()
    state_dim = env.observation_space
    action_dim = env.action_space.shape[1]
    ppo = PPO(state_dim* args[4], action_dim, args[3], use_gpu=False)

    ppo.policy.load_state_dict(args[1])
    ppo.policy_old.load_state_dict(args[2])

    ppo.policy.eval()
    ppo.policy_old.eval()
    running_reward = 0.0

    for i_episode in range(1, args[0]['eps']+1):
        action = np.zeros(3)
        # using string comparison, not the best, that is why I am keeping it short for now
        if args[0]['track'] == 'M1':
            state_block = env.reset(True, True)
            state = state_block['aim1']
            # max thrust 
            action[2] = 1

        elif args[0]['track'] == 'M2':
            state_block = env.reset(True, True)
            state = np.concatenate((state_block['aim1'][0], state_block['aim2'][0]))
            # max thrust 
            action[2] = 1
        
        elif args[0]['track'] == 'Dog':
            state = env.reset()
            # If you activate the afterburner, both aircraft will fall from the sky after 10 min  
            action[2] = 0.0
        
        elif args[0]['track'] == 'DogR':
            state = env.reset()
            # If you activate the afterburner, both aircraft will fall from the sky after 10 min  
            action[2] = 0.0
        
        done = False
        while not done:
            # heading [-1, 1] altitude [-1, 1] thrust [-1, 1]
            act = ppo.select_action(state, memory)
            action[0] = act[0]
            action[1] = act[1]

            if args[0]['track'] == 'M1':
                state_block, reward, done, _ = env.step(action, action_type= maneuver.value)
                state = state_block['aim1']
            
            elif args[0]['track'] == 'M2':
                state_block, reward, done, _ = env.step(action, action_type= maneuver.value)
                state = np.concatenate((state_block['aim1'], state_block['aim2']))
            
            elif args[0]['track'] == 'Dog':
                state, reward, done, _ = env.step(action, action_type= maneuver.value, blue_armed= True, red_armed= True)

            elif args[0]['track'] == 'DogR':
                state, reward, done, _ = env.step(action, action_type= maneuver.value, blue_armed= False, red_armed= True)


            memory.rewards.append(reward)
            memory.is_terminals.append(done)

        running_reward += reward 
    

    running_reward = running_reward/args[0]['eps']
    # tensorboard 
    if args[0]['track'] == 'M1':
        tb_obs = {}
    elif args[0]['track'] == 'M2':
        tb_obs = {}            
    elif args[0]['track'] == 'Dog':
        tb_obs = get_tb_obs_dog(env)
    elif args[0]['track'] == 'DogR':
        tb_obs = get_tb_obs_dog(env)




    actions = [i.detach().numpy() for i in memory.actions]
    states = [i.detach().numpy() for i in memory.states]
    logprobs = [i.detach().numpy() for i in memory.logprobs]
    rewards = [i for i in memory.rewards]
    #print(rewards)
    is_terminals = [i for i in memory.is_terminals]     
    return [actions, states, logprobs, rewards, is_terminals, running_reward, tb_obs]


def get_tb_obs_dog(env):
    tb_obs = {}
    tb_obs['Blue_ground'] = env.reward_f16_hit_ground
    tb_obs['Red_ground'] = env.reward_f16r_hit_ground
    tb_obs['maxTime'] = env.reward_max_time

    tb_obs['Blue_alive'] = env.f16_alive
    tb_obs['Red_alive'] = env.f16r_alive


    tb_obs['aim1_active'] = env.aim_block['aim1'].active
    tb_obs['aim1_alive'] = env.aim_block['aim1'].alive
    tb_obs['aim1_target_lost'] = env.aim_block['aim1'].target_lost
    tb_obs['aim1_target_hit'] = env.aim_block['aim1'].target_hit


    tb_obs['aim2_active'] = env.aim_block['aim2'].active
    tb_obs['aim2_alive'] = env.aim_block['aim2'].alive
    tb_obs['aim2_target_lost'] = env.aim_block['aim2'].target_lost
    tb_obs['aim2_target_hit'] = env.aim_block['aim2'].target_hit

    tb_obs['aim1r_active'] = env.aimr_block['aim1r'].active
    tb_obs['aim1r_alive'] = env.aimr_block['aim1r'].alive
    tb_obs['aim1r_target_lost'] = env.aimr_block['aim1r'].target_lost
    tb_obs['aim1r_target_hit'] = env.aimr_block['aim1r'].target_hit

    tb_obs['aim2r_active'] = env.aimr_block['aim2r'].active
    tb_obs['aim2r_alive'] = env.aimr_block['aim2r'].alive
    tb_obs['aim2r_target_lost'] = env.aimr_block['aim2r'].target_lost
    tb_obs['aim2r_target_hit'] = env.aimr_block['aim2r'].target_hit


    if env.aim_block['aim1'].target_lost:
        tb_obs['aim1_MD'] = env.aim_block['aim1'].position_tgt_NED_norm
        
    if env.aim_block['aim2'].target_lost:
        tb_obs['aim2_lost'] = 1
        tb_obs['aim2_MD'] = env.aim_block['aim2'].position_tgt_NED_norm
    
    if env.aimr_block['aim1r'].target_lost:
        tb_obs['aim1r_lost'] = 1
        tb_obs['aim1r_MD'] = env.aimr_block['aim1r'].position_tgt_NED_norm
    
    if env.aimr_block['aim2r'].target_lost:
        tb_obs['aim2r_lost'] = 1
        tb_obs['aim2r_MD'] = env.aimr_block['aim2r'].position_tgt_NED_norm

    return tb_obs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vizualize", action='store_true', help="Render in FG")
    parser.add_argument("-track", "--track", type = str, help="Tracks: M1, M2, Dog, DogR", default=' ')
    parser.add_argument("-cpus", "--cpu_cores", type = int, help="Nuber of cores to use", default= None)
    parser.add_argument("-Eps", "--Eps", type = int, help="Nuber of cores to use", default= int(1e3))
    parser.add_argument("-eps", "--eps", type = int, help="Nuber of cores to use", default= 5)
    #parser.add_argument("-seed", "--seed", type = int, help="radnom seed", default= None)
    args = vars(parser.parse_args())

    #if args['seed'] != None:
    #    torch.manual_seed(args['seed'])
    #    np.random.seed(args['seed'])
    
    runPPO(args)

# training: 
# python mainBVRGym_MultiCore.py -track M1  -cpus 10 -Eps 100000 -eps 1
# python mainBVRGym_MultiCore.py -track M2  -cpus 10 -Eps 100000 -eps 1
# python mainBVRGym_MultiCore.py -track Dog -cpus 10 -Eps 10000 -eps 1

