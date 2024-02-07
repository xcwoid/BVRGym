
#PPO code is base on pytorch_minimal_ppo
#@misc{pytorch_minimal_ppo,
#    author = {Barhate, Nikhil},
#    title = {Minimal PyTorch Implementation of Proximal Policy Optimization},
#    year = {2021},
#    publisher = {GitHub},
#    journal = {GitHub repository},
#    howpublished = {\url{https://github.com/nikhilbarhate99/PPO-PyTorch}},
#}


import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std, NN_conf, use_gpu = True):
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        # action dim = 3 / cmd slack throttle
        if NN_conf == 'tanh':
            self.actor =  nn.Sequential(
                    nn.Linear(state_dim, 128),
                    nn.Tanh(),
                    nn.Linear(128, 64),
                    nn.Tanh(),
                    nn.Linear(64, action_dim),
                    nn.Tanh()
                    )
            # critic
            self.critic = nn.Sequential(
                    nn.Linear(state_dim, 128),
                    nn.Tanh(),
                    nn.Linear(128, 64),
                    nn.Tanh(),
                    nn.Linear(64, 1)
                    )
        elif NN_conf == 'relu':
            print('ReLU')
            self.actor =  nn.Sequential(
                    nn.Linear(state_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, action_dim),
                    nn.ReLU()
                    )
            # critic
            self.critic = nn.Sequential(
                    nn.Linear(state_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1)
                    )
            
        self.set_device(use_gpu)

        self.action_var = torch.full((action_dim,), action_std*action_std).to(self.device)

    def set_device(self, use_gpu = False):
        if use_gpu:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"


    def forward(self):
        raise NotImplementedError
    
    def act(self, state, memory, gready):
        action_mean = self.actor(state)
        if not gready:
            cov_mat = torch.diag(self.action_var).to(self.device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(action_logprob)        
            return action.detach()
        
        else:
            return action_mean.detach()

    def evaluate(self, state, action):   
        # action mean, get the 3 actions from NN
        action_mean = self.actor(state)
        #print(action_mean)
        # [0.25, 0.25 , 0.25]. Expand this tesor to same size as other
        action_var = self.action_var.expand_as(action_mean)
        # creates diag matrix 3x3
        #print(self.device)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        action_logprobs = dist.log_prob(action)
        
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        #print('state', state_value)
        return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim, conf_ppo, use_gpu = False):
        self.lr = conf_ppo['lr']
        self.betas = conf_ppo['betas']
        self.gamma = conf_ppo['gamma']
        self.eps_clip = conf_ppo['eps_clip']
        self.K_epochs = conf_ppo['K_epochs']
        action_std = conf_ppo['action_std']
        self.set_device(use_gpu)
        self.policy = ActorCritic(state_dim, action_dim, action_std, NN_conf= conf_ppo['nn_type'], use_gpu= use_gpu).to(self.device)
        self.policy_old = ActorCritic(state_dim, action_dim, action_std, NN_conf= conf_ppo['nn_type'], use_gpu= use_gpu).to(self.device)
    
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr= self.lr, betas= self.betas)            
        self.policy_old.load_state_dict(self.policy.state_dict())
            
        self.MseLoss = nn.MSELoss()

        self.lam_a = conf_ppo['lam_a']
        self.normalize_rewards = conf_ppo['normalize_rewards']

        self.loss_a = 0.0
        self.loss_max = 0.0
        self.loss_min = 0.0
    
    def set_device(self, use_gpu = True, set_policy = False):
        if use_gpu:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"

        if set_policy:
            self.policy.actor.to(self.device)
            self.policy.critic.to(self.device)
            self.policy.action_var.to(self.device)
            self.policy.set_device(self.device)

            self.policy_old.actor.to(self.device)
            self.policy_old.critic.to(self.device)
            self.policy_old.action_var.to(self.device)
            self.policy_old.set_device(self.device)


    def select_action(self, state, memory, gready = False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.policy_old.act(state, memory, gready).cpu().data.numpy().flatten()
    
    def estimate_action(self, state, action):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = torch.FloatTensor(action.reshape(1, -1)).to(self.device)
        return self.policy_old.evaluate(state, action)
      
    def update(self, memory, to_tensor = False, use_gpu = True):
        self.set_device(use_gpu, set_policy=True)

        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        if to_tensor:
            memory.states = [torch.FloatTensor(i.reshape(1, -1)).to(self.device) for i in memory.states]
            memory.actions = [torch.FloatTensor(i.reshape(1, -1)).to(self.device) for i in memory.actions]
            memory.logprobs = [torch.FloatTensor(i.reshape(1, -1)).to(self.device) for i in memory.logprobs]
        
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        if self.normalize_rewards:
            rewards = ((rewards - rewards.mean())/(rewards.std() + 1e-7)).to(self.device)
        
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states).to(self.device), 1).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(self.device), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).to(self.device).detach()
        
        #print(next(self.policy.actor.parameters()).device)
        #print(next(self.policy.critic.parameters()).device)
        #print(old_states.device)
        #print(old_actions.device)
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :

            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            MseLoss = 0.5*self.MseLoss(state_values, rewards)
            loss = -torch.min(surr1, surr2) + MseLoss - 0.01*dist_entropy
            if self.lam_a != 0:
                mu     = torch.squeeze(torch.stack(memory.actions[:-1]).to(self.device), 1).detach()
                mu_nxt = torch.squeeze(torch.stack(memory.actions[1:]).to(self.device), 1).detach()
                loss += 0.5*self.MseLoss(mu_nxt, mu)*self.lam_a
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.loss_a = MseLoss.cpu().data.numpy().flatten()[0]
        self.loss_max = advantages.max().cpu().data.numpy().flatten()[0]
        self.loss_min = advantages.min().cpu().data.numpy().flatten()[0]
                
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        
