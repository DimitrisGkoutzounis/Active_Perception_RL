
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
from src.models import ActorCritic

class Memory:
    def __init__(self):
        self.actions = []
        self.states_img = []
        self.states_vec = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states_img[:]
        del self.states_vec[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class PPO:
    def __init__(self, state_vec_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, device, action_std_init):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device

        self.policy = ActorCritic(state_vec_dim, action_dim, action_std_init).to(self.device)
        self.optimizer = optim.Adam([
            {'params': self.policy.cnn_base.parameters(), 'lr': lr_actor},
            {'params': self.policy.mlp_base.parameters(), 'lr': lr_actor},
            {'params': self.policy.actor_head.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic_head.parameters(), 'lr': lr_critic},
        ])

        self.policy_old = ActorCritic(state_vec_dim, action_dim, action_std_init).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def act(self, state_img, state_vec):
        # action_mean, _ = self.policy_old.forward(state_img, state_vec) #this is self.policy_old.forward instead of self.forward
        action_mean, _ = self.policy.forward(state_img, state_vec)

        
        cov_mat = torch.diag(self.policy.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()

    def evaluate(self, state_img, state_vec, action):
        action_mean, state_values = self.policy.forward(state_img, state_vec)
        
        action_var = self.policy.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        return action_logprobs, torch.squeeze(state_values), dist_entropy

    def update(self, memory):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states_img = torch.cat(memory.states_img, dim=0).detach().to(self.device)
        old_states_vec = torch.squeeze(torch.stack(memory.states_vec, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(memory.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs, dim=0)).detach().to(self.device)
        
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.evaluate(old_states_img, old_states_vec, old_actions)

            # PPO FORMULA
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())