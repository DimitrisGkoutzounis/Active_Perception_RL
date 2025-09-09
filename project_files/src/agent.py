# The file containing your PPO class (e.g., src/agent.py)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
from src.models import ActorCritic
import numpy as np

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
    def __init__(self, state_vec_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, device,
                 action_std_init, action_std_decay_rate, min_action_std):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device
        self.action_dim = action_dim # Store action_dim

        self.action_std = action_std_init
        self.action_std_decay_rate = action_std_decay_rate
        self.min_action_std = min_action_std

        self.policy = ActorCritic(state_vec_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam([
            {'params': self.policy.cnn_base.parameters(), 'lr': lr_actor},
            {'params': self.policy.mlp_base.parameters(), 'lr': lr_actor},
            {'params': self.policy.actor_head.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic_head.parameters(), 'lr': lr_critic},
        ])

        self.policy_old = ActorCritic(state_vec_dim, action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def decay_action_std(self):
        """Linearly decays the action standard deviation."""
        print(f"Current action_std: {self.action_std:.4f}")
        self.action_std = max(self.action_std - self.action_std_decay_rate, self.min_action_std)
        print(f"New action_std: {self.action_std:.4f}")

    def act(self, state_img, state_vec):
        action_mean, _ = self.policy_old.forward(state_img, state_vec)

        action_var = self.action_std * self.action_std
        cov_mat = torch.diag(torch.full((self.action_dim,), action_var)).unsqueeze(0).to(self.device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()

    def evaluate(self, state_img, state_vec, action):
        action_mean, state_values = self.policy.forward(state_img, state_vec)
        
        action_var = torch.full_like(action_mean, self.action_std * self.action_std)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        
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
        
        policy_losses, value_losses, entropies, values = [], [], [], []

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.evaluate(old_states_img, old_states_vec, old_actions)

            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * self.MseLoss(state_values, rewards)
            entropy_loss = -0.01 * dist_entropy.mean()
            loss = policy_loss + value_loss + entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(dist_entropy.mean().item())
            values.append(state_values.mean().item())

        self.policy_old.load_state_dict(self.policy.state_dict())
        return np.mean(policy_losses), np.mean(value_losses), np.mean(entropies), np.mean(values)