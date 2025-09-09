# src/agent_rnn.py
# The file containing your PPO class modified for an RNN policy.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
from src.models_rnn import ActorCriticRNN # Import the new RNN model
import numpy as np

class MemoryRNN:
    """Stores trajectories for RNN training."""
    def __init__(self):
        self.trajectories = []
        self._clear_current_trajectory()

    def store_timestep(self, action, state_img, state_vec, logprob, reward, done, h_in, c_in):
        """Stores a single timestep of experience."""
        self.current_trajectory['actions'].append(action)
        self.current_trajectory['states_img'].append(state_img)
        self.current_trajectory['states_vec'].append(state_vec)
        self.current_trajectory['logprobs'].append(logprob)
        self.current_trajectory['rewards'].append(reward)
        self.current_trajectory['is_terminals'].append(done)
        # Store the hidden state that was INPUT to the RNN for this step
        self.current_trajectory['h_ins'].append(h_in)
        self.current_trajectory['c_ins'].append(c_in)

    def finish_trajectory(self):
        """Marks the end of a trajectory (e.g., at the end of an episode)."""
        if len(self.current_trajectory['actions']) > 0:
            self.trajectories.append(self.current_trajectory)
        self._clear_current_trajectory()

    def _clear_current_trajectory(self):
        self.current_trajectory = {
            'actions': [], 'states_img': [], 'states_vec': [],
            'logprobs': [], 'rewards': [], 'is_terminals': [],
            'h_ins': [], 'c_ins': []
        }

    def clear(self):
        """Clears all stored trajectories."""
        self.trajectories = []
        self._clear_current_trajectory()

    def __len__(self):
        """Returns the total number of timesteps stored."""
        return sum(len(traj['actions']) for traj in self.trajectories)

class PPO_RNN:
    def __init__(self, state_vec_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, device,
                 action_std_init, action_std_decay_rate, min_action_std,
                 rnn_hidden_size=64, rnn_n_layers=1):
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device
        self.action_dim = action_dim
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_n_layers = rnn_n_layers

        self.action_std = action_std_init
        self.action_std_decay_rate = action_std_decay_rate
        self.min_action_std = min_action_std

        self.policy = ActorCriticRNN(state_vec_dim, action_dim, rnn_hidden_size, rnn_n_layers).to(self.device)
        self.optimizer = optim.Adam([
            {'params': self.policy.cnn_base.parameters(), 'lr': lr_actor},
            {'params': self.policy.mlp_base.parameters(), 'lr': lr_actor},
            {'params': self.policy.rnn.parameters(), 'lr': lr_actor}, # Add RNN params
            {'params': self.policy.actor_head.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic_head.parameters(), 'lr': lr_critic},
        ])

        self.policy_old = ActorCriticRNN(state_vec_dim, action_dim, rnn_hidden_size, rnn_n_layers).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def decay_action_std(self):
        print(f"Current action_std: {self.action_std:.4f}")
        self.action_std = max(self.action_std - self.action_std_decay_rate, self.min_action_std)
        print(f"New action_std: {self.action_std:.4f}")
        
    def init_hidden(self, batch_size=1):
        """Initializes the hidden state for the start of an episode."""
        return ActorCriticRNN.init_hidden(batch_size, self.rnn_hidden_size, self.rnn_n_layers, self.device)

    def act(self, state_img, state_vec, hidden_state):
        """Selects an action given the current state and hidden state."""
        # Add a sequence length dimension of 1 for single-step inference
        state_img = state_img.unsqueeze(1)
        state_vec = state_vec.unsqueeze(1)

        action_mean, _, hidden_state_out = self.policy_old.forward(state_img, state_vec, hidden_state)
        action_mean = action_mean.squeeze(1) # Remove sequence dim

        action_var = self.action_std * self.action_std
        cov_mat = torch.diag(torch.full((self.action_dim,), action_var)).unsqueeze(0).to(self.device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach(), hidden_state_out

    def evaluate(self, state_img, state_vec, action, hidden_state):
        """Evaluates actions and state values for a sequence of states."""
        action_mean, state_values, _ = self.policy.forward(state_img, state_vec, hidden_state)
        
        action_var = torch.full_like(action_mean, self.action_std * self.action_std)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        return action_logprobs, torch.squeeze(state_values, -1), dist_entropy

    def update(self, memory):
        policy_losses, value_losses, entropies, values = [], [], [], []

        # Update using all trajectories collected since the last update
        for trajectory in memory.trajectories:
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(trajectory['rewards']), reversed(trajectory['is_terminals'])):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)

            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

            # Convert lists to tensors. Add a batch dim of 1 for the trajectory.
            old_states_img = torch.cat(trajectory['states_img']).unsqueeze(0).detach().to(self.device)
            old_states_vec = torch.cat(trajectory['states_vec']).unsqueeze(0).detach().to(self.device)
            old_actions = torch.cat(trajectory['actions']).unsqueeze(0).detach().to(self.device)
            old_logprobs = torch.cat(trajectory['logprobs']).unsqueeze(0).detach().to(self.device)
            
            # Initial hidden state for this trajectory
            h_in = trajectory['h_ins'][0].detach()
            c_in = trajectory['c_ins'][0].detach()
            initial_hidden_state = (h_in, c_in)

            for _ in range(self.K_epochs):
                logprobs, state_values, dist_entropy = self.evaluate(old_states_img, old_states_vec, old_actions, initial_hidden_state)
                
                logprobs = logprobs.squeeze(0)
                state_values = state_values.squeeze(0)

                ratios = torch.exp(logprobs - old_logprobs.squeeze(0).detach())
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
        
        if not policy_losses: return 0, 0, 0, 0
        return np.mean(policy_losses), np.mean(value_losses), np.mean(entropies), np.mean(values)