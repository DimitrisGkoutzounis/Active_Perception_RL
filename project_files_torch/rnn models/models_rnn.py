# src/models_rnn.py
# Contains the definition of the Actor-Critic RNN model.

import torch
import torch.nn as nn
import numpy as np

class ActorCriticRNN(nn.Module):
    def __init__(self, state_vec_dim, action_dim, rnn_hidden_size=64, n_layers=1):
        
        super(ActorCriticRNN, self).__init__()
        
        # --- CNN Branch for Image Input (Same as before) ---
        self.cnn_base = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )
        cnn_output_size = 32 * 7 * 7  # 1568

        # --- MLP Branch for Vector Input (Same as before) ---
        self.mlp_base = nn.Sequential(
            nn.Linear(state_vec_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        
        # --- RNN Layer ---
        combined_feature_size = cnn_output_size + 32
        self.rnn = nn.LSTM(combined_feature_size, rnn_hidden_size, n_layers, batch_first=True)
        
        # --- Actor Head ---
        self.actor_head = nn.Sequential(
            nn.Linear(rnn_hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh() 
        )

        # --- Critic Head ---
        self.critic_head = nn.Sequential(
            nn.Linear(rnn_hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state_img, state_vec, hidden_state):
       
        batch_size = state_img.size(0)
        seq_len = state_img.size(1)
        
        # Process image and vector inputs for the entire sequence at once
        img_features = self.cnn_base(state_img.view(batch_size * seq_len, 1, 30, 30))
        vec_features = self.mlp_base(state_vec.view(batch_size * seq_len, -1))
        
        combined_features = torch.cat((img_features, vec_features), dim=1)
        
        # Reshape for the RNN layer
        rnn_input = combined_features.view(batch_size, seq_len, -1)
        
        # Pass combined features through the LSTM
        rnn_out, hidden_state_out = self.rnn(rnn_input, hidden_state)
        
        # Pass the LSTM's output to the actor and critic heads
        action_mean = self.actor_head(rnn_out)
        state_value = self.critic_head(rnn_out)
        
        return action_mean, state_value, hidden_state_out

    @staticmethod
    def init_hidden(batch_size, rnn_hidden_size, n_layers, device):
        """Helper function to create initial hidden states for the LSTM."""
        h_0 = torch.zeros(n_layers, batch_size, rnn_hidden_size).to(device)
        c_0 = torch.zeros(n_layers, batch_size, rnn_hidden_size).to(device)
        return (h_0, c_0)