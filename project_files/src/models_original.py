# src/models.py

import torch
import torch.nn as nn
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_vec_dim, action_dim, action_std_init=0.5):
        super(ActorCritic, self).__init__()
        
        # --- CNN Branch for Image Input ---
        self.cnn_base = nn.Sequential(
            # Input: [batch_size, 1, 30, 30] from config.BINS
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1), # -> [16, 30, 30]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # -> [16, 15, 15]
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # -> [32, 15, 15]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # -> [32, 7, 7]
            nn.Flatten(),
        )
        # Calculate the flattened size automatically
        # Flattened size will be 32 * 7 * 7 = 1568
        cnn_output_size = self._get_conv_output_size((1, 30, 30))

        self.mlp_base = nn.Sequential(
            nn.Linear(state_vec_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        
        # Concatenated feature size
        combined_feature_size = cnn_output_size + 32

        # --- Actor Head ---
        self.actor_head = nn.Sequential(
            nn.Linear(combined_feature_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh() 
        )

        # --- Critic Head ---
        self.critic_head = nn.Sequential(
            nn.Linear(combined_feature_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.action_var = nn.Parameter(torch.full((action_dim,), action_std_init * action_std_init))

    def forward(self, state_img, state_vec):
        img_features = self.cnn_base(state_img)
        vec_features = self.mlp_base(state_vec)
        combined_features = torch.cat((img_features, vec_features), dim=1)
        
        action_mean = self.actor_head(combined_features)
        state_value = self.critic_head(combined_features)
        
        return action_mean, state_value

    def get_action_mean(self, state_img, state_vec):
        img_features = self.cnn_base(state_img)
        vec_features = self.mlp_base(state_vec)
        combined_features = torch.cat((img_features, vec_features), dim=1)
        action_mean = self.actor_head(combined_features)
        return action_mean.detach()
    
    def _get_conv_output_size(self, shape):
        
        with torch.no_grad():
            o = self.cnn_base(torch.zeros(1, *shape))
        return int(np.prod(o.size()))