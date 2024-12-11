import torch
import copy
from torch import nn
import torch.nn.functional as F

# Start with DDPG and then extend to SAC.
# Actor policy: obs -> mean, std dev for each action
# Critic: Q-function taking s, a -> rew
class Actor(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Let's do a simple ConvNet for now.
        self.conv1 = nn.Conv2d(3, config.hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(config.hidden_dim, config.hidden_dim, 3, padding=1)
        self.mean = nn.Linear(config.hidden_dim * config.img_height * config.img_width, config.action_dim)
        self.log_std_dev = nn.Linear(config.hidden_dim * config.img_height * config.img_width, config.action_dim)

    def forward(self, state):
        # state: [b, 3, h, w]
        x = F.relu(self.conv1(state)) # [b, hidden_dim, h, w]
        x = F.relu(self.conv2(x)) # [b, hidden_dim, h, w]
        x = x.flatten(1) # [b, hidden_dim * h * w]
        means = F.tanh(self.mean(x)) # [b, action_dim], -1 to 1
        log_std_devs = self.log_std_dev(x) # [b, action_dim]
        return means, log_std_devs

class Critic(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Let's do a simple ConvNet for now.
        self.conv1 = nn.Conv2d(3, config.hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(config.hidden_dim, config.hidden_dim, 3, padding=1)
        self.fc = nn.Linear(config.hidden_dim * config.img_height * config.img_width + config.action_dim, 1)
    
    def forward(self, state, action):
        # state: [b, 3, h, w]
        # action: [b, action_dim]
        x = F.relu(self.conv1(state)) # [b, hidden_dim, h, w]
        x = F.relu(self.conv2(x)) # [b, hidden_dim, h, w]
        x = x.flatten(1) # [b, hidden_dim * h * w]
        x = torch.concat([x, action], dim=-1) # [b, hidden_dim * h * w + action_dim]
        q = self.fc(x)
        return q
