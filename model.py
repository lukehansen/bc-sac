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
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=4, bias=False),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=3, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=3, bias=False),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(32 * config.img_height//48 * config.img_width//48, 64)
        self.fc2 = nn.Linear(64, config.action_dim)

    def forward(self, state):
        # state: [b, h, w, 3]
        state = state.permute(0, 3, 1, 2)
        x = self.cnn(state) # [b, 32, h, w]
        x = x.flatten(1) # [b, 32 * h * w]
        x = F.relu(self.fc1(x)) # [b, 64]
        means = F.tanh(self.fc2(x)) # [b, action_dim], -1 to 1
        # log_std_devs = self.log_std_dev(x) # [b, action_dim]
        return means #, log_std_devs

class Critic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=4, bias=False),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=3, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=3, bias=False),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(32 * config.img_height//48 * config.img_width//48 + config.action_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, state, action):
        # state: [b, h, w, 3]
        # action: [b, action_dim]
        state = state.permute(0, 3, 1, 2)
        x = self.cnn(state) # [b, 32, h, w]
        x = x.flatten(1) # [b, 32 * h * w]
        x = torch.concat([x, action], dim=-1) # [b, 32 * h * w + action_dim]
        x = F.relu(self.fc1(x)) # [b, 64]
        x = F.relu(self.fc2(x)) # [b, 32]
        q = self.fc3(x) # [b, 1]
        return q.squeeze(1)
