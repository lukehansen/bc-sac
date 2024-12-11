import torch
import copy
from model import Actor, Critic
import itertools
import numpy as np

class ActorCriticAgent:
    def __init__(self, config):
        self.config = config

        # Policy and two Q networks.
        self.pi = Actor(config)
        self.q1 = Critic(config)
        self.q2 = Critic(config)
        self.q_params = itertools.chain(self.q1.parameters(), self.q2.parameters())

        # Each with a separate target network only updated via Polyak smoothing.
        self.pi_target = copy.deepcopy(self.pi)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        for n in [self.pi_target, self.q1_target, self.q2_target]:
            for p in n.parameters():
                p.requires_grad = False

        # Optimizers.
        self.pi_optimizer = torch.optim.Adam(self.pi.parameters(), lr=config.pi_lr)
        self.q_optimizer = torch.optim.Adam(self.q_params, lr=config.q_lr) # only need one optimizer for both q updates

    def update(self, batch):
        # batch: dict of
        #   state: [b, h, w, 3]
        #   new_state: [b, h, w, 3]
        #   action: [b, action_dim]
        #   reward: [b,]
        #   done: [b,]

        # -- Compute Q targets: y = r + gamma * (1-d) * min Qtarget(s', utarget(s'))
        # -- Update Q via minimizing MSE
        # -- Update policy via minimizing Qphi(s, u(s))

        state, action, reward, new_state, done = batch["state"], batch["action"], batch["reward"], batch["new_state"], batch["done"]

        # 1. Update Q.
        self.q_optimizer.zero_grad()
        q1_out = self.q1(state) # [b,]
        q2_out = self.q2(state) # [b,]

        with torch.no_grad():
            pi_target_out = self.pi_target(new_state) # [b, action_dim]
            q1_target_out = self.q1_target(new_state, pi_target_out) # [b,]
            q2_target_out = self.q2_target(new_state, pi_target_out) # [b,]
            min_q = torch.min(q1_target_out, q2_target_out) # [b,]
            bellman_target = reward + self.config.gamma * (1 - done) * min_q # [b,]

        q1_loss = ((q1_out - bellman_target)**2).mean()
        q2_loss = ((q2_out - bellman_target)**2).mean()
        q_loss = q1_loss + q2_loss
        q_loss.backward()
        self.q_optimizer.step()

        # 2. Update Pi. Freeze Q.
        for p in self.q_params:
            p.requires_grad = False
        self.pi_optimizer.zero_grad()
        pi_out = self.pi(state) # [b, action_dim]
        pi_loss = -torch.min(self.q1(state, pi_out), self.q2(state, pi_out))
        pi_loss.backward()
        self.pi_optimizer.step()
        for p in self.q_params:
            p.requires_grad = True

        # 3. Update target networks via Polyak averaging.
        with torch.no_grad():
            params = itertools.chain(self.pi.parameters(), q_params)
            target_params = itertools.chain(self.pi_target.parameters(), self.q1_target.parameters(), self.q2_target.parameters())
            for p, p_targ in zip(params, target_params):
                # In-place as in Spinningup.
                p_targ.data.mul_(self.config.polyak)
                p_targ.data.add_((1 - self.config.polyak) * p.data)

    def act(self, state):
        # state: [h, w, 3]
        state = torch.as_tensor(state, dtype=torch.float32)
        state = state.permute(2, 0, 1).unsqueeze(0) # [1, 3, h, w]
        with torch.no_grad():
            means, _ = self.pi(state) # [b, action_dim]
        # means are [-1, 1]. scale to action space limits.
        means = self.config.action_space.low + (means.numpy() + 1) * (self.config.action_space.high - self.config.action_space.low) / 2
        return means # [b, action_dim]
