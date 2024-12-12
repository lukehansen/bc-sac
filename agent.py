import torch
from torch import nn
import copy
from model import Actor, Critic
import itertools
import numpy as np

class ActorCriticAgent(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Policy and two Q networks.
        self.pi = Actor(config)
        print("Actor params: {}".format(sum([p.numel() for p in self.pi.parameters() if p.requires_grad])))
        self.q1 = Critic(config)
        print("Critic params: {}".format(sum([p.numel() for p in self.q1.parameters() if p.requires_grad])))
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

    def update(self, batch, step, writer=None):
        # batch: dict of {
        #   state: [b, h, w, 3]
        #   new_state: [b, h, w, 3]
        #   action: [b, action_dim]
        #   reward: [b,]
        #   done: [b,]
        # }

        state, action, reward, new_state, done = batch["state"], batch["action"], batch["reward"], batch["new_state"], batch["done"]

        # 1. Update Q.
        self.q_optimizer.zero_grad()
        q1_out = self.q1(state, action) # [b,]
        q2_out = self.q2(state, action) # [b,]

        if writer:
            q_avg = torch.stack([q1_out.detach(), q2_out.detach()]).mean().item()
            print("Train QAvg: {}".format(q_avg))
            writer.add_scalar("Train QAvg", q_avg, step)

        with torch.no_grad():
            pi_target_out = self.pi_target(new_state) # [b, action_dim]
            q1_target_out = self.q1_target(new_state, pi_target_out) # [b,]
            q2_target_out = self.q2_target(new_state, pi_target_out) # [b,]
            min_q = torch.min(q1_target_out, q2_target_out) # [b,]
            bellman_target = reward + self.config.gamma * (1 - done) * min_q # [b,]

        q1_loss = ((q1_out - bellman_target)**2).mean()
        q2_loss = ((q2_out - bellman_target)**2).mean()
        q_loss = (q1_loss + q2_loss) / 2
        if writer:
            ql = q_loss.detach()
            print("Train QLoss: {}".format(ql))
            writer.add_scalar("Train QLoss", ql.item(), step)
        q_loss.backward()
        self.q_optimizer.step()

        # 2. Update Pi. Freeze Q.
        for p in self.q_params:
            p.requires_grad = False
        self.pi_optimizer.zero_grad()
        pi_out = self.pi(state) # [b, action_dim]
        pi_loss = -torch.min(self.q1(state, pi_out), self.q2(state, pi_out)).mean()
        if writer:
            print("Train PiLoss: {}".format(pi_loss))
            writer.add_scalar("Train PiLoss", pi_loss.detach().item(), step)
        pi_loss.backward()
        self.pi_optimizer.step()
        for p in self.q_params:
            p.requires_grad = True

        # 3. Update target networks via Polyak averaging.
        with torch.no_grad():
            params = itertools.chain(self.pi.parameters(), self.q_params)
            target_params = itertools.chain(self.pi_target.parameters(), self.q1_target.parameters(), self.q2_target.parameters())
            for p, p_targ in zip(params, target_params):
                # In-place as in Spinningup.
                p_targ.data.mul_(self.config.polyak)
                p_targ.data.add_((1 - self.config.polyak) * p.data)

    def act(self, state, noise=0):
        # state: [h, w, 3]
        state = torch.as_tensor(state, dtype=torch.float32).to(self.config.device)
        state = state.unsqueeze(0) # [1, h, w, 3]
        with torch.no_grad():
            means = self.pi(state).cpu().numpy() # [b, action_dim]
        # means are [-1, 1]. add noise, scale, and clip.
        print("Means: {}".format(means))
        means = means + noise * np.random.randn(self.config.action_dim)
        means = self.config.action_space.low + (means + 1) * (self.config.action_space.high - self.config.action_space.low) / 2
        means = np.clip(means, self.config.action_space.low, self.config.action_space.high)
        return means # [b, action_dim]
