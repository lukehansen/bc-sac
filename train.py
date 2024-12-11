import random
import gymnasium as gym
import numpy as np
from config import BcSacConfig
from agent import ActorCriticAgent
from replay_buffer import ReplayBuffer
import torch
import itertools

"""
Training procedure (TD3 for now):
0. Set up Actor, Critic, and target copies.
1. Interact with environment:
    -- Agent outputs u,o for each continuous action
    -- Sample action
    -- Get env update
    -- Store (s, a, r, s', d) into buffer
    -- If s' is terminal, reset env
2. To train:
    -- Sample batch from buffer
    -- Compute Q targets: y = r + gamma * (1-d) * min Qtarget(s', utarget(s'))
    -- Update Q via minimizing MSE
    -- Update policy via minimizing Qphi(s, u(s))
"""

def train():
    env = gym.make("CarRacing-v3", render_mode="human", lap_complete_percent=0.95, domain_randomize=False, continuous=True)
    config = BcSacConfig()
    config.img_height = env.observation_space.shape[0]
    config.img_width = env.observation_space.shape[1]
    config.action_dim = env.action_space.shape[0]
    config.action_space = env.action_space

    replay_buffer = ReplayBuffer(env.observation_space.shape, config.action_dim, config.max_buffer_size)
    agent = ActorCriticAgent(config)

    state, _ = env.reset()
    for step in range(config.num_steps):
        # Interact with environment.
        action = agent.act(state).squeeze(0) # [action_dim]
        next_state, reward, done, trunc, info = env.step(action)
        print("Step: {}, Reward: {}".format(step, reward))

        # Store to buffer.
        replay_buffer.store(state, action, reward, next_state, done)

        state = next_state

        if done or trunc:
            state = env.reset()

        if step >= config.warmup_steps and step % config.update_interval == 0:
            for i in range(config.update_interval):
                batch = replay_buffer.sample_batch(config.batch_size)
                agent.update(batch)

    env.close()

if __name__ == "__main__":
    train()
