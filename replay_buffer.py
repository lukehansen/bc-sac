import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, config):
        self.config = config
        state_dim = config.observation_space.shape
        self.state_buf = np.zeros((config.max_buffer_size, *state_dim), dtype=np.float32)
        self.new_state_buf = np.zeros((config.max_buffer_size, *state_dim), dtype=np.float32)
        self.action_buf = np.zeros((config.max_buffer_size, config.action_dim), dtype=np.float32)
        self.reward_buf = np.zeros(config.max_buffer_size, dtype=np.float32)
        self.done_buf = np.zeros(config.max_buffer_size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, config.max_buffer_size

    def store(self, state, action, reward, new_state, done):
        self.state_buf[self.ptr] = state
        self.new_state_buf[self.ptr] = new_state
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(state=self.state_buf[idxs],
                     new_state=self.new_state_buf[idxs],
                     action=self.action_buf[idxs],
                     reward=self.reward_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32).to(config.device) for k,v in batch.items()}
