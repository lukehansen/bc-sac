class BcSacConfig():
    def __init__(self):
        # RL
        self.num_epochs = 100
        self.steps_per_epoch = 4000
        self.warmup_steps = 1000
        self.prefill_steps = 1000 # Fill the buffer before training.
        self.update_interval = 50 # Alternate 50 steps env interaction, 50 steps training.
        self.max_buffer_size = 10_000
        self.gamma = 0.99
        self.polyak = 0.995
        self.action_noise = 0.1 # Not needed for SAC
        self.max_episode_len = 1000

        # NN
        self.pi_lr = 1e-5
        self.q_lr = 2e-3
        self.batch_size = 64

