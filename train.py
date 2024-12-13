import random
from datetime import datetime
import os
import argparse
import gymnasium as gym
import numpy as np
from config import BcSacConfig
from agent import ActorCriticAgent
from replay_buffer import ReplayBuffer
import torch
import itertools
from torch.utils.tensorboard import SummaryWriter

CHECKPOINT_DIR = "checkpoints/"
LOG_DIR = "logs/"

def train(args):
    env = gym.make("CarRacing-v3", render_mode="human", lap_complete_percent=0.95, domain_randomize=False, continuous=True)
    config = BcSacConfig()
    config.img_height = env.observation_space.shape[0]
    config.img_width = env.observation_space.shape[1]
    config.observation_space = env.observation_space
    config.action_dim = 2 # env.action_space.shape[0]. Restricting accel/brake to a single axis. -1 is full brake, 1 is full gas.
    config.action_space = env.action_space
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(config.device))

    if args.toy:
        config.prefill_steps = 10
        config.update_interval = 10
        config.warmup_steps = 5000
        config.action_noise = 0.1
        config.polyak = 0.6
        config.gamma = 0.1

    agent = ActorCriticAgent(config).to(config.device)
    if args.resume_run_id:
        run_id = args.resume_run_id

        # Get latest checkpoint file.
        checkpoint_dir = os.path.join(CHECKPOINT_DIR, args.resume_run_id)
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("model_epoch") and f.endswith(".pt")]
        epochs = [int(f.split("model_epoch")[1].split(".pt")[0]) for f in checkpoint_files]
        if not epochs:
            print("No checkpoints found.")
            exit(1)
        latest_epoch = max(epochs)
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch{latest_epoch}.pt")
        print(f"Resuming training from {checkpoint_path}")
        
        # Load model and optimizer.
        checkpoint = torch.load(checkpoint_path)
        agent.load_state_dict(checkpoint['model_state_dict'])
        agent.pi_optimizer.load_state_dict(checkpoint['pi_optimizer_state_dict'])
        agent.q_optimizer.load_state_dict(checkpoint['q_optimizer_state_dict'])
        starting_epoch = checkpoint['epoch'] + 1 # saved out at the end so we should start from the next one
    else:
        starting_epoch = 0
        run_id = "debug" if args.toy else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        checkpoint_dir = os.path.join(CHECKPOINT_DIR, run_id)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    log_dir = os.path.join(LOG_DIR, run_id)
    writer = SummaryWriter(log_dir=log_dir)

    replay_buffer = ReplayBuffer(config)

    state, _ = env.reset()
    state = state / 255.0
    episode_reward = 0
    episode_len = 0
    last_actions = np.zeros([10, config.action_dim])
    print("Starting training at epoch: {}".format(starting_epoch))
    for epoch in range(starting_epoch, config.num_epochs):
        for step_idx in range(config.steps_per_epoch):
            step = epoch * config.steps_per_epoch + step_idx
            # Interact with environment.
            if step > config.warmup_steps:
                raw_action, env_action = agent.act(state, noise_factor=config.action_noise) # [b, 2], [b, 3]
            else:
                raw_action = np.random.uniform(low=-1.0, high=1.0, size=[2])
                env_action = np.concatenate([raw_action, -raw_action[1:2]])
                env_action = np.clip(env_action, [-1, 0, 0], [1, 1, 1])
            last_actions[step % 10] = raw_action
            if step % 10 == 0:
                avg_action = np.mean(last_actions, axis=0)
                print("Avg action: {}".format([round(a, 2) for a in avg_action]))
                writer.add_scalar("Avg Steering", avg_action[0], step)
                writer.add_scalar("Avg Accel", avg_action[1], step)
            print("Executing env action: {}".format(env_action))
            next_state, reward, done, trunc, info = env.step(env_action)
            next_state = next_state / 255.0
            # DEBUG!!!
            reward = 1/(abs(raw_action[1])+0.1) # should make accel 0
            episode_reward += reward
            episode_len += 1
            if episode_len >= config.max_episode_len:
                done = True
            print("Step: {}, Reward: {}".format(step, round(reward, 2)))

            # Store to buffer.
            replay_buffer.store(state, raw_action, reward, next_state, done)

            state = next_state

            if done or trunc:
                print("Training episode finished, len: {}, reward: {}".format(episode_len, episode_reward))
                writer.add_scalar("Train Episode Len", episode_len, step)
                writer.add_scalar("Train Episode Reward", episode_reward, step)
                episode_reward = 0
                episode_len = 0
                state, _ = env.reset()

            if replay_buffer.size >= config.prefill_steps and (step+1) % config.update_interval == 0: # update at the end of a round
                for i in range(config.update_interval):
                    print("Update step {}".format(i))
                    batch = replay_buffer.sample_batch(config.batch_size)
                    agent.update(batch, step, writer if i == config.update_interval-1 else None) # only log training metrics on last update
                writer.flush()

        # Save each epoch
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': agent.state_dict(),
            'pi_optimizer_state_dict': agent.pi_optimizer.state_dict(),
            'q_optimizer_state_dict': agent.q_optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Saved checkpoint at {checkpoint_path}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_run_id", type=str, default=None, help="Run ID to resume training from.")
    parser.add_argument("--toy", action="store_true", default=False, help="Just for debugging.")
    args = parser.parse_args()
    train(args)
