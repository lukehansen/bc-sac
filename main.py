import random
from datetime import datetime
import os
import argparse
import gymnasium as gym
import numpy as np
import pygame
import pickle
import time
from train import train_or_test

TRAIN_DIR = "expert/train/"

# Modified from Gymnasium.
def human_play():
    if not os.path.exists(TRAIN_DIR):
        os.makedirs(TRAIN_DIR)
    traj_nums = [int(fn.split("expert_data_")[1].split(".pkl")[0]) for fn in os.listdir(TRAIN_DIR)]
    traj_num = max(traj_nums) + 1 if traj_nums else 0

    a = np.array([0.0, 0.0, 0.0])

    def register_input():
        a[0] /= 2
        a[1] /= 2
        a[2] /= 2
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    a[0] = -1.0
                if event.key == pygame.K_RIGHT:
                    a[0] = 1.0
                if event.key == pygame.K_UP:
                    a[2] = 0
                    a[1] = 0.8
                if event.key == pygame.K_DOWN:
                    a[1] = 0
                    a[2] = 1.0
                if event.key == pygame.K_ESCAPE:
                    return False
        return True

    env = gym.make("CarRacing-v3", render_mode="human", lap_complete_percent=0.95, domain_randomize=False, continuous=True)
    env_seed = 1

    all_data = []
    started = False
    quit = False
    while not quit:
        state, _ = env.reset(seed=env_seed)
        all_data.append(env_seed)
        env_seed += 1
        total_reward = 0.0
        steps = 0
        while True:
            if not register_input():
                quit = True
                break
            if not started and np.sum(a) > 0:
                print("Starting!")
                started = True
            next_state, r, terminated, truncated, info = env.step(a)
            print("Color under car: {}, step: {}".format(next_state[71, 48, 1], steps))
            struct = {
                "state": state,
                "action": a.copy(),
                "reward": r,
                "next_state": next_state
            }
            if started:
                all_data.append(struct)
            state = next_state
            # print("Action: {}, Reward: {}".format(a, r))
            time.sleep(0.05)
            total_reward += r

            if steps % 10 == 0 and started:
                fn = os.path.join(TRAIN_DIR, "expert_data_%04d.pkl"%traj_num)
                # print("Saving pickle")
                with open(fn, "wb") as f:
                    pickle.dump(all_data, f)
            steps += 1
            if terminated or truncated or quit:
                break
    env.close()

def replay():
    for fn in os.listdir(TRAIN_DIR):
        pth = os.path.join(TRAIN_DIR, fn)
        print("Replaying file: {}".format(pth))
        with open(pth, "rb") as f:
            all_data = pickle.load(f)
        env = gym.make("CarRacing-v3", render_mode="human", lap_complete_percent=0.95, domain_randomize=False, continuous=True)
        for data in all_data:
            if type(data) == int:
                print("Starting new env with seed {}".format(data))
                env.reset(seed=data)
            else:
                # struct = {
                #     "state": state,
                #     "action": a,
                #     "reward": r,
                #     "next_state": next_state
                # }
                action = data["action"]
                print("Simulating action: {}".format([round(a, 2) for a in action]))
                time.sleep(0.05)
                env.step(data["action"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["human", "replay", "train_rl", "train_bc", "test"])
    parser.add_argument("--resume_run_id", type=str, default=None, help="Run ID to resume training from.")
    parser.add_argument("--toy", action="store_true", default=False, help="Just for debugging.")
    args = parser.parse_args()
    if args.mode == "human":
        human_play()
    elif args.mode == "replay":
        replay()
    elif "train" in args.mode or "test" in args.mode:
        train_or_test(args)
