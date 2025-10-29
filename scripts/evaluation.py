import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULE_DIR = os.path.join(ROOT_DIR, "checkpoints")

from stable_baselines3 import DDPG
from envs.smallcity_env import SmallCityEnv

def main():
    env = SmallCityEnv()
    
    model_name = "ddpg_smallcity_20251028_222251.zip"
    model = DDPG.load(os.path.join(MODULE_DIR, model_name))
    
    for _ in range(5):
        obs, info = env.reset()
        done = truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)

if __name__ == "__main__":
    main()