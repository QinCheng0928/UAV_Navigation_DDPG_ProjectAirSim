import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
MODULE_DIR = os.path.join(ROOT_DIR, "checkpoints", "20251030_202615")

from stable_baselines3 import PPO
from envs.projectairsim_smallcity_env import ProjectAirSimSmallCityEnv

def main():
    env = ProjectAirSimSmallCityEnv()
    
    model_name = "ppo_smallcity_20251030_202615.zip"
    model = PPO.load(os.path.join(MODULE_DIR, model_name))
    
    for _ in range(5):
        obs, info = env.reset()
        done = truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs)
            print(action)
            obs, reward, done, truncated, info = env.step(action)

if __name__ == "__main__":
    main()