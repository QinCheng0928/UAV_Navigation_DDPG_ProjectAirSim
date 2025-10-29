import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
MODULE_DIR = os.path.join(ROOT_DIR, "checkpoints")
print(f"ROOT_DIR = {ROOT_DIR}")
print(f"MODULE_DIR = {MODULE_DIR}")

from datetime import datetime
from stable_baselines3 import DDPG
from envs.smallcity_env import SmallCityEnv

def main():
    env = SmallCityEnv()
    model = DDPG(
        "MlpPolicy",
        env
    )
    model.learn(total_timesteps=int(1e2))
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"ddpg_smallcity_{current_time}"
    save_path = os.path.join(MODULE_DIR, model_name)
    model.save(save_path)
    
    
if __name__ == "__main__":
    main()