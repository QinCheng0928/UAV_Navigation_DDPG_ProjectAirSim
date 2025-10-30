import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
MODULE_DIR = os.path.join(ROOT_DIR, "checkpoints")
print(f"ROOT_DIR = {ROOT_DIR}")
print(f"MODULE_DIR = {MODULE_DIR}")

from datetime import datetime
from stable_baselines3 import PPO
from envs.projectairsim_smallcity_env import ProjectAirSimSmallCityEnv

def main():
    env = ProjectAirSimSmallCityEnv()
    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=dict(net_arch=(dict(pi=[256, 256], vf=[256, 256]))),
        n_steps=128,
        batch_size=64,
        n_epochs=10,
        learning_rate=5e-4,
        gamma=0.99,
        verbose=2,
        device="cuda",
        tensorboard_log="../tensorboard_log/ppo/"
    )
    model.learn(total_timesteps=int(1e4))
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"ddpg_smallcity_{current_time}"
    save_path = os.path.join(MODULE_DIR, model_name)
    model.save(save_path)
    
    
if __name__ == "__main__":
    main()