import numpy as np
import math

import asyncio
import gymnasium as gym
from gymnasium import spaces
from projectairsim import ProjectAirSimClient, Drone, World
from projectairsim.image_utils import ImageDisplay, unpack_image
from projectairsim.utils import load_scene_config_as_dict

from envs.utils.type import ActionType

import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class ProjectAirSimSmallCityEnv(gym.Env):
    def __init__(self):
        self.sim_config_fname = os.path.join(ROOT_DIR, "sim_config", "scene_basic_drone.jsonc")
        self.velocity_change = 1.0
        self.thresh_dist = 5.0
        self.max_sim_steps = 500
        self._ignore_first_collision = True
        self.image_shape = (84, 84, 1)
        self.target_point = np.array([140.0, 10.0, -20.0])
        
        self.loop = asyncio.get_event_loop()
        
        super().__init__()

        self.client = ProjectAirSimClient()
        self.client.connect()
        self.world = World(self.client, self.sim_config_fname)
        
        self.image_display = ImageDisplay()
        
        self.observation_space = spaces.Dict({
            'depth_image': spaces.Box(0, 255, self.image_shape, np.uint8),
            'position': spaces.Box(-np.inf, np.inf, (3,), np.float32), 
            "pose": spaces.Box(-np.inf, np.inf, (4,), np.float32),
            'velocity': spaces.Box(-np.inf, np.inf, (3,), np.float32), 
            "collision": spaces.Discrete(2),
        })
        self.action_space = spaces.Discrete(7)

        self.state = {
            'depth_image': np.zeros_like(self.image_shape, dtype=np.uint8),
            'position': np.zeros(3, dtype=np.float32), 
            'pose': np.zeros(4, dtype=np.float32), 
            'velocity': np.zeros(3, dtype=np.float32),
            'collision': 0,  
        }
        self.preprocessed_image = np.zeros(self.image_shape, dtype=np.uint8)
        
    def reset(self, *, seed=None, options=None):
        # reset the value of state variables
        self.state = {
            'depth_image': np.zeros_like(self.image_shape, dtype=np.uint8),
            'position': np.zeros(3, dtype=np.float32), 
            'pose': np.zeros(4, dtype=np.float32), 
            'velocity': np.zeros(3, dtype=np.float32),
            'collision': 0,  
        }
        self.sim_step = 0
        self.preprocessed_image = np.zeros(self.image_shape, dtype=np.uint8)
        self._ignore_first_collision = True
        
        # reset the sim world
        config_loaded, _ = load_scene_config_as_dict(
            self.sim_config_fname, 
            sim_config_path="../sim_config/", 
            sim_instance_idx=-1
            )
        self.world.load_scene(config_loaded, delay_after_load_sec=0)
        self.drone = Drone(self.client, self.world, "Drone1")
        
        self.client.subscribe(
            self.drone.robot_info["collision_info"],
            self._collision_callback,
        )

        self.chase_cam_window = "Depth-Image"
        self.image_display.add_chase_cam(self.chase_cam_window)
        self.client.subscribe(
            self.drone.sensors["FrontCamera"]["depth_camera"],
            self._depth_image_callback,
        )
        
        rgb_name = "RGB-Image"
        self.image_display.add_image(rgb_name, subwin_idx=0)
        self.client.subscribe(
            self.drone.sensors["FrontCamera"]["scene_camera"],
            lambda _, rgb: self.image_display.receive(rgb, rgb_name),
        )   
        
        self.image_display.start()           
        
        self.drone.enable_api_control()
        self.drone.arm()        
        
        obs = self.get_observation()
        info = {}
        return obs, info
    
    def step(self, action):
        self.sim_step += 1
        
        self.loop.run_until_complete(self._simulate(action))
        
        obs = self.get_observation()
        reward = self.get_reward()
        done = self._is_terminal()
        truncated = self._is_truncated()
        info = {}
        print(f"Reward: {reward}, Done: {done}, Truncated: {truncated}")
        return obs, reward, done, truncated, info
        # return obs, reward, done, truncated, info

    async def _simulate(self, action):
        print(f"Action taken: {ActionType.NUM2NAME[action]}")
        
        curr_velocity = self.state["velocity"]
        vx = curr_velocity[0]
        vy = curr_velocity[1]
        vz = curr_velocity[2]
        
        # update velocity based on action
        if action == ActionType.NORTH:
            vx += self.velocity_change
        elif action == ActionType.EAST:
            vy += self.velocity_change
        elif action == ActionType.DOWN:
            vz += self.velocity_change
        elif action == ActionType.SOUTH:
            vx -= self.velocity_change
        elif action == ActionType.WEST:
            vy -= self.velocity_change
        elif action == ActionType.UP:
            vz -= self.velocity_change
        elif action == ActionType.NONE:
            pass
        
        # send velocity command to the drone
        move_up_task = await self.drone.move_by_velocity_async(v_north=vx, v_east=vy, v_down=vz, duration=0.5)
        await move_up_task        

    
    def get_observation(self):
        # get depth image
        depth_image = self.preprocessed_image
        self.state['depth_image'] = depth_image.astype(np.uint8)
        
        # get position, pose, velocity
        self.drone_state = self.drone.get_ground_truth_kinematics()
        self.state["position"] = [self.drone_state["pose"]["position"]["x"], self.drone_state["pose"]["position"]["y"], self.drone_state["pose"]["position"]["z"]]
        self.state["pose"] =[ self.drone_state["pose"]["orientation"]["w"], self.drone_state["pose"]["orientation"]["x"], self.drone_state["pose"]["orientation"]["y"], self.drone_state["pose"]["orientation"]["z"]]
        self.state["velocity"] = [self.drone_state["twist"]["linear"]["x"], self.drone_state["twist"]["linear"]["y"], self.drone_state["twist"]["linear"]["z"]]
        # collison state is updated in the _collision_callback function

        return self.state

    def get_reward(self):
        # rewards for speed and distance to the target point 
        dist = self.distance_3d(self.state["position"], self.target_point)
        reward_dist = math.exp(-dist) - 0.5
        reward_speed = (np.linalg.norm([self.state["velocity"][0], self.state["velocity"][1], self.state["velocity"][2]]) - 0.5)            
        reward = reward_dist + reward_speed
        
        # rewards for collision and terminal state
        if self.state["collision"] or self._is_terminal():
            reward = -100.0
        
        # rewards for arrival
        if self._has_arrived():
            reward += 10.0
        
        return reward
        
    def close(self):
        self.drone.disarm()
        self.drone.disable_api_control()
        self.image_display.stop()
        self.client.disconnect()

    # ==================================================
    # utils functions
    # ==================================================
    def preprocess_image(self, responses):
        img2d = np.array(responses, dtype=np.float32)
        img2d = 255 / np.maximum(np.ones_like(img2d), img2d)
        img2d = np.clip(img2d, 0, 255).astype(np.uint8)
        from PIL import Image
        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert("L"))

        return im_final.reshape([84, 84, 1])

    def distance_3d(self, p1, p2):
        p1 = np.array(p1, dtype=float)
        p2 = np.array(p2, dtype=float)
        return np.linalg.norm(p1 - p2)

    def _has_arrived(self):
        return self.distance_3d(self.state["position"], self.target_point) < self.thresh_dist

    def _is_terminal(self):
        return bool((self.state["collision"] or self._has_arrived()))

    def _is_truncated(self):
        return self.sim_step >= self.max_sim_steps
        
    # ==================================================
    # Callback functions
    # ==================================================
    def _collision_callback(self, topic, msg):
        if self._ignore_first_collision:
            self._ignore_first_collision = False
            return
        print("Collision detected!")
        self.state["collision"] = True
    
    def _depth_image_callback(self, topic, msg):
        self.image_display.receive(msg, self.chase_cam_window)
        self.preprocessed_image = self.preprocess_image(unpack_image(msg))
