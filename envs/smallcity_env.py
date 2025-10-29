import numpy as np
import math

import asyncio
import gymnasium as gym
from gymnasium import spaces
from projectairsim import ProjectAirSimClient, Drone, World
from projectairsim.image_utils import ImageDisplay, unpack_image

from envs.utils.type import ActionType
import time

class SmallCityEnv(gym.Env):
    def __init__(self, velocity_change = 1, image_shape = (84, 84, 1)):
        super().__init__()
        self.velocity_change = velocity_change
        self.image_shape = image_shape
        self.observation_space = spaces.Dict({
            'depth_image': spaces.Box(0, 255, image_shape, np.uint8),
            'position': spaces.Box(-np.inf, np.inf, (3,), np.float32), 
            'velocity': spaces.Box(-np.inf, np.inf, (3,), np.float32), 
            'speed': spaces.Box(0, np.inf, (1,), np.float32)  
        })

        self.state = {
            "position": np.zeros(3),
            "collision": False,
            "prev_position": np.zeros(3),
            "velocity": np.zeros(3),
        }

        self.client = ProjectAirSimClient()
        self.client.connect()
        self.world = World(self.client, "scene_basic_drone.jsonc", delay_after_load_sec=2)
        self.drone = Drone(self.client, self.world, "Drone1")
        self.action_space = spaces.Discrete(7)

        time.sleep(0.5)

        self.image_display = ImageDisplay()
        self.client.subscribe(
            self.drone.robot_info["collision_info"],
            self._collision_callback,
        )

        self.chase_cam_window = "Depth-Image"
        self.image_display.add_chase_cam(self.chase_cam_window)
        self.client.subscribe(
            self.drone.sensors["FrontCamera"]["depth_camera"],
            self._image_update,
        )
        
        rgb_name = "RGB-Image"
        self.image_display.add_image(rgb_name, subwin_idx=0)
        self.client.subscribe(
            self.drone.sensors["FrontCamera"]["scene_camera"],
            lambda _, rgb: self.image_display.receive(rgb, rgb_name),
        )        
        
        self.sim_step = 0
        self.max_sim_steps = 50
        self.arrived = np.array([140.0, 10.0, -20.0])
        self.thresh_dist = 5.0
        
        self._setup_flight()
        
    def _collision_callback(self, topic, msg):
        print("Collision detected!")
        self.state["collision"] = True
        
    def _image_update(self, topic, msg):
        self.image_display.receive(msg, self.chase_cam_window)
        self.preprocessed_image = self.preprocess_image(unpack_image(msg))
        
    def __del__(self):
        self.drone.disarm()
        self.drone.disable_api_control()
        self.image_display.stop()
        self.client.disconnect()

    def _setup_flight(self):
        self.drone.enable_api_control()
        self.drone.arm()
        self.image_display.start()

    def preprocess_image(self, responses):
        img2d = np.array(responses, dtype=np.float32)
        img2d = 255 / np.maximum(np.ones_like(img2d), img2d)
        img2d = np.clip(img2d, 0, 255).astype(np.uint8)
        from PIL import Image
        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert("L"))

        return im_final.reshape([84, 84, 1])

    def _obs(self):
        depth_image = self.preprocessed_image
        
        self.drone_state = self.drone.get_ground_truth_kinematics()

        self.state["prev_position"] = self.state["position"]
        self.state["position"] = self.drone_state["pose"]["position"]
        self.state["velocity"] = self.drone_state["twist"]["linear"]
        
        velocity = self.state["velocity"]
        speed = np.sqrt(velocity["x"]**2 + velocity["y"]**2 + velocity["z"]**2)

        self.quad_position = np.array([self.state["position"]["x"], self.state["position"]["y"], self.state["position"]["z"]])
        print(f"Quad Position: {self.quad_position}, Target Position: {self.arrived}, Distance to Target: {self.distance_3d(self.quad_position, self.arrived)}")

        observation = {
            'depth_image': depth_image.astype(np.uint8),
            'position': np.array([
                self.state["position"]["x"],
                self.state["position"]["y"],
                self.state["position"]["z"]
            ], dtype=np.float32),
            'velocity': np.array([
                self.state["velocity"]["x"],
                self.state["velocity"]["y"],
                self.state["velocity"]["z"]
            ], dtype=np.float32),
            'speed': np.array([speed], dtype=np.float32)
        }

        return observation

    def interpret_action(self, action):
        print(f"Action taken: {ActionType.NUM2NAME[action]}")
        if action == ActionType.NORTH:
            quad_offset = (self.velocity_change, 0, 0)
        elif action == ActionType.EAST:
            quad_offset = (0, self.velocity_change, 0)
        elif action == ActionType.DOWN:
            quad_offset = (0, 0, self.velocity_change)
        elif action == ActionType.SOUTH:
            quad_offset = (-self.velocity_change, 0, 0)
        elif action == ActionType.WEST:
            quad_offset = (0, -self.velocity_change, 0)
        elif action == ActionType.UP:
            quad_offset = (0, 0, -self.velocity_change)
        else:
            quad_offset = (0, 0, 0)

        ## debug mode
        if self.sim_step < 2:
            print("Debug mode UP")
            quad_offset = (0, 0, -1)
        else:
            print("Debug mode NORTH")
            quad_offset = (1, 0, 0)
        
        return quad_offset 

    async def _do_action(self, action):
        self.sim_step += 1
        quad_offset = self.interpret_action(action)
        quad_vel = self.state["velocity"]
        move_up_task = await self.drone.move_by_velocity_async(
            v_north=quad_vel["x"] + quad_offset[0], 
            v_east=quad_vel["y"] + quad_offset[1], 
            v_down=quad_vel["z"] + quad_offset[2], 
            duration=5)
        await move_up_task        


    def distance_3d(self, p1, p2):
        p1 = np.array(p1, dtype=float)
        p2 = np.array(p2, dtype=float)
        return np.linalg.norm(p1 - p2)

    def _has_arrived(self):
        return self.distance_3d(self.quad_position, self.arrived) < self.thresh_dist

    def _rewards(self):
        dist = np.inf
        
        if self.state["collision"]:
            reward = -100.0
        else:
            dist = self.distance_3d(self.quad_position, self.arrived)
            reward_dist = math.exp(-dist) - 0.5
            reward_speed = (np.linalg.norm([self.state["velocity"]["x"], self.state["velocity"]["y"], self.state["velocity"]["z"]]) - 0.5)
            reward = reward_dist + reward_speed

        if self._has_arrived():
            reward += 10.0
        
        done = self._is_terminal()
        
        return reward, done

    def _is_terminal(self):
        return bool((self.state["collision"] or self._has_arrived()))

    def _is_truncated(self):
        return self.sim_step >= self.max_sim_steps
    
    def step(self, action):
        asyncio.run(self._do_action(action))
        obs = self._obs()
        reward, done = self._rewards()
        truncated = self._is_truncated()
        info = {}
        print(f"Reward: {reward}, Done: {done}, Truncated: {truncated}")
        return obs, reward, done, truncated, info

    def reset(self, *, seed=None, options=None):
        self.world = World(self.client, "scene_basic_drone.jsonc", delay_after_load_sec=2)
        self.drone = Drone(self.client, self.world, "Drone1")
        self.observation_space = spaces.Dict({
            'depth_image': spaces.Box(0, 255, self.image_shape, np.uint8),
            'position': spaces.Box(-np.inf, np.inf, (3,), np.float32), 
            'velocity': spaces.Box(-np.inf, np.inf, (3,), np.float32), 
            'speed': spaces.Box(0, np.inf, (1,), np.float32)  
        })

        self.state = {
            "position": np.zeros(3),
            "collision": False,
            "prev_position": np.zeros(3),
            "velocity": np.zeros(3),
        }    
        self._setup_flight()
        obs = self._obs()
        info = {}
        return obs, info
    
    