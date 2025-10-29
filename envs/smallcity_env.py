import numpy as np
import math

import gymnasium as gym
from gymnasium import spaces
from projectairsim import ProjectAirSimClient, Drone, World
from projectairsim.image_utils import ImageDisplay

class SmallCityEnv(gym.Env):
    def __init__(self, step_length = 0.1, image_shape = (84, 84, 1)):
        super().__init__()
        self.step_length = step_length
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
        self.image_display = ImageDisplay()
        
        self.action_space = spaces.Discrete(7)

        self.client.subscribe(
            self.drone.robot_info["collision_info"],
            lambda topic, msg: self.state.update({"collision": True}),
        )

        self.chase_cam_window = "FrontCamera"
        self.image_display.add_chase_cam(self.chase_cam_window)
        self.client.subscribe(
            self.drone.sensors["FrontCamera"]["scene_camera"],
            self._image_update,
        )
        self._setup_flight()
        
        
    def _image_update(self, topic, msg):
        self.image_display.receive(msg, self.chase_cam_window)
        self.preprocessed_image = self.preprocess_image(msg)
        
        
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
        img1d = np.array(responses, dtype=np.float32)
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (responses.height, responses.width))

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
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

        collision = self.drone.collision()
        self.state["collision"] = collision

        observation = {
            'depth_image': depth_image.astype(np.uint8),
            'position': np.array([
                self.state["position"].x,
                self.state["position"].y,
                self.state["position"].z
            ], dtype=np.float32),
            'velocity': np.array([
                self.state["velocity"].x,
                self.state["velocity"].y,
                self.state["velocity"].z
            ], dtype=np.float32),
            'speed': np.array([speed], dtype=np.float32)
        }

        return observation

    async def _do_action(self, action):
        quad_offset = self.interpret_action(action)
        quad_vel = self.state["velocity"]
        move_up_task = await self.drone.move_by_velocity_async(
            v_north=quad_vel.x + quad_offset[0], 
            v_east=quad_vel.y + quad_offset[1], 
            v_down=quad_vel.z + quad_offset[2], 
            duration=5)
        await move_up_task        

    def _rewards(self):
        thresh_dist = 7.0
        beta = 0.99

        waypoints = [
            np.array([-0.55265, -31.9786, -19.0225]),
            np.array([48.59735, -63.3286, -60.07256]),
            np.array([193.5974, -55.0786, -46.32256]),
            np.array([369.2474, 35.32137, -62.5725]),
            np.array([541.3474, 143.6714, -32.07256]),
        ]
        n_waypoints = len(waypoints)
        quad_position = np.array([self.state["position"].x, self.state["position"].y, self.state["position"].z])

        if self.state["collision"]:
            reward = -100.0
        else:
            dist = np.inf
            for i in range(n_waypoints - 1):
                dist = min(
                    dist,
                    np.linalg.norm(np.cross((quad_position - waypoints[i]), (quad_position - waypoints[i + 1])))
                    / np.linalg.norm(waypoints[i] - waypoints[i + 1]),
                )

            if dist > thresh_dist:
                reward = -10.0
            else:
                reward_dist = math.exp(-beta * dist) - 0.5
                reward_speed = (
                    np.linalg.norm([
                        self.state["velocity"].x,
                        self.state["velocity"].y,
                        self.state["velocity"].z,
                    ]) - 0.5
                )
                reward = reward_dist + reward_speed

        done = False
        if reward <= -10:
            done = False

        return reward, done

    def step(self, action):
        self._do_action(action)
        obs = self._obs()
        reward, done = self._rewards()
        truncated = False
        info = {}
        return obs, reward, done, truncated, info
        # return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        self._setup_flight()
        obs = self._obs()
        info = {}
        return obs, info

    def interpret_action(self, action):
        if action == 0:
            quad_offset = (self.step_length, 0, 0)
        elif action == 1:
            quad_offset = (0, self.step_length, 0)
        elif action == 2:
            quad_offset = (0, 0, self.step_length)
        elif action == 3:
            quad_offset = (-self.step_length, 0, 0)
        elif action == 4:
            quad_offset = (0, -self.step_length, 0)
        elif action == 5:
            quad_offset = (0, 0, -self.step_length)
        else:
            quad_offset = (0, 0, 0)

        return quad_offset