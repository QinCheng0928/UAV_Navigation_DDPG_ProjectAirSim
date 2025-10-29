import asyncio

from projectairsim import ProjectAirSimClient, Drone, World
from projectairsim.utils import projectairsim_log

async def main():
    client = ProjectAirSimClient()
    try:
        client.connect()
        world = World(client, "scene_basic_drone.jsonc", delay_after_load_sec=2)
        drone = Drone(client, world, "Drone1")

        drone.enable_api_control()
        drone.arm()

        pose = drone.get_ground_truth_pose()
        print(f"Initial Pose: {pose}")
        
        """
        Example pose output when setting "rpy-deg": "0 0 -45" in <...>/sim_config/scene_basic_drone.jsonc:
        
        Euler Angles "0 0 -45" correspond to Quaternion "w: 0.9238795, x: 0, y: 0, z: -0.3826834"
        
        Expected output format:
        {
            'frame_id': 'DEFAULT_ID', 
            'rotation': {
                'w': 0.9238795042037964, 
                'x': 0.0, 
                'y': 0.0, 
                'z': -0.3826834559440613}, 
            'translation': {
                'x': -1.0, 
                'y': 8.0, 
                'z': -1.1926817893981934}
        }
        """
        kinematics = drone.get_ground_truth_kinematics()
        print(f"Initial kinematics: {kinematics}")
        """
        Example kinematics output:
        {
            'accels': {
                'angular': {
                    'x': 0.0, 
                    'y': 0.0, 
                    'z': 0.0}, 
                'linear': {
                    'x': 0.0, 
                    'y': 0.0, 
                    'z': 0.0}
                }, 
            'pose': {
                'orientation': {
                    'w': 0.9238795042037964, 
                    'x': 0.0, 
                    'y': 0.0, 
                    'z': -0.3826834559440613}, 
                'position': {
                    'x': -1.0, 
                    'y': 8.0, 
                    'z': -1.1926817893981934}
                }, 
            'time_stamp': 2013000000, 
            'twist': {
                'angular': {
                    'x': 0.0, 
                    'y': 0.0, 
                    'z': 0.0}, 
                'linear': {
                    'x': 0.0, 
                    'y': 0.0, 
                    'z': 0.0}
                }
            }
        """
        
        
        drone.disarm()
        drone.disable_api_control()
    except Exception as err:
        projectairsim_log().error(f"Exception occurred: {err}", exc_info=True)

    finally:
        client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())