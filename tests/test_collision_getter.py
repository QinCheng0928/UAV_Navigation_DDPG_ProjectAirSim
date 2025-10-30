import asyncio

from projectairsim import ProjectAirSimClient, Drone, World
from projectairsim.utils import projectairsim_log

"""
Collision detected: 
{
    'time_stamp': 1416000000, 
    'object_name': 'Ground', 
    'segmentation_id': 148, 
    'position': {
        'x': -1, 
        'y': 15, 
        'z': -1.1916818618774414}, 
    'impact_point': {
        'x': -1.0015671253204346, 
        'y': 15.46981430053711, 
        'z': -1}, 
    'normal': {
        'x': 0, 
        'y': 9.536517625932661e-13, 
        'z': -1}, 
    'penetration_depth': 0
}
"""
def collision_callback(topic, collision):
    print(f"Collision detected: {collision}")

async def main():
    client = ProjectAirSimClient()
    try:
        client.connect()
        world = World(client, "scene_basic_drone.jsonc", delay_after_load_sec=2)
        drone = Drone(client, world, "Drone1")

        drone.enable_api_control()
        drone.arm()

        client.subscribe(
            drone.robot_info["collision_info"],
            collision_callback,
        )

        move_up_task = await drone.move_by_velocity_async(v_north=0,v_east=0, v_down=-1, duration=2)
        await move_up_task  
        
        move_north_task = await drone.move_by_velocity_async(v_north=1,v_east=0, v_down=0, duration=2)
        await move_north_task  
            
        drone.disarm()
        drone.disable_api_control()
    except Exception as err:
        projectairsim_log().error(f"Exception occurred: {err}", exc_info=True)

    finally:
        client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())