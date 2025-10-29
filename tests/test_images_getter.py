import asyncio

from projectairsim import ProjectAirSimClient, Drone, World
from projectairsim.utils import projectairsim_log
from projectairsim.image_utils import ImageDisplay

async def main():
    client = ProjectAirSimClient()
    image_display = ImageDisplay()
    try:
        client.connect()
        world = World(client, "scene_basic_drone.jsonc", delay_after_load_sec=2)
        drone = Drone(client, world, "Drone1")

        # callback
        chase_cam_window = "FrontCamera"
        image_display.add_chase_cam(chase_cam_window)
        client.subscribe(
            drone.sensors["FrontCamera"]["scene_camera"],
            lambda _, chase: image_display.receive(chase, chase_cam_window),
        )
        image_display.start()


        drone.enable_api_control()
        drone.arm()
        
        takeoff_task = (await drone.takeoff_async()) 
        await takeoff_task    
        move_up_task = await drone.move_by_velocity_async(v_north=0.0, v_east=0.0, v_down=-1.0, duration=4.0)
        await move_up_task        
        move_forward_task = await drone.move_by_velocity_async(v_north=5.0, v_east=0.0, v_down=0.0, duration=4.0)
        await move_forward_task       
   
        drone.disarm()
        drone.disable_api_control()
        image_display.stop()
    except Exception as err:
        projectairsim_log().error(f"Exception occurred: {err}", exc_info=True)

    finally:
        client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())