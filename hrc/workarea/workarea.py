import threading

from personnel.motion.walk_gen import create_walkers
from personnel.tracking.position_node import create_trackers
from robot.arm2.model import create_robots
from config_file_list import CONFIG_FILES, CONFIG_DIR
import os, yaml, asyncio, time


class Workspace:
    def __init__(self):
        self.id = 0
        self.personnels = []
        self.robots = []
        self.workspace_dimension = ()

        file = CONFIG_DIR + CONFIG_FILES["workspace"]
        if os.path.exists( file ):
            with open( file, 'r' ) as json_file:
                # client_json = json.load( json_file )
                client_json = yaml.load( json_file, Loader=yaml.FullLoader )
                for entry in client_json["work_area"]:
                    self.id = entry["area_id"]
                    self.dimension = (entry["dimension"]["length"], entry["dimension"]["breadth"], entry["dimension"]["height"])
                    self.robots = create_robots()
                    self.personnels = create_walkers()


ws = Workspace()


async def run_robots():
    while True:
        for robot in ws.robots:
            robot.generate_motion()
        await asyncio.sleep( 0 )


async def run_personnel():
    while True:
        for personnel in ws.personnels:
            personnel.update( tdelta=1 )
        await asyncio.sleep( 0 )


if __name__ == '__main__':
    threading.Thread( target=asyncio.run(run_robots()), args=() ).start()
    asyncio.run(run_robots())
