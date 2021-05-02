import os

import yaml

from Algorithm.RAKF3D import RAKF3D
import asyncio


class Tag:
    def __init__(self, tag_id, algorithm):
        self.tag_id = tag_id
        self.algorithm = algorithm
        self.instance = None
        if algorithm == "RAKF3D":
            self.instance = RAKF3D( "Config/RAKF3D.yaml", tag_id=tag_id )
        else:
            print( "Algorithm is not supported" );
            exit( -1 )

    def run(self):
        self.instance.run()


class Personnel:
    def __init__(self, json_file_name):
        self.tags = []
        self.name = None
        self.UUID = None

        if os.path.exists( json_file_name ):
            with open( json_file_name, 'r' ) as json_file:
                # client_json = json.load( json_file )
                client_json = yaml.load( json_file, Loader=yaml.FullLoader )
                for entry in client_json["personnel"]:
                    self.name = entry["name"]
                    self.UUID = entry["UUID"]
                    for tag in entry["tags"]:
                        self.add_tag(tag["id"],tag["algorithm"])

    def add_tag(self, tag_id, algorithm):
        new_tag = Tag( tag_id=tag_id, algorithm=algorithm )
        self.tags.append( new_tag )

    def append_measurement(self,tag_id,data):
        for tag in self.tags:
            if tag.tag_id == tag_id:
                tag.instance.append_measurement(data)

    async def run(self):
        while True:
            for tag in self.tags:
                tag.run()
        await asyncio.sleep( 0.01 )


person = Personnel("Config/config.yaml")