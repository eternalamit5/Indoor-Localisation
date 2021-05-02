import asyncio
import os
import yaml
import pprint
from Communication.TelegraphMQTTClient import telegraph_mqtt_client
import pandas as pd
import json
import pymongo


class DataPlayer:
    def __init__(self, config_file_name):
        self.record_status = "stop"
        self.client = None
        self.db = None
        self.collection = None
        self.measurements = []
        if os.path.exists( config_file_name ):
            with open( config_file_name, 'r' ) as config_file:
                client_json = yaml.load( config_file, Loader=yaml.FullLoader )
                for entry in client_json["data_player"]:
                    if entry["mongodb_host"] and entry["mongodb_port"] and entry["db_name"] and entry["db_collection_name"] is not None:
                        self.client = pymongo.MongoClient( entry["mongodb_host"], int( entry["mongodb_port"] ) )
                        self.db = self.client[entry["db_name"]]
                        self.collection = self.db[entry["db_collection_name"]]

    def record(self, data_point):
        if self.record_status == "start":
            if self.client and self.db and self.collection is not None:
                if isinstance( data_point, list ):
                    self.collection.insert_many( data_point )
                else:
                    self.collection.insert_one( data_point )

    def start_recording(self):
        if self.client and self.db and self.collection is not None:
            self.record_status = "start"
        else:
            self.record_status = "stop"

    def stop_recording(self):
        self.record_status = "stop"

    def play_from_db(self, tag_id, start_time, end_time):
        # start_time = 1605710762
        # end_time = 1605714368
        for result in self.collection.find({"tagId":tag_id, "timestamp":{'$gt': start_time, '$lt': end_time}}):
            pprint.pprint(result)

    async def play_from_csv(self, csv_file_name):
        df = pd.read_csv( csv_file_name )
        for index, row in df.iterrows():
            epoch_ms = pd.Timestamp( row['date'], tz='UTC' )
            self.measurements.append(
                {"timestamp": row['date'], "timestamp_epoch_ms": int( epoch_ms ), "x": float( row["x"] ),
                 "y": float( row["y"] ), "z": float( row["z"] )} )
            await asyncio.sleep( 0.01 )

        for current_index, measurement in self.measurements:
            next_index = current_index + 1
            if next_index < len( self.measurements ):
                telegraph_mqtt_client.publish( topic="player", payload=json.dumps( [measurement] ) )
                sleep_time_sec = (self.measurements[next_index]["timestamp_epoch_ms"] - measurement[
                    "timestamp_epoch_ms"]) / 1000.0
                await asyncio.sleep( sleep_time_sec )


data_player = DataPlayer("Config/DataPlayer.yaml")