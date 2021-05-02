import os
import paho.mqtt.client as mqtt
import json
import yaml


def on_connect(client, userdata, flags, rc):
    for event in telegraph_mqtt_client.subscribe_topics:
        client.subscribe( event )


def on_message(client, userdata, msg):
    if msg.topic == telegraph_mqtt_client.subscribe_topics[0]:
        pass


class LocalTelemetryRadio:
    def __init__(self, on_connect_cb, on_message_cb):
        self.client = mqtt.Client()
        self.name = None
        self.client.on_connect = on_connect_cb
        self.client.on_message = on_message_cb
        self.host_name = ""
        self.port = 1886
        self.keep_alive_timeout = ""
        self.subscribe_topics = []

    def run(self, json_file_name):
        if os.path.exists( json_file_name ):
            with open( json_file_name, 'r' ) as json_file:
                # client_json = json.load( json_file )
                client_json = yaml.load( json_file, Loader=yaml.FullLoader )
                for entry in client_json["local-telemetry_config"]:
                    # --- add more attributes ------
                    self.name = entry["name"]
                    self.host_name = entry["host_name"]
                    self.port = entry["port"]
                    self.keep_alive_timeout = entry["keep_alive_timeout"]
                    for topic in entry["subscribe_topics"]:
                        self.subscribe_topics.append( topic )
        self.client.connect( self.host_name, self.port, self.keep_alive_timeout )
        self.client.loop_forever()

    def publish(self, topic, payload):
        self.client.publish( topic=topic, payload=payload )


telegraph_mqtt_client = LocalTelemetryRadio( on_connect, on_message )
