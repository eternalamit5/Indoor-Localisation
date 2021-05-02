import os
import paho.mqtt.client as mqtt
import yaml
import threading, queue
from config_file_list import CONFIG_FILES, CONFIG_DIR


# def on_connect(client, userdata, flags, rc):
#     pass
#
#
# def on_message(client, userdata, msg):
#     pass


class MessageTelemetryClient:
    def __init__(self, on_connect_cb=None, on_message_cb=None):
        self.client = mqtt.Client()
        self.subscribe_topic_list = []
        self.msg_queue = queue.Queue()
        self.msg_queue_capacity = 0.0

        if on_connect_cb is not None:
            self.client.on_connect = on_connect_cb
        else:
            self.client.on_connect = self.on_connect

        if on_message_cb is not None:
            self.client.on_message = on_message_cb
        else:
            self.client.on_message = self.on_message

    def start_service(self):
        file = CONFIG_DIR + CONFIG_FILES["msg_telemetry_client"]
        if os.path.exists( file ):
            with open( file, 'r' ) as json_file:
                # client_json = json.load( json_file )
                client_json = yaml.load( json_file, Loader=yaml.FullLoader )
                for entry in client_json["mqtt_client_config"]:
                    self.msg_queue_capacity = entry["message_queue_capacity"]
                    self.client.connect( entry["host_name"], int(entry["port"]), int(entry["keep_alive_timeout"]) )
                # threading.Thread( target=self.client.loop_forever, daemon=True ).start()
                self.client.loop_forever()

    def publish(self, topic, payload):
        self.client.publish( topic=topic, payload=payload )

    def subscribe(self, topic, qos=0):
        self.subscribe_topic_list.append( topic )
        self.client.subscribe( topic=topic, qos=qos )

    def add_to_msg_queue(self, data):
        if self.msg_queue.qsize() < self.msg_queue_capacity:
            self.msg_queue.put( item=data )
            return True
        else:
            return False

    def get_from_msg_queue(self):
        try:
            if not self.msg_queue.empty():
                data = self.msg_queue.get()
                return data
            else:
                return None
        except queue.Empty as e:
            return None

    def stop(self):
        self.client.loop_stop( force=True )

    def on_connect(self, client, userdata, flags, rc):
        for topic in self.subscribe_topic_list:
            self.client.subscribe(topic=topic)

    def on_message(self, client, userdata, msg):
        try:
            if not self.add_to_msg_queue( data=dict(topic=msg.topic,payload=msg.payload.decode( 'UTF-8' )) ):
                raise queue.Full("Queue is full, message discarded")
        except queue.Full as e:
            print(e)
            print(self.get_from_msg_queue()["payload"])

if __name__ == '__main__':
    msgclient = MessageTelemetryClient()
    msgclient.subscribe(topic="apple/")
    msgclient.start_service()
