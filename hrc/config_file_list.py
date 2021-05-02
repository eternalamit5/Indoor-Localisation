import os

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__)) +"/config/"

CONFIG_FILES = dict(
    personnel="personnel.yaml",
    tracker="tracker.yaml",
    robot="robot.yaml",
    workspace="workarea.yaml",
    msg_telemetry_client="msgtelemetryclient.yaml"
)


