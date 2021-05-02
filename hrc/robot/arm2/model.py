"""
Inverse kinematics of a two-joint arm
Left-click the plot to set the goal position of the end effector

Author: Daniel Ingram (daniel-s-ingram)
        Atsushi Sakai (@Atsushi_twi)

Ref: P. I. Corke, "Robotics, Vision & Control", Springer 2017, ISBN 978-3-319-54413-7 p102
- [Robotics, Vision and Control \| SpringerLink](https://link.springer.com/book/10.1007/978-3-642-20144-8)


Author: Karthik 
	update-1:Dec-06-2020 :Converted the Original function implementation to class based implementation
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import math
import logging
import asyncio
import os
import yaml,threading

from commons.mqttclient import MessageTelemetryClient
from config_file_list import CONFIG_FILES, CONFIG_DIR

logging.basicConfig( level=logging.WARNING, format='%(levelname)-8s [%(filename)s:%(lineno)d] %(message)s' )


class RobotArm2:
    """This class implements Robot Arm with 2 joint ARM
    """

    def __init__(self, robot_id, mode="motion", base_x_coordinate=0, base_y_coordinate=0, len_shoulder_to_elbow=0.5, len_elbow_to_gripper=0.5,
                 sample_rate=0.01, proportional_gain=10, show_animation=False, motion_sequence="seq-1", pub_topic=None, sub_topic=None):
        """Initializes the Robot ARM

        Args:
            len_shoulder_to_elbow (int, optional): Length of arm from shoulder to elbow. Defaults to 1.
            len_elbow_to_gripper (int, optional): Length of arm from elbow to gripper. Defaults to 1.
            coordinate_x (int, optional): start x coordinate . Defaults to 2.
            coordinate (int, optional): start y coordinate. Defaults to 2.
            sample_rate (float, optional): Sample rate for data acquisition from robot ARM. Defaults to 0.01.
            proportional_gain (int, optional): Gain value, this defines settling time near destination coordinate. Defaults to 10.
        """
        if mode == "motion":
            pass  # publisher of position
        elif mode == "tracker":
            pass  # subscriber of position
        else:
            pass  # raise exception
        self.robot_id = robot_id
        self.kp = proportional_gain
        self.dt = sample_rate
        self.l1 = len_shoulder_to_elbow
        self.l2 = len_elbow_to_gripper
        self.theta1 = 0.0
        self.theta2 = 0.0
        self.GOAL_TH = 0.01
        self.dest_x = 0
        self.dest_y = 0
        self.prev_dest_x = self.dest_x
        self.prev_dest_y = self.dest_y
        self.shoulder = np.array([base_x_coordinate, base_y_coordinate])
        self.show_animation = show_animation
        self.sequence_count = 0
        self.motion_sequence = motion_sequence
        if self.show_animation:
            plt.ion()
        self.topics = dict( position_pub=pub_topic, position_sub=sub_topic )
        self.telemetry_client = MessageTelemetryClient()
        if self.topics["position_sub"] is not None:
            self.telemetry_client.subscribe( topic=self.topics["position_sub"] )
        self.telemetry_client_thread = threading.Thread( target=self.telemetry_client.start_service )
        self.telemetry_client_thread.start();

    def get_sample_time(self):
        return self.dt

    def publish(self, msg_type, msg):
        if msg_type in self.topics.keys():
            if self.topics[msg_type] is not None:
                self.telemetry_client.publish( topic=self.topics[msg_type], payload=msg )

    def get_telemetry_data(self):
        pass

    def generate_motion(self):
        """Computes the inverse kinematics for a planar 2DOF arm. When out of bounds, rewrite x and y with last correct values

        Returns:
            [type]: [description]
        """
        try:
            if math.sqrt( (self.dest_x ** 2) + (self.dest_y ** 2) ) > (self.l1 + self.l2):
                raise RuntimeError(
                    "Coordinates cannot be reached by the Robot" )

            theta2_inner = (self.dest_x ** 2 + self.dest_y ** 2 -
                            self.l1 ** 2 - self.l2 ** 2) / (2 * self.l1 * self.l2)
            if (theta2_inner > 1) or (theta2_inner < -1):
                raise RuntimeError(
                    "Coordinates cannot be reached by the Robot" )

            theta2_goal = np.arccos( theta2_inner )
            if theta2_goal < 0:
                theta1_goal = np.math.atan2( self.dest_y, self.dest_x ) + np.math.atan2(
                    self.l2 * np.sin( theta2_goal ), (self.l1 + self.l2 * np.cos( theta2_goal )) )
            else:
                theta1_goal = np.math.atan2( self.dest_y, self.dest_x ) - np.math.atan2(
                    self.l2 * np.sin( theta2_goal ), (self.l1 + self.l2 * np.cos( theta2_goal )) )

            ang_diff = lambda theta1, theta2: (theta1 - theta2 + np.pi) % (2 * np.pi) - np.pi

            self.theta1 = self.theta1 + self.kp * \
                          ang_diff( theta1_goal, self.theta1 ) * self.dt
            self.theta2 = self.theta2 + self.kp * \
                          ang_diff( theta2_goal, self.theta2 ) * self.dt

            self.prev_dest_x = self.dest_x
            self.prev_dest_y = self.dest_y

            wrist = self.update_joint_coordinates()

            # check goal
            if self.dest_x is not None and self.dest_y is not None:
                d2goal = np.hypot( wrist[0] - self.dest_x,
                                   wrist[1] - self.dest_y )

            if abs( d2goal ) < self.GOAL_TH and self.dest_x is not None:
                # return theta1, theta2
                self.get_motion_sequence()

        except ValueError as e:
            logging.critical( e )
            exit( -1 )
        except TypeError as e:
            logging.critical( e )
            exit( -1 )
        except RuntimeError as e:
            logging.critical( e )
            self.dest_x = self.prev_dest_x
            self.dest_y = self.prev_dest_y

    def get_motion_sequence(self):
        if self.motion_sequence == "seq-1":
            if self.sequence_count == 1:
                self.dest_x = 0.6
                self.dest_y = 0.6
            elif self.sequence_count == 2:
                self.dest_x = 1
                self.dest_y = 1
            else:
                self.sequence_count = 0
        elif self.motion_sequence == "seq-2":
            if self.sequence_count == 1:
                self.dest_x = -0.6
                self.dest_y = -0.6
            elif self.sequence_count == 2:
                self.dest_x = 0.1
                self.dest_y = 0.1
            elif self.sequence_count == 3:
                self.dest_x = 0.5
                self.dest_y = 0.3
                self.sequence_count += 1
            else:
                self.sequence_count = 0

        self.sequence_count += 1

    def update_joint_coordinates(self):  # pragma: no cover
        """Ploting arm

            Returns:
                [type]: [description]
            """
        result = dict()

        elbow = self.shoulder + \
                np.array( [self.l1 * np.cos( self.theta1 ), self.l1 * np.sin( self.theta1 )] )
        wrist = elbow + \
                np.array( [self.l2 * np.cos( self.theta1 + self.theta2 ),
                           self.l2 * np.sin( self.theta1 + self.theta2 )] )
        result.update({
            "robot_id": self.robot_id,
            "shoulder":np.array2string(self.shoulder),
            "elbow": np.array2string( elbow ),
            "wrist": np.array2string( wrist )
        })
        self.publish( msg_type="position_pub", msg=json.dumps( result ) )

        self.animate( shoulder=self.shoulder, elbow=elbow, wrist=wrist )

        wrist[0] -= self.shoulder[0]
        wrist[1] -= self.shoulder[1]
        return wrist

    def get_joint_coordinates(self):
        elbow = self.shoulder + \
                np.array( [self.l1 * np.cos( self.theta1 ), self.l1 * np.sin( self.theta1 )] )
        wrist = elbow + \
                np.array( [self.l2 * np.cos( self.theta1 + self.theta2 ),
                           self.l2 * np.sin( self.theta1 + self.theta2 )] )
        return dict(
            shoulder=(self.shoulder[0], self.shoulder[1]),
            elbow=(elbow[0], elbow[1]),
            wrist=(wrist[0], wrist[1])
        )

    def animate(self, shoulder, elbow, wrist):
        if self.show_animation:
            plt.cla()
            plt.plot( [shoulder[0], elbow[0]], [shoulder[1], elbow[1]], 'k-' )
            plt.plot( [elbow[0], wrist[0]], [elbow[1], wrist[1]], 'k-' )

            plt.plot( shoulder[0], shoulder[1], 'ro' )  # base of the robot
            plt.plot( elbow[0], elbow[1], 'go' )  # joint 1
            plt.plot( wrist[0], wrist[1], 'bo' )  # Tool tip

            # plt.plot( [wrist[0], self.x], [wrist[1], self.y], 'g--' )
            # plt.plot( self.x, self.y, 'g*' )
            plt.xlim( shoulder[0] - 2, shoulder[0] + 2 )
            plt.ylim( shoulder[1] - 2, shoulder[1] + 2 )
            plt.show()
            plt.pause( self.dt )


def create_robots():
    try:
        filename = CONFIG_DIR + CONFIG_FILES["robot"]
        robots = []
        if os.path.exists( filename ):
            with open( filename, 'r' ) as json_file:
                robot_config = yaml.load( json_file, Loader=yaml.FullLoader )
                for robot in robot_config["robots"]:
                    if robot["arm_count"] == 2:
                        specs = robot["specs"]
                        base_coordinate = robot["base_coordinate"]
                        robot = RobotArm2( robot_id=robot["robot_id"],
                                           mode=robot["mode"],
                                           base_x_coordinate=base_coordinate["x"],
                                           base_y_coordinate=base_coordinate["y"],
                                           len_shoulder_to_elbow=specs["length_shoulder_to_elbow"],
                                           len_elbow_to_gripper=specs["length_elbow_to_gripper"],
                                           sample_rate=robot["sample_rate"],
                                           proportional_gain=robot["proportional_gain"],
                                           show_animation=robot["show_animation"],
                                           motion_sequence=robot["motion_sequence"],
                                           pub_topic=robot["telemetry_topics"]["position_publish"],
                                           sub_topic=None )
                        robots.append( robot )
            return robots
        else:
            raise FileNotFoundError( "File not found. Check filename and filepath" )
    except FileNotFoundError as e:
        logging.critical( e )
        exit( -1 )
    except yaml.YAMLError as e:
        logging.critical( e )
        exit( -1 )
    except Exception as e:
        logging.critical( e )
        exit( -1 )


async def test():
    machines = create_robots()
    while True:
        for robot in machines:
            robot.generate_motion()
        await asyncio.sleep( 0.01 )


if __name__ == "__main__":
    asyncio.run( test() )
