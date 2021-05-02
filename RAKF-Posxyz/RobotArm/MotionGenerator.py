from Arm2 import RobotArm2
import matplotlib.pyplot as plt
import numpy as np
from random import random

# Motion pattern variables
motion_pattern_var1 = 1
motion_pattern_var2 = 0
motion_pattern_var3 = 0
motion_pattern_var4 = 0


def motion_pattern_1(robot_type="ARM-2"):
    if robot_type == "ARM-2":
        robot = RobotArm2()
        theta1 = theta2 = 0.0
        for i in range( 5 ):
            robot.x = 2.0 * random() - 1.0
            robot.y = 2.0 * random() - 1.0
            theta1, theta2 = robot.two_joint_arm(
                GOAL_TH=0.01, theta1=theta1, theta2=theta2 )


def motion_pattern_2(robot_type="ARM-2"):
    if robot_type == "ARM-2":
        robot = RobotArm2( len_elbow_to_gripper=1, len_shoulder_to_elbow=0.6 )
        global motion_pattern_var1, motion_pattern_var2, motion_pattern_var3, motion_pattern_var4
        theta1 = theta2 = 0.0
        while True:

            if motion_pattern_var4 < 10:
                motion_pattern_var2 = 0.6
                motion_pattern_var3 = 0.6
                motion_pattern_var4 = motion_pattern_var4 + 1
            elif motion_pattern_var4 < 20:
                motion_pattern_var2 = 1.0
                motion_pattern_var3 = 1.0
                motion_pattern_var4 = motion_pattern_var4 + 1
            else:
                motion_pattern_var4 = 0

            robot.x = motion_pattern_var2
            robot.y = motion_pattern_var3
            theta1, theta2 = robot.two_joint_arm(
                GOAL_TH=0.01, theta1=theta1, theta2=theta2 )


def interactive_motion(robot_type="ARM-2"):  # pragma: no cover
    if robot_type == "ARM-2":
        robot = RobotArm2()
        fig = plt.figure()
        fig.canvas.mpl_connect( "button_press_event", robot.click )
        # for stopping simulation with the esc key.
        fig.canvas.mpl_connect( 'key_release_event', lambda event: [
            exit( 0 ) if event.key == 'escape' else None] )
        robot.two_joint_arm()


if __name__ == "__main__":
    motion_pattern_2()
