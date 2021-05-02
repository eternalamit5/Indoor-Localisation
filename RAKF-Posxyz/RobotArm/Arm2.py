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

import matplotlib.pyplot as plt
import numpy as np
import math

class RobotArm2:

    def __init__(self, len_shoulder_to_elbow=1, len_elbow_to_gripper=1, coordinate_x=2, coordinate=2, sample_rate=0.01, proportional_gain=10):
        self.kp = proportional_gain
        self.dt = sample_rate
        self.l1 = len_shoulder_to_elbow
        self.l2 = len_elbow_to_gripper
        self.x =coordinate_x
        self.y =coordinate

        self.show_animation = True
        if self.show_animation:
            plt.ion()

    def two_joint_arm(self, GOAL_TH=0.0, theta1=0.0, theta2=0.0):
        """
        Computes the inverse kinematics for a planar 2DOF arm
        When out of bounds, rewrite x and y with last correct values
        """
        while True:
            try:
                if self.x is not None and self.y is not None:
                    x_prev = self.x
                    y_prev = self.y
                if np.sqrt( self.x ** 2 + self.y ** 2 ) > (self.l1 + self.l2):
                    theta2_goal = 0
                else:
                    theta2_goal = np.arccos(
                        (self.x ** 2 + self.y ** 2 - self.l1 ** 2 - self.l2 ** 2) / (2 * self.l1 * self.l2) )
                theta1_goal = np.math.atan2( self.y, self.x ) - np.math.atan2( self.l2 *
                                                                     np.sin( theta2_goal ), (self.l1 + self.l2 * np.cos( theta2_goal )) )

                if theta1_goal < 0:
                    theta2_goal = -theta2_goal
                    theta1_goal = np.math.atan2(
                        self.y, self.x ) - np.math.atan2( self.l2 * np.sin( theta2_goal ), (self.l1 + self.l2 * np.cos( theta2_goal )) )

                theta1 = theta1 + self.kp * self.ang_diff( theta1_goal, theta1 ) * self.dt
                theta2 = theta2 + self.kp * self.ang_diff( theta2_goal, theta2 ) * self.dt
            except ValueError as e:
                print( "Unreachable goal" )
            except TypeError:
                self.x = x_prev
                self.y = y_prev

            wrist = self.plot_arm( theta1, theta2)

            # check goal
            if self.x is not None and self.y is not None:
                d2goal = np.hypot( wrist[0] - self.x, wrist[1] - self.y )

            if abs( d2goal ) < GOAL_TH and self.x is not None:
                return theta1, theta2

    def plot_arm(self, theta1, theta2):  # pragma: no cover
        shoulder = np.array( [0, 0] )
        elbow = shoulder + np.array( [self.l1 * np.cos( theta1 ), self.l1 * np.sin( theta1 )] )
        wrist = elbow + \
                np.array( [self.l2 * np.cos( theta1 + theta2 ), self.l2 * np.sin( theta1 + theta2 )] )

        if self.show_animation:
            plt.cla()

            plt.plot( [shoulder[0], elbow[0]], [shoulder[1], elbow[1]], 'k-' )
            plt.plot( [elbow[0], wrist[0]], [elbow[1], wrist[1]], 'k-' )

            plt.plot( shoulder[0], shoulder[1], 'ro' )
            plt.plot( elbow[0], elbow[1], 'ro' )
            plt.plot( wrist[0], wrist[1], 'ro' )

            plt.plot( [wrist[0], self.x], [wrist[1], self.y], 'g--' )
            plt.plot( self.x, self.y, 'g*' )

            plt.xlim( -2, 2 )
            plt.ylim( -2, 2 )

            plt.show()
            plt.pause( self.dt )

        return wrist

    def ang_diff(self, theta1, theta2):
        # Returns the difference between two angles in the range -pi to +pi
        return (theta1 - theta2 + np.pi) % (2 * np.pi) - np.pi

    def click(self, event):  # pragma: no cover
        self.x = event.xdata
        self.y = event.ydata




