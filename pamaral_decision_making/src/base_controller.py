#!/usr/bin/env python3

import json
import rospy
import sys

from geometry_msgs.msg import PointStamped

from arm.srv import MoveArmToPoseGoal, MoveArmToPoseGoalRequest, StopArm
from pamaral_perception_system.msg import CentroidList


class BaseController:
    """State machine to decide what should be the action of the robot.

    States:
        idle - waiting for the desired sequence of objects
        picking_up - picking up a given object
        moving_closer - moving closer to the user
        putting_down - putting down the object close to the user
        stop_side_switch - while the robot is stopping because the user changed sides
        stop_wrong_guess - the robot made a wrong guess and must recover
    """

    def __init__(self, position_list):
        # define state variables
        self.state = 'idle'
        self.blocks = []
        self.current_block = None
        self.holding = None

        # read registed positions
        try:
            f = open(position_list)
            self.positions = json.load(f)
            self.positions = self.positions["positions"]
            f.close()

        except:
            rospy.logerr("Invalid positions file! Closing...")
            sys.exit(0)

        # set up arm controller service proxy
        rospy.wait_for_service('move_arm_to_pose_goal')
        rospy.wait_for_service('stop_arm')
        self.move_arm_to_pose_goal_proxy = rospy.ServiceProxy('move_arm_to_pose_goal', MoveArmToPoseGoal)
        self.stop_arm_proxy = rospy.ServiceProxy('stop_arm', StopArm)

        # subscribe data derived from sensors
        self.centroids = {"red": None, "dark_blue": None, "light_blue": None, "green": None,
                          "yellow": None, "orange": None, "violet": None, "white": None}

        self.table_centroids_subscriber = rospy.Subscriber("/table_centroids", CentroidList, self.table_centroids_callback)

        self.user_pose = ""
        self.user_pose_subscriber = rospy.Subscriber("/user_pose", PointStamped, self.user_pose_callback)

        rospy.loginfo("Controller Ready")

    def table_centroids_callback(self, msg):
        centroids = msg.points
        color = msg.color
        
        if self.centroids[color] is not None and len(centroids) == 1 and len(self.centroids[color])==0:
            if self.state == "idle" and color != "violet" and self.current_block is None:
                self.current_block = color
                self.state = "picking_up"
            
            if (self.state == "picking_up" or self.state == "moving_closer") and color == "violet":
                self.state = "stop_wrong_guess"

                try:
                    self.stop_arm_proxy()
                except rospy.ServiceException as exc:
                    print("Service did not process request: " + str(exc))
                    return

        self.centroids[color] = centroids

    def user_pose_callback(self, msg):
        old_pose = self.user_pose

        point = msg.point
        if point.y > 240:
            self.user_pose = "left"
        
        elif point.y <= 240:
            self.user_pose = "right"
        
        if self.state == "moving_closer" and old_pose != self.user_pose:
            self.state = "stop_side_switch"
            
            try:
                self.stop_arm_proxy()
            except rospy.ServiceException as exc:
                print("Service did not process request: " + str(exc))
                return

    def loop(self):
        while True:
            if self.state == "idle":
                self.idle_state()
            
            elif self.state == "picking_up":
                self.picking_up_state()
            
            elif self.state == "moving_closer":
                self.moving_closer_state()
            
            elif self.state == "putting_down":
                self.putting_down_state()
            
            elif self.state == "stop_side_switch":
                self.stop_side_switch_state()
            
            elif self.state == "stop_wrong_guess":
                self.stop_wrong_guess_state()

    def idle_state(self):
        pass

    def picking_up_state(self):
        block_name = self.current_block + "1" if self.current_block not in self.blocks else self.current_block + "2"

        self.go_to(f'above_{block_name}')
        # self.arm_gripper_comm.gripper_open_fast()
        self.go_to(f'{block_name}')
        # self.arm_gripper_comm.gripper_close_fast()

        self.holding = self.current_block

        self.go_to(f'above_{block_name}')

        self.go_to('retreat')

        if self.state == "picking_up":
            self.state = "moving_closer"
    
    def moving_closer_state(self):
        if self.user_pose == "left":
            self.go_to("above_table2")
        elif self.user_pose == "right":
            self.go_to("above_table1")
        
        if self.state == "moving_closer":
            self.state = "putting_down"

    def putting_down_state(self):
        self.blocks.append(self.holding)

        self.holding = None

        self.current_block = None

        if len(self.blocks) % 3 == 0:
            self.blocks = []

        if self.user_pose == "left":
            self.go_to("table2")
            # self.arm_gripper_comm.gripper_open_fast()
            self.go_to("above_table2")
        else:# self.user_pose == "right":
            self.go_to("table1")
            # self.arm_gripper_comm.gripper_open_fast()
            self.go_to("above_table1")
        
        self.go_to('retreat')

        if self.state == "putting_down":
            self.state = 'idle'

    def stop_side_switch_state(self):
        if self.state == "stop_side_switch":
            self.state = "moving_closer"

    def stop_wrong_guess_state(self):
        if self.holding is not None:
            block_name = self.holding + "1" if self.holding not in self.blocks else "2"

            self.go_to(f"above_{block_name}")
            self.go_to(f"{block_name}")
            # self.arm_gripper_comm.gripper_open_fast()
            self.go_to(f"above_{block_name}")

            self.holding = None

        self.current_block = None
        
        if self.state == "stop_wrong_guess":
            self.state = "idle"

    def go_to(self, pos):
        pos = self.positions[pos]

        req = MoveArmToPoseGoalRequest(translation=(pos[0], pos[1], pos[2]-0.26), quaternions=(pos[3], pos[4], pos[5], pos[6]),
                                       velocity=0.3, acceleration=0.3)

        try:
            resp = self.move_arm_to_pose_goal_proxy(req)
            print(resp)

        except rospy.ServiceException as exc:
            print("Service did not process request: " + str(exc))
            sys.exit(0)


def main():
    # ---------------------------------------------------
    # INITIALIZATION
    # ---------------------------------------------------
    default_node_name = 'base_controller'
    rospy.init_node(default_node_name, anonymous=False)

    quaternion_poses = rospy.get_param(rospy.search_param('quaternion_poses'))

    base_controller = BaseController(position_list = quaternion_poses)

    base_controller.loop()

    rospy.spin()

    # base_controller.arm_gripper_comm.gripper_disconnect()


if __name__ == '__main__':
    main()
