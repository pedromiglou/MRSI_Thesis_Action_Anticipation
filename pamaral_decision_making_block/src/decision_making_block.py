#!/usr/bin/env python3

import rospy
from pamaral_color_image_processing.msg import PointListStamped
from geometry_msgs.msg import PointStamped
import sys
import argparse
import json
from larcc_classes.ur10e_control.ArmGripperComm import ArmGripperComm
import time
from std_msgs.msg import String


class DecisionMakingBlock:
    """State machine to decide what should be the action of the robot.

    States:
        idle - waiting for the desired sequence of objects
        picking_up - picking up a given object
        waiting - waiting for the user to finish so it can give him the next piece
        moving_closer - moving closer to the user
        putting_down - putting down the object close to the user
        stop_side_switch - while the robot is stopping because the user changed sides
        stop_wrong_guess - the robot made a wrong guess and must recover
    """

    def __init__(self, position_list):
        # read registed positions
        self.path = "/home/miglou/catkin_ws/src/MRSI_Thesis/pamaral_decision_making_block/config/"

        try:
            f = open(self.path + position_list + ".json")
            self.positions = json.load(f)
            self.positions = self.positions["positions"]
            f.close()

        except:
            rospy.logerr("Invalid positions file! Closing...")
            sys.exit(0)

        # start the robot
        self.arm_gripper_comm = ArmGripperComm()

        time.sleep(0.2)

        self.arm_gripper_comm.gripper_connect()

        if not self.arm_gripper_comm.state_dic["activation_completed"]: 
            self.arm_gripper_comm.gripper_init()

        # subscribe data derived from sensors
        self.red_centroids = []
        self.green_centroids = []
        self.orientation = ""
        self.red_centroids_subscriber = rospy.Subscriber("/red_centroids", PointListStamped, self.red_centroids_callback)
        self.green_centroids_subscriber = rospy.Subscriber("/green_centroids", PointListStamped, self.green_centroids_callback)
        self.orientation_subscriber = rospy.Subscriber("/orientation", String, self.orientation_callback)

        self.user_pose = ""
        self.user_pose_subscriber = rospy.Subscriber("/user_pose", PointStamped, self.user_pose_callback)

        # define pieces and states
        self.pieces = []
        self.pieces_index = 0
        self.holding = ''
        self.state = 'idle'

        self.loop()
    

    def orientation_callback(self, msg):
        self.orientation = msg.data

        if len(self.pieces)>0:
            if self.pieces[1] == "4G" and self.pieces_index == 1 and self.orientation == "parallel":
                self.pieces = ["8G", "8R", "4R", "4G", "2G"]

                #if self.state == "waiting" or self.state == "moving_closer":
                if len(self.holding) > 0:
                    self.state = "stop_wrong_guess"
                    self.arm_gripper_comm.stop_arm()
            
            elif self.pieces[1] == "8R" and self.pieces_index == 1 and self.orientation == "perpendicular":
                self.pieces = ["8G", "4G", "2G"]

                #if self.state == "waiting" or self.state == "moving_closer":
                if len(self.holding) > 0:
                    self.state = "stop_wrong_guess"
                    self.arm_gripper_comm.stop_arm()


    def red_centroids_callback(self, msg):
        red_centroids = msg.points
        if len(red_centroids) > len(self.red_centroids):
            self.red_centroids = red_centroids

            if self.state == "idle" and self.pieces_index == 0:
                self.pieces = ["8R", "4R"]
                self.state = "picking_up"
            
            if self.pieces_index < len(self.pieces):
                if self.state == "waiting":
                    self.state = "moving_closer"

        else:
            self.red_centroids = red_centroids
    

    def green_centroids_callback(self, msg):
        green_centroids = msg.points
        if len(green_centroids) > len(self.green_centroids):
            self.green_centroids = green_centroids

            if self.state == "idle" and self.pieces_index == 0:
                self.pieces = ["8G", "4G", "2G"]
                self.state = "picking_up"

            if self.pieces_index < len(self.pieces):                
                if self.state == "waiting":
                    self.state = "moving_closer"

        else:
            self.green_centroids = green_centroids
    

    def user_pose_callback(self, msg):
        old_pose = self.user_pose

        point = msg.point
        if point.y > 200:
            self.user_pose = "left"
        
        elif point.y <= 200:
            self.user_pose = "right"
        
        if self.state == "moving_closer" and old_pose != self.user_pose:
            self.state = "stop_side_switch"
            self.arm_gripper_comm.stop_arm()
    

    def loop(self):
        while True:
            if self.state == "idle":
                self.idle_state()
            
            elif self.state == "picking_up":
                self.picking_up_state()
            
            elif self.state == "waiting":
                self.waiting_state()
            
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
        p = self.pieces[self.pieces_index]
        self.go_to(f'above_{p}')
        self.arm_gripper_comm.gripper_open_fast()
        self.go_to(f'{p}')
        self.arm_gripper_comm.gripper_close_fast()
        self.go_to(f'above_{p}')

        self.holding = p

        self.go_to('retreat')

        if self.state == "picking_up":
            self.state = "waiting"
    

    def waiting_state(self):
        if self.pieces_index == 0:
            self.state = "moving_closer"
        else:
            pass
    
    
    def moving_closer_state(self):
        if self.user_pose == "left":
            self.go_to("above_table2")
        elif self.user_pose == "right":
            self.go_to("above_table1")
        
        if self.state == "moving_closer":
            self.state = "putting_down"


    def putting_down_state(self):
        if self.user_pose == "left":
            self.go_to("table2")
            self.arm_gripper_comm.gripper_open_fast()
            self.go_to("above_table2")
        elif self.user_pose == "right":
            self.go_to("table1")
            self.arm_gripper_comm.gripper_open_fast()
            self.go_to("above_table1")
        
        self.holding = ''

        self.pieces_index += 1
        
        self.go_to('retreat')

        if self.state == "putting_down":
            if self.pieces_index < len(self.pieces):
                self.state = 'picking_up'
            else:
                self.state = 'idle'

    
    def stop_side_switch_state(self):
        #self.go_to('retreat')
        self.arm_gripper_comm.stop_arm()

        if self.state == "stop_side_switch":
            self.state = "moving_closer"
    

    def stop_wrong_guess_state(self):
        #self.arm_gripper_comm.stop_arm()

        time.sleep(0.5)

        if self.holding != '':
            self.go_to(f"above_{self.holding}")
            self.go_to(f"{self.holding}")
            self.arm_gripper_comm.gripper_open_fast()
            self.go_to(f"above_{self.holding}")

            self.holding = ''
        
        time.sleep(0.5)
        
        if self.state == "stop_wrong_guess":
            self.state = "picking_up"


    def go_to(self, pos):
        pos = self.positions[pos]
        self.arm_gripper_comm.move_arm_to_joints_state(pos[0], pos[1], pos[2], pos[3], pos[4], pos[5])


def main():
    # ---------------------------------------------------
    # INITIALIZATION
    # ---------------------------------------------------
    default_node_name = 'decision_making_block'
    rospy.init_node(default_node_name, anonymous=False)

    parser = argparse.ArgumentParser(description="Arguments for trainning script")
    parser.add_argument("-pl", "--position_list", type=str, default="positions",
                    help="It is the name of the configuration JSON containing the list of positions in the config directory of this package")

    args, _ = parser.parse_known_args()

    decision_making_block = DecisionMakingBlock(position_list = args.position_list)

    rospy.spin()

    decision_making_block.arm_gripper_comm.gripper_disconnect()


if __name__ == '__main__':
    main()
