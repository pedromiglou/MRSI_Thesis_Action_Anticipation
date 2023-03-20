#!/usr/bin/env python3

import rospy
from pamaral_color_image_processing.msg import PointListStamped
from geometry_msgs.msg import PointStamped
import sys
import argparse
import json
from larcc_classes.ur10e_control.ArmGripperComm import ArmGripperComm
import time


class DecisionMakingBlock(object):
    # states = ['idle', 'picking_up', 'moving_closer', 'putting_down', 'retreating']

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
        self.red_centroids_subscriber = rospy.Subscriber("/red_centroids", PointListStamped, self.red_centroids_callback)
        self.green_centroids_subscriber = rospy.Subscriber("/green_centroids", PointListStamped, self.green_centroids_callback)

        self.user_pose = ""
        self.user_pose_subscriber = rospy.Subscriber("/user_pose", PointStamped, self.user_pose_callback)

        # define pieces and states
        self.pieces = []
        self.state = 'idle'

        self.loop()


    def red_centroids_callback(self, msg):
        red_centroids = msg.points
        if len(red_centroids) > len(self.red_centroids):
            self.red_centroids = red_centroids
            
            if self.state == "idle":
                if len(self.pieces) > 0 and self.pieces[0][1] == "R":
                    self.state = "picking_up"
                else:
                    self.pieces = ["8R", "4R"]
                    self.state = "picking_up"

        else:
            self.red_centroids = red_centroids
    

    def green_centroids_callback(self, msg):
        green_centroids = msg.points
        if len(green_centroids) > len(self.green_centroids):
            self.green_centroids = green_centroids

            if self.state == "idle":
                if len(self.pieces) > 0 and self.pieces[0][1] == "G":
                    self.state = "picking_up"
                else:
                    self.pieces = ["8G", "4G", "2G"]
                    self.state = "picking_up"
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
            self.state = "retreating"
            self.arm_gripper_comm.stop_arm()
    

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
            
            elif self.state == "retreating":
                self.retreating_state()


    def idle_state(self):
        pass


    def picking_up_state(self):
        self.go_to(f'above_{self.pieces[0]}')
        self.arm_gripper_comm.gripper_open_fast()
        self.go_to(f'{self.pieces[0]}')
        self.arm_gripper_comm.gripper_close_fast()
        self.go_to(f'above_{self.pieces[0]}')
        self.go_to('retreat')

        self.pieces = self.pieces[1:]

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
        if self.user_pose == "left":
            self.go_to("table2")
            self.arm_gripper_comm.gripper_open_fast()
            self.go_to("above_table2")
        elif self.user_pose == "right":
            self.go_to("table1")
            self.arm_gripper_comm.gripper_open_fast()
            self.go_to("above_table1")
        
        self.go_to('retreat')

        if self.state == "putting_down":
            self.state = 'idle'

    
    def retreating_state(self):
        #self.go_to('retreat')
        self.arm_gripper_comm.stop_arm()

        if self.state == "retreating":
            self.state = "moving_closer"


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
