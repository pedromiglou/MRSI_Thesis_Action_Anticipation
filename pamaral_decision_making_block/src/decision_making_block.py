#!/usr/bin/env python3

import argparse
import json
import sys

from larcc_classes.ur10e_control.ArmGripperComm import ArmGripperComm
import rospy
import time

from pamaral_color_segmentation.msg import PointListStamped
from geometry_msgs.msg import PointStamped


class Decision_Making_Block:

    def __init__(self, args) -> None:
        self.path = "/home/miglou/catkin_ws/src/MRSI_Thesis/pamaral_decision_making_block/config/"

        self.state = {"holdind": "", "position": ""}

        self.moving = ""

        self.pickedup_green = 0
        self.pickedup_red = 0
        self.green_visible = 100
        self.red_visible = 100

        self.red_centroids = []
        self.green_centroids = []

        self.red_centroids_subscriber = rospy.Subscriber("/red_centroids", PointListStamped, self.red_centroids_callback)
        self.green_centroids_subscriber = rospy.Subscriber("/green_centroids", PointListStamped, self.green_centroids_callback)

        self.user_pose = None
        self.side = "left"
        self.user_pose_subscriber = rospy.Subscriber("/user_pose", PointStamped, self.user_pose_callback)

        try:
            f = open(self.path + args['position_list'] + ".json")
            self.positions = json.load(f)
            self.positions = self.positions["positions"]
            f.close()

        except:
            rospy.logerr("Invalid positions file! Closing...")
            sys.exit(0)

        self.arm_gripper_comm = ArmGripperComm()

        time.sleep(0.2)

        self.arm_gripper_comm.gripper_connect()

        if not self.arm_gripper_comm.state_dic["activation_completed"]: 
            self.arm_gripper_comm.gripper_init()
        
        self.loop()
    

    def user_pose_callback(self, msg):
        point = msg.point
        if point.y > 200:
            self.side = "left"
        
        elif point.y <= 200:
            self.side = "right"
        
        if len(self.moving)>0:
            if self.moving != self.side:
                self.arm_gripper_comm.stop_arm()
                self.moving = ""
                self.do_json("retreat.json")


    def red_centroids_callback(self, msg):
        self.red_centroids = msg.points
        self.red_centroids_seq = msg.header.seq
    

    def green_centroids_callback(self, msg):
        self.green_centroids = msg.points
        self.green_centroids_seq = msg.header.seq
    

    def loop(self):
        green_seq = -1
        red_seq = -1
        while True:
            if self.state["position"] == "retreat" and self.state["holding"]:
                if self.side == "left":
                    self.moving = "left"
                    self.do_json("putclose2.json")
                
                else:
                    self.moving = "right"
                    self.do_json("putclose1.json")
                
                self.moving = ""
                self.do_json("retreat.json")

                if self.state["holding"] == "green":
                    self.pickedup_green += 1
                
                elif self.state["holding"] == "red":
                    self.pickedup_red += 1

            elif len(self.green_centroids) > 0 and self.green_centroids_seq > green_seq:
                green_seq = self.green_centroids_seq

                if self.green_visible < len(self.green_centroids):
                    if self.pickedup_green==0:
                        self.do_json("pickup_8G.json")

                    if self.pickedup_green==1:
                        self.do_json("pickup_4G.json")

                    elif self.pickedup_green==2:
                        self.do_json("pickup_2G.json")
                    
                    if self.pickedup_green < 3:
                        self.do_json("retreat.json")
                        self.state["position"] = "retreat"
                        self.state["holding"] = "green"
                    
                self.green_visible = len(self.green_centroids)

            elif len(self.red_centroids) > 0 and self.red_centroids_seq > red_seq:
                red_seq = self.red_centroids_seq

                if self.red_visible < len(self.red_centroids):
                    if self.pickedup_red==0:
                        self.do_json("pickup_8R.json")

                    elif self.pickedup_red==1:
                        self.do_json("pickup_4R.json")
                    
                    if self.pickedup_red < 2:
                        self.do_json("retreat.json")
                        self.state["position"] = "retreat"
                        self.state["holding"] = "red"

                self.red_visible = len(self.red_centroids)


    def do_json(self, filename) -> None:
        try:
            f = open(self.path + filename)
            config = json.load(f)
            f.close()
        
        except:
            rospy.logerr("Invalid file! Closing...")
            self.arm_gripper_comm.gripper_disconnect()
            sys.exit(0)

        for pos, gripper in config["positions"]:
            pos = self.positions[pos]
            self.arm_gripper_comm.move_arm_to_joints_state(pos[0], pos[1], pos[2], pos[3], pos[4], pos[5])

            if gripper == 1:
                self.arm_gripper_comm.gripper_open_fast()
            if gripper == -1:
                self.arm_gripper_comm.gripper_close_fast()


def main():
    # ---------------------------------------------------
    # INITIALIZATION
    # ---------------------------------------------------
    default_node_name = 'decision_making_block'
    rospy.init_node(default_node_name, anonymous=False)

    parser = argparse.ArgumentParser(description="Arguments for trainning script")
    parser.add_argument("-pl", "--position_list", type=str, default="positions",
                    help="It is the name of the configuration JSON containing the list of positions in the config directory of this package")
    parser.add_argument("-m", "--movement", type=str, default="",
                        help="'R', 'G' or ''")

    args = vars(parser.parse_args())

    decision_making_block = Decision_Making_Block(args)
    rospy.spin()

    decision_making_block.arm_gripper_comm.gripper_disconnect()


if __name__ == '__main__':
    main()
