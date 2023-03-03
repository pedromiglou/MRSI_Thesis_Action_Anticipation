#!/usr/bin/env python3
import argparse
import json
import os
import sys

from larcc_classes.ur10e_control.ArmGripperComm import ArmGripperComm
import rospy
import time


class Robot_Controller:

    def __init__(self, args) -> None:
        self.path = "/home/miglou/catkin_ws/src/MRSI_Thesis/robot_movement/config/"

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

        if args['movement'] == "":
            res = os.listdir(self.path)
            res.remove(args['position_list'] + ".json")

            while True:
                i = 0

                for file in res:
                    print(f'[{i}]:' + file)
                    i += 1

                idx = input("Select idx from test json: ")

                self.do_json(res[int(idx)])

        elif args['movement'] == 'G':
            self.give_green_pieces()
        
        elif args['movement'] == 'R':
            self.give_red_pieces()
    

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
    

    def give_green_pieces(self):
        time.sleep(6)

        self.do_json("pickup_8G.json")

        self.do_json("putclose.json")

        time.sleep(3)

        self.do_json("pickup_4G.json")

        self.do_json("putclose.json")

        time.sleep(3)

        self.do_json("pickup_2G.json")

        self.do_json("putclose.json")
    

    def give_red_pieces(self):
        time.sleep(6)

        self.do_json("pickup_8R.json")

        self.do_json("putclose.json")

        time.sleep(3)

        self.do_json("pickup_4R.json")

        self.do_json("putclose.json")


def main():
    # ---------------------------------------------------
    # INITIALIZATION
    # ---------------------------------------------------
    default_node_name = 'my_robot_controller'
    rospy.init_node(default_node_name, anonymous=False)

    parser = argparse.ArgumentParser(description="Arguments for trainning script")
    parser.add_argument("-pl", "--position_list", type=str, default="positions",
                    help="It is the name of the configuration JSON containing the list of positions in the config directory of this package")
    parser.add_argument("-m", "--movement", type=str, default="",
                        help="'R', 'G' or ''")

    args = vars(parser.parse_args())

    controller = Robot_Controller(args)
    rospy.spin()

    controller.arm_gripper_comm.gripper_disconnect()


if __name__ == '__main__':
    main()
