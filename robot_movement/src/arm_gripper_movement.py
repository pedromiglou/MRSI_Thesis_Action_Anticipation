#!/usr/bin/env python3
import argparse
import json
import os
import sys

#from config.definitions import ROOT_DIR
from larcc_classes.ur10e_control.ArmGripperComm import ArmGripperComm
import rospy
import time

parser = argparse.ArgumentParser(description="Arguments for trainning script")
parser.add_argument("-m", "--movement", type=str, default="",
                    help="It is the name of the movement configuration JSON in the config directory of this package")

args = vars(parser.parse_args())

# path = ROOT_DIR + "/use_cases/config/"
path = "/home/miglou/catkin_ws/src/MRSI_Thesis/robot_movement/config/"

rospy.init_node("arm_gripper_movement", anonymous=True)

arm_gripper_comm = ArmGripperComm()

time.sleep(0.2)

arm_gripper_comm.gripper_connect()

#arm_gripper_comm.gripper_status()

if not arm_gripper_comm.state_dic["activation_completed"]: 
    arm_gripper_comm.gripper_init()

if args['movement'] == "":
    res = os.listdir(path)

    while True:
        i = 0

        for file in res:
            print(f'[{i}]:' + file)
            i += 1

        idx = input("Select idx from test json: ")

        try:
            f = open(path + res[int(idx)])
            config = json.load(f)
            f.close()
        
        except:
            arm_gripper_comm.gripper_disconnect()
            sys.exit(0)

        for pos in config["positions"]:
            arm_gripper_comm.move_arm_to_joints_state(pos[0], pos[1], pos[2], pos[3], pos[4], pos[5])

            if pos[6] == 1:
                arm_gripper_comm.gripper_open_fast()
            if pos[6] == -1:
                arm_gripper_comm.gripper_close_fast()

else:
    f = open(path + args["movement"] + '.json')
    config = json.load(f)
    f.close()

    for pos in config["positions"]:
        arm_gripper_comm.move_arm_to_joints_state(pos[0], pos[1], pos[2], pos[3], pos[4], pos[5])

        if pos[6] == 1:
            arm_gripper_comm.gripper_open_fast()
        if pos[6] == -1:
            arm_gripper_comm.gripper_close_fast()

arm_gripper_comm.gripper_disconnect()

# positions down already switched:
# 4R -> [-0.7381933371173304, -0.9487016958049317, 1.5844123999225062, -2.2606059513487757, -1.5595677534686487, 0.8322415351867676]
# 4G -> [-0.6696541945086878, -0.7871526044658204, 1.2948415915118616, -2.1347090206541957, -1.549922291432516, 0.9193868637084961]
# 2G -> [-0.6016314665423792, -0.5837524694255372, 0.8884509245501917, -1.874162336389059, -1.550450627003805, 0.9975481033325195]