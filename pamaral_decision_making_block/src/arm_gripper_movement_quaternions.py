#!/usr/bin/env python3

import json
import rospy
import sys
import time

from larcc_classes.ur10e_control.ArmGripperComm import ArmGripperComm

try:
    f = open("/home/miglou/catkin_ws/src/MRSI_Thesis_Action_Anticipation/pamaral_decision_making_block/config/quaternion_poses/positions.json")
    positions = json.load(f)
    positions = positions["positions"]
    positions = list(positions.items())
    positions = [["open"], ["close"]] + positions
    f.close()

except:
    rospy.logerr("Invalid positions file! Closing...")
    sys.exit(0)

rospy.init_node("arm_gripper_movement", anonymous=True)

arm_gripper_comm = ArmGripperComm()

time.sleep(0.2)

arm_gripper_comm.gripper_connect()

#arm_gripper_comm.gripper_status()

if not arm_gripper_comm.state_dic["activation_completed"]: 
    arm_gripper_comm.gripper_init()

while True:
    i = 0

    for pos in positions:
        print(f'[{i}]:' + pos[0])
        i += 1

    idx = int(input("Select idx: "))

    if idx == 0:
        arm_gripper_comm.gripper_open_fast()

    elif idx == 1:
        arm_gripper_comm.gripper_close_fast()

    else:
        pos = positions[idx][1]
        arm_gripper_comm.move_arm_to_pose_goal(pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6])

arm_gripper_comm.gripper_disconnect()
