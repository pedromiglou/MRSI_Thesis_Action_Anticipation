#!/usr/bin/env python3

import argparse
import json
import rospy
import sys
import time

from geometry_msgs.msg import PointStamped

from larcc_classes.ur10e_control.ArmGripperComm import ArmGripperComm
from pamaral_color_image_processing.msg import CentroidList
from pamaral_decision_making_block.srv import GetProbabilities, GetProbabilitiesRequest


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
        # define state variables
        self.state = 'idle'
        self.blocks = []
        self.current_block = []
        self.holding = None

        # read registed positions
        self.path = "/home/pedroamaral/catkin_ws/src/MRSI_Thesis_Action_Anticipation/pamaral_decision_making_block/config/quaternion_poses/"

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
        self.centroids = {"red": None, "dark_blue": None, "light_blue": None, "green": None,
                          "yellow": None, "orange": None, "violet": None, "white": None}

        self.centroids_subscriber = rospy.Subscriber("/centroids", CentroidList, self.centroids_callback)

        self.user_pose = ""
        self.user_pose_subscriber = rospy.Subscriber("/user_pose", PointStamped, self.user_pose_callback)

        # wait for database service
        rospy.wait_for_service('get_probabilities')
        self.get_probabilities_proxy = rospy.ServiceProxy('get_probabilities', GetProbabilities)
        
        rospy.loginfo("Class Initialized")

        self.loop()


    def centroids_callback(self, msg):
        centroids = msg.points

        if self.centroids[msg.color] is not None and len(centroids) == 1 and len(self.centroids[msg.color])==0:
            self.centroids[msg.color] = centroids

            if self.state == "idle" and (self.current_block is None or len(self.current_block)==0) and msg.color != "violet":
                self.current_block = [msg.color]
                self.state = "picking_up"
            
            if (self.state == "picking_up" or self.state == "moving_closer") and msg.color == "violet":
                self.state = "stop_wrong_guess"
                self.arm_gripper_comm.stop_arm()

        else:
            self.centroids[msg.color] = centroids
    

    def user_pose_callback(self, msg):
        old_pose = self.user_pose

        point = msg.point
        if point.y > 240:
            self.user_pose = "left"
        
        elif point.y <= 240:
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
            
            #elif self.state == "waiting":
            #    self.waiting_state()
            
            elif self.state == "moving_closer":
                self.moving_closer_state()
            
            elif self.state == "putting_down":
                self.putting_down_state()
            
            elif self.state == "stop_side_switch":
                self.stop_side_switch_state()
            
            elif self.state == "stop_wrong_guess":
                self.stop_wrong_guess_state()


    def idle_state(self):
        if self.current_block is not None: # get colors from database, else wait for user input
            if len(self.blocks) % 3 == 0:
                return
            elif len(self.blocks) % 3 == 1:
                resp = self.get_probabilities_proxy(GetProbabilitiesRequest(color1=self.blocks[-1], color2=""))
            elif len(self.blocks) % 3 == 2:
                resp = self.get_probabilities_proxy(GetProbabilitiesRequest(color1=self.blocks[-2], color2=self.blocks[-1]))

            colors, probabilities = resp.colors, resp.probabilities

            colors = zip(colors, probabilities)

            colors = sorted(colors, key=lambda x: x[1])
            colors = sorted(colors, key=lambda x: x[0], reverse=True)

            self.current_block = [c[0] for c in colors if c[1] > 0]

            if self.state == "idle":
                if len(self.current_block)>0:
                    self.state = "picking_up"
                else:
                    self.current_block = None


    def picking_up_state(self):
        p = self.current_block[0]
        self.go_to(f'above_{p}1')
        self.arm_gripper_comm.gripper_open_fast()
        self.go_to(f'{p}1')
        self.arm_gripper_comm.gripper_close_fast()

        self.holding = p

        self.go_to(f'above_{p}1')
        self.go_to('retreat')

        if self.state == "picking_up":
            self.state = "moving_closer"
    

    #def waiting_state(self):
    #    if self.pieces_index == 0:
    #        self.state = "moving_closer"
    #    else:
    #        pass
    
    
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

        self.current_block = []

        if self.user_pose == "left":
            self.go_to("table2")
            self.arm_gripper_comm.gripper_open_fast()
            self.go_to("above_table2")
        elif self.user_pose == "right":
            self.go_to("table1")
            self.arm_gripper_comm.gripper_open_fast()
            self.go_to("above_table1")

        #if len(self.blocks) == 3:
        #    self.blocks = []

        #if len(self.blocks)==1:
        #    flags = get_flags(self.blocks[0])
        #    self.piece = flags[0].color2
        #elif len(self.blocks)==2:
        #    flags = get_flags(self.blocks[0], self.blocks[1])
        #    self.piece = flags[0].color3
        
        self.go_to('retreat')

        if self.state == "putting_down":
            self.state = 'idle'

    
    def stop_side_switch_state(self):
        #self.go_to('retreat')
        #self.arm_gripper_comm.stop_arm()

        #time.sleep(0)

        if self.state == "stop_side_switch":
            self.state = "moving_closer"
    

    def stop_wrong_guess_state(self):
        #self.arm_gripper_comm.stop_arm()

        #time.sleep(0)

        if self.holding is not None:
            self.go_to(f"above_{self.holding}1")
            self.go_to(f"{self.holding}1")
            self.arm_gripper_comm.gripper_open_fast()
            self.go_to(f"above_{self.holding}1")

            #self.blocks = self.blocks[-1:]

            self.holding = None

        self.current_block = self.current_block[1:]
        
        #time.sleep(0.5)
        
        if self.state == "stop_wrong_guess":
            if len(self.current_block) == 0:
                self.state = "idle"
                self.current_block = None
            else:
                self.state = "picking_up"


    def go_to(self, pos):
        pos = self.positions[pos]
        self.arm_gripper_comm.move_arm_to_pose_goal(pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6], vel=0.3, a=0.3)


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
