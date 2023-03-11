#!/usr/bin/env python3
import argparse
import json
import os
import sys

from larcc_classes.ur10e_control.ArmGripperComm import ArmGripperComm
import rospy
import time

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

import numpy as np
from sklearn.cluster import DBSCAN


class Robot_Controller:

    def __init__(self, args) -> None:
        self.path = "/home/miglou/catkin_ws/src/MRSI_Thesis/robot_movement/config/"

        self.pickedup_green = 0
        self.pickedup_red = 0
        self.green_visible = None
        self.red_visible = None

        self.green_pieces = []
        self.red_pieces = []

        self.bridge = CvBridge()
        self.cimage = None
        self.cimage_subscriber = rospy.Subscriber("/camera/color/image_raw", Image, self.cimage_callback)

        f = open(self.path + "red.json")
        self.red = json.load(f)
        f.close()

        f = open(self.path + "green.json")
        self.green = json.load(f)
        f.close()   

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
        
        self.showImage() 

        # if args['movement'] == "":
        #     res = os.listdir(self.path)
        #     res.remove(args['position_list'] + ".json")

        #     while True:
        #         i = 0

        #         for file in res:
        #             print(f'[{i}]:' + file)
        #             i += 1

        #         idx = input("Select idx from test json: ")

        #         self.do_json(res[int(idx)])

        # elif args['movement'] == 'G':
        #     self.give_green_pieces()
        
        # elif args['movement'] == 'R':
        #     self.give_red_pieces()

        # self.timer = rospy.Timer(rospy.Duration(1), self.analyzeImage)
    

    def cimage_callback(self, msg):
        try:
            img_bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            self.cimage = img
        except:
            print("Error reading image")
            return
    

    def showImage(self):
        while True:
            if self.cimage is not None:
                img = self.cimage.copy()
                self.analyzeImage(img)
                img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

                green_centroids = []
                red_centroids = []

                green_pieces = np.array([p for pl in self.green_pieces[-3:] for p in pl])
                red_pieces = np.array([p for pl in self.red_pieces[-3:] for p in pl])

                if len(green_pieces) > 0:
                    clustering = DBSCAN(eps=15, min_samples=2).fit(green_pieces)

                    green_centroids = []

                    num_centroids = len(set(clustering.labels_))

                    if any([l == -1 for l in clustering.labels_]):
                        num_centroids -= 1

                    # iterate different labels
                    for i in range(num_centroids):
                        l = []
                        # iterate all labels
                        for j in range(len(clustering.labels_)):
                            if clustering.labels_[j] == i:
                                l.append(green_pieces[j])
                        
                        green_centroids.append(l[0])
                    
                    for c in green_centroids:
                        cv2.putText(img, '+', (c[0], c[1]), cv2.FONT_ITALIC, 1, (0,255,0), 2, cv2.LINE_8)
                
                if len(red_pieces) > 0:
                    clustering = DBSCAN(eps=15, min_samples=2).fit(red_pieces)

                    red_centroids = []

                    num_centroids = len(set(clustering.labels_))

                    if any([l == -1 for l in clustering.labels_]):
                        num_centroids -= 1

                    # iterate different labels
                    for i in range(num_centroids):
                        l = []
                        # iterate all labels
                        for j in range(len(clustering.labels_)):
                            if clustering.labels_[j] == i:
                                l.append(red_pieces[j])
                        
                        red_centroids.append(l[0])
                    
                    for c in red_centroids:
                        cv2.putText(img, '+', (c[0], c[1]), cv2.FONT_ITALIC, 1, (0,0,255), 2, cv2.LINE_8)
                
                cv2.imshow("Image", img)
                cv2.imshow("Red Mask", self.red_mask)
                cv2.imshow("Green Mask", self.green_mask)
                
                key = cv2.waitKey(100)

                if key == ord('q'):  # q for quit
                    print('You pressed q ... aborting')
                    break
                
                if len(green_centroids) > 0:
                    if len(self.green_pieces) > 20 and self.green_visible < len(green_centroids):
                        self.green_visible = len(green_centroids)

                        if self.pickedup_green==0:
                            self.do_json("pickup_8G.json")

                            self.do_json("putclose.json")

                        if self.pickedup_green==1:
                            self.do_json("pickup_4G.json")

                            self.do_json("putclose.json")
                        
                        elif self.pickedup_green==2:
                            self.do_json("pickup_2G.json")

                            self.do_json("putclose.json")
                        
                        self.pickedup_green += 1
                    
                    self.green_visible = len(green_centroids)

                if len(red_centroids) > 0:
                    if len(self.red_pieces) > 20 and self.red_visible < len(red_centroids):
                        self.red_visible = len(red_centroids)

                        if self.pickedup_red==0:
                            self.do_json("pickup_8R.json")

                            self.do_json("putclose.json")

                        elif self.pickedup_red==1:
                            self.do_json("pickup_4R.json")

                            self.do_json("putclose.json")
                        
                        self.pickedup_red += 1
                    
                    self.red_visible = len(red_centroids)


    def analyzeImage(self, img):
        red_mins = np.array([self.red['limits']['h']['min'], self.red['limits']['s']['min'], self.red['limits']['v']['min']])
        red_maxs = np.array([self.red['limits']['h']['max'], self.red['limits']['s']['max'], self.red['limits']['v']['max']])
        green_mins = np.array([self.green['limits']['h']['min'], self.green['limits']['s']['min'], self.green['limits']['v']['min']])
        green_maxs = np.array([self.green['limits']['h']['max'], self.green['limits']['s']['max'], self.green['limits']['v']['max']])

        red_mask = cv2.inRange(img, red_mins, red_maxs)
        green_mask = cv2.inRange(img, green_mins, green_maxs)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10,10))

        self.red_mask = cv2.morphologyEx(red_mask,cv2.MORPH_CLOSE,kernel)
        self.green_mask = cv2.morphologyEx(green_mask,cv2.MORPH_CLOSE,kernel)

        red_pieces = cv2.connectedComponentsWithStats(self.red_mask)
        (numLabels, labels, stats, centroids) = red_pieces

        red_pieces = []
        for i in range(len(centroids)):
            area = stats[i, cv2.CC_STAT_AREA]
            (cX, cY) = centroids[i]
            cX, cY = int(cX), int(cY)

            if area > 250 and cX > 100 and cY<415:
                red_pieces.append((cX, cY))
        
        self.red_pieces.append(red_pieces[1:])
        
        green_pieces = cv2.connectedComponentsWithStats(self.green_mask)
        (numLabels, labels, stats, centroids) = green_pieces

        green_pieces = []
        for i in range(len(centroids)):
            area = stats[i, cv2.CC_STAT_AREA]
            (cX, cY) = centroids[i]
            cX, cY = int(cX), int(cY)

            if area > 250 and cX > 100 and cY<415:
                green_pieces.append((cX, cY))
        
        self.green_pieces.append(green_pieces[1:])

        # if self.green_pieces:
        #     self.green_pieces = [[p,2] for p in green_pieces]
        
        # else:
        #     temp = []

        #     # iterate trough new points
        #     for piece in green_pieces:
        #         if all([(p[0]-piece[0])**2 + (p[1]-piece[1])**2 > 50 for p, _ in self.green_pieces]):
        #             temp.append([piece,1])
            
        #     # remove old points
        #     for piece, n in self.green_pieces:
        #         if all([(p[0]-piece[0])**2 + (p[1]-piece[1])**2 > 50 for p in green_pieces]):
        #             if n>1:
        #                 temp.append([piece,-1])
            
        #     # copy remaining points
        #     for piece, n in self.green_pieces:
        #         if any([(p[0]-piece[0])**2 + (p[1]-piece[1])**2 < 50 for p in green_pieces]):
        #             temp.append([piece,2])
            
        #     self.green_pieces = temp
        
        # I could keep the detected points in all the iterations and apply clustering


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
