#!/usr/bin/env python3

import argparse
import cv2
import json
import numpy as np
import rospy

from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from sklearn.cluster import DBSCAN
from std_msgs.msg import String

from pamaral_color_image_processing.msg import PointListStamped


class ObjectColorSegmenter:
    """This node should receive color images and detect objects using color segmentation."""

    def __init__(self, debug):
        self.debug = debug
        
        self.path = "/home/miglou/catkin_ws/src/MRSI_Thesis/pamaral_color_image_processing/config/"

        f = open(self.path + "red.json")
        red = json.load(f)
        f.close()

        f = open(self.path + "green.json")
        green = json.load(f)
        f.close()

        self.red_mins = np.array([red['limits']['h']['min'], red['limits']['s']['min'], red['limits']['v']['min']])
        self.red_maxs = np.array([red['limits']['h']['max'], red['limits']['s']['max'], red['limits']['v']['max']])
        self.green_mins = np.array([green['limits']['h']['min'], green['limits']['s']['min'], green['limits']['v']['min']])
        self.green_maxs = np.array([green['limits']['h']['max'], green['limits']['s']['max'], green['limits']['v']['max']])

        self.green_pieces = []
        self.red_pieces = []

        self.red_centroids_publisher = rospy.Publisher("/red_centroids", PointListStamped, queue_size=1)
        self.green_centroids_publisher = rospy.Publisher("/green_centroids", PointListStamped, queue_size=1)
        self.orientation_publisher = rospy.Publisher("/orientation", String, queue_size=1)

        self.bridge = CvBridge()
        self.red_mask = None
        self.green_mask = None
        self.cimage_subscriber = rospy.Subscriber("/camera/color/image_raw", Image, self.cimage_callback)
        
        if self.debug:
            self.showImage()
    

    def cimage_callback(self, msg):
        try:
            img_bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cimage = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        except:
            print("Error reading color image")
        
        # process red and green mask
        self.red_mask = cv2.inRange(cimage, self.red_mins, self.red_maxs)
        self.green_mask = cv2.inRange(cimage, self.green_mins, self.green_maxs)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10,10))

        self.red_mask = cv2.morphologyEx(self.red_mask, cv2.MORPH_CLOSE, kernel)
        self.green_mask = cv2.morphologyEx(self.green_mask, cv2.MORPH_CLOSE, kernel)

        # detect red objects
        red_pieces = cv2.connectedComponentsWithStats(self.red_mask)
        (numLabels, labels, stats, centroids) = red_pieces

        red_pieces = []
        for i in range(len(centroids)):
            area = stats[i, cv2.CC_STAT_AREA]
            (cX, cY) = centroids[i]
            cX, cY = int(cX), int(cY)

            if area > 250 and cX > 180 and cY<415:
                red_pieces.append((cX, cY))
        
        self.red_pieces.append(red_pieces[1:])
        self.red_pieces = self.red_pieces[-3:]
        
        # detect green objects
        green_pieces = cv2.connectedComponentsWithStats(self.green_mask)
        (numLabels, labels, stats, centroids) = green_pieces

        green_pieces = []
        for i in range(len(centroids)):
            area = stats[i, cv2.CC_STAT_AREA]
            (cX, cY) = centroids[i]
            cX, cY = int(cX), int(cY)

            if area > 250 and cX > 180 and cY<415:
                green_pieces.append((cX, cY))

                if area < 2000:
                    piece_mask = labels == i

                    s = piece_mask.shape

                    top, bottom, left, right = np.inf,0,np.inf,0
                    
                    for i in range(s[0]):
                        for j in range(s[1]):
                            if piece_mask[i,j]:
                                if i < top:
                                    top = i
                                if i > bottom:
                                    bottom = i
                                if j < left:
                                    left = j
                                if j > right:
                                    right = j
                    
                    if bottom-top > right-left:
                        self.orientation_publisher.publish("perpendicular")
                    else:
                        self.orientation_publisher.publish("parallel")
        
        self.green_pieces.append(green_pieces[1:])
        self.green_pieces = self.green_pieces[-3:]

        # obtain centroids
        green_centroids = []
        red_centroids = []

        green_pieces = np.array([p for pl in self.green_pieces for p in pl])
        red_pieces = np.array([p for pl in self.red_pieces for p in pl])

        if len(green_pieces) > 0:
            clustering = DBSCAN(eps=15, min_samples=2).fit(green_pieces)

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
        
        if len(red_pieces) > 0:
            clustering = DBSCAN(eps=15, min_samples=2).fit(red_pieces)

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

        # publish centroids
        msg = PointListStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "/camera/color/image_raw"
        msg.points = [Point(c[0],c[1],0) for c in red_centroids]
        self.red_centroids_publisher.publish(msg)

        msg = PointListStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "/camera/color/image_raw"
        msg.points = [Point(c[0],c[1],0) for c in green_centroids]
        self.green_centroids_publisher.publish(msg)


    def showImage(self):
        while True:
            if self.red_mask is not None and self.green_mask is not None:
                cv2.imshow("Red Mask", self.red_mask)
                cv2.imshow("Green Mask", self.green_mask)

                key = cv2.waitKey(100)

                if key == ord('q'):  # q for quit
                    print('You pressed q ... aborting')
                    break


def main():
    # ---------------------------------------------------
    # INITIALIZATION
    # ---------------------------------------------------
    default_node_name = 'object_color_segmenter'
    rospy.init_node(default_node_name, anonymous=False)

    parser = argparse.ArgumentParser(description="Arguments for object color segmenter")
    parser.add_argument("-d", "--debug", action='store_true',
                    help="if present, then the masks are shown")

    args, _ = parser.parse_known_args()

    object_color_segmenter = ObjectColorSegmenter(debug = args.debug)

    rospy.spin()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
