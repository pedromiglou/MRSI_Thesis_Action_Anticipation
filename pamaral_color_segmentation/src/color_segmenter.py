#!/usr/bin/env python3

from cv_bridge import CvBridge
import cv2
from geometry_msgs.msg import Point
import json
import numpy as np
import rospy
from sensor_msgs.msg import Image
from sklearn.cluster import DBSCAN
from pamaral_color_segmentation.msg import PointListStamped


class Color_Segmenter:
    """This node should receive color images and detect objects using color segmentation."""

    def __init__(self) -> None:
        self.path = "/home/miglou/catkin_ws/src/MRSI_Thesis/pamaral_color_segmentation/config/"

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

        self.bridge = CvBridge()
        self.cimage = None
        self.red_mask = None
        self.green_mask = None
        self.cimage_subscriber = rospy.Subscriber("/camera/color/image_raw", Image, self.cimage_callback)
        
        self.showImage()
    

    def cimage_callback(self, msg):
        try:
            img_bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cimage = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        except:
            print("Error reading color image")
        
        # process red and green mask
        red_mask = cv2.inRange(cimage, self.red_mins, self.red_maxs)
        green_mask = cv2.inRange(cimage, self.green_mins, self.green_maxs)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10,10))

        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

        # detect red objects
        red_pieces = cv2.connectedComponentsWithStats(red_mask)
        (numLabels, labels, stats, centroids) = red_pieces

        red_pieces = []
        for i in range(len(centroids)):
            area = stats[i, cv2.CC_STAT_AREA]
            (cX, cY) = centroids[i]
            cX, cY = int(cX), int(cY)

            if area > 250 and cX > 100 and cY<415:
                red_pieces.append((cX, cY))
        
        self.red_pieces.append(red_pieces[1:])
        self.red_pieces = self.red_pieces[-5:]
        
        # detect green objects
        green_pieces = cv2.connectedComponentsWithStats(green_mask)
        (numLabels, labels, stats, centroids) = green_pieces

        green_pieces = []
        for i in range(len(centroids)):
            area = stats[i, cv2.CC_STAT_AREA]
            (cX, cY) = centroids[i]
            cX, cY = int(cX), int(cY)

            if area > 250 and cX > 100 and cY<415:
                green_pieces.append((cX, cY))
        
        self.green_pieces.append(green_pieces[1:])
        self.green_pieces = self.green_pieces[-5:]

        # draw centroids
        cimage = cv2.cvtColor(cimage, cv2.COLOR_HSV2BGR)

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
            
            for c in green_centroids:
                cv2.putText(cimage, '+', (c[0], c[1]), cv2.FONT_ITALIC, 1, (0,255,0), 2, cv2.LINE_8)
        
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
            
            for c in red_centroids:
                cv2.putText(cimage, '+', (c[0], c[1]), cv2.FONT_ITALIC, 1, (0,0,255), 2, cv2.LINE_8)
        
        self.cimage, self.red_mask, self.green_mask = cimage, red_mask, green_mask

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
            if self.cimage is not None and self.red_mask is not None and self.green_mask is not None:
                cv2.imshow("Color Image", self.cimage)
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
    default_node_name = 'color_segmenter'
    rospy.init_node(default_node_name, anonymous=False)

    color_segmenter = Color_Segmenter()

    rospy.spin()


if __name__ == '__main__':
    main()
