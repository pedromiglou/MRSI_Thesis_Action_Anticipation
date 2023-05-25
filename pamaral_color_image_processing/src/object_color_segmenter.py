#!/usr/bin/env python3

import cv2
import json
import numpy as np
import rospy

from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from sklearn.cluster import DBSCAN

from pamaral_color_image_processing.msg import CentroidList


class ObjectColorSegmenter:
    """This node should receive color images and detect objects of a certain color using segmentation."""

    def __init__(self, colors_path, color):
        self.color = color

        f = open(f"{colors_path}{self.color}.json")
        c_limits = json.load(f)['limits']
        f.close()

        self.c_mins = np.array([c_limits['h']['min'], c_limits['s']['min'], c_limits['v']['min']])
        self.c_maxs = np.array([c_limits['h']['max'], c_limits['s']['max'], c_limits['v']['max']])

        self.pieces = []

        self.centroids_publisher = rospy.Publisher("/centroids", CentroidList, queue_size=1)

        self.bridge = CvBridge()
        #self.mask_publisher = rospy.Publisher(f"/{self.color}_mask", Image, queue_size=1)

    def analyze_image(self, img):
        # process mask
        mask = cv2.inRange(img, self.c_mins, self.c_maxs)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # detect objects
        (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(mask)

        pieces = []
        for i in range(len(centroids)):
            area = stats[i, cv2.CC_STAT_AREA]
            (cX, cY) = centroids[i]
            cX, cY = int(cX + 193), int(cY + 17)

            if area > 150:
                pieces.append((cX, cY))

        self.pieces.append(pieces[1:])
        self.pieces = self.pieces[-3:]

        # obtain centroids
        centroids = []

        pieces = np.array([p for pl in self.pieces for p in pl])

        if len(pieces) > 0:
            clustering = DBSCAN(eps=15, min_samples=2).fit(pieces)

            num_centroids = len(set(clustering.labels_))

            if any([l == -1 for l in clustering.labels_]):
                num_centroids -= 1

            # iterate different labels
            for i in range(num_centroids):
                l = []
                # iterate all labels
                for j in range(len(clustering.labels_)):
                    if clustering.labels_[j] == i:
                        l.append(pieces[j])

                centroids.append(l[0])

        # publish centroids and mask
        msg = CentroidList()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "/camera/color/image_raw"
        msg.color = self.color
        msg.points = [Point(c[0], c[1], 0) for c in centroids]
        self.centroids_publisher.publish(msg)
        #self.mask_publisher.publish(self.bridge.cv2_to_imgmsg(mask))


class ObjectColorSegmenterManager:

    def __init__(self, colors_path, colors):
        self.colors = colors
        self.segmenters = []

        for c in self.colors:
            self.segmenters.append(ObjectColorSegmenter(colors_path=colors_path, color=c))

        self.bridge = CvBridge()
        self.subscriber = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)

    def image_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        except:
            print("Error reading color image")
            return
        
        # crop to ROI
        img = img[17:474, 193:454]

        # convert to hsv
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        for segmenter in self.segmenters:
            segmenter.analyze_image(img)


def main():
    # ---------------------------------------------------
    # INITIALIZATION
    # ---------------------------------------------------
    default_node_name = 'object_color_segmenter'
    rospy.init_node(default_node_name, anonymous=False)

    colors_path = rospy.get_param(rospy.search_param('colors_path'))
    colors = rospy.get_param(rospy.search_param('colors')).split(";")

    ObjectColorSegmenterManager(colors_path, colors)

    rospy.spin()


if __name__ == '__main__':
    main()
