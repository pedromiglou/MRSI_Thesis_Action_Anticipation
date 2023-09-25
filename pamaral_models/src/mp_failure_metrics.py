#!/usr/bin/env python3

import numpy as np
import os
import rosbag
import rospy

from sensor_msgs.msg import Image

from pamaral_models.msg import PointList


class MPFailureMetrics:

    def __init__(self, input_folder):
        # Iterate over all files in the folder
        self.bag_paths = []
        for filename in os.listdir(input_folder):
            # Create the absolute path to the file
            bag_path = os.path.join(input_folder, filename)

            # Check if the file path is a file (not a directory)
            if os.path.isfile(bag_path) and bag_path.endswith(".bag") and not bag_path.endswith("orig.bag"):
                self.bag_paths.append(bag_path)
        
        self.image_publisher = rospy.Publisher("front_camera/color/image_raw", Image, queue_size=300)
        self.right_hand_keypoints_sub = rospy.Subscriber("right_hand_keypoints", PointList, self.right_hand_keypoints_callback)

    def right_hand_keypoints_callback(self, msg):
        if len(msg.points)>0:
            points = [[p.x, p.y, p.z] for p in msg.points]
            points = np.array(points)

            self.frames_with_points += 1
        
        else:
            self.empty_streak += 1
            self.longest_empty_streak = max(self.longest_empty_streak, self.empty_streak)
        
        self.num_messages_processed += 1

        # if finished playing the bag file
        if self.num_messages_processed == self.num_messages:
            # print metrics
            print("File: " + self.bag_paths[0])
            print("Frames with points percentage: " + str(self.frames_with_points/self.num_messages))
            print("Longest empty streak: " + str(self.longest_empty_streak))

            self.bag_paths = self.bag_paths[1:]
            self.play_next_bag_file()
    
    def play_next_bag_file(self):
        # reset metrics
        self.frames_with_points = 0
        self.longest_empty_streak = 0
        self.empty_streak = 0
        self.num_messages_processed = 0

        bag = rosbag.Bag(self.bag_paths[0])

        self.num_messages = bag.get_message_count()

        for topic, msg, t in bag.read_messages():
            self.image_publisher.publish(msg)


def main():
    default_node_name = 'dataset_processing'
    rospy.init_node(default_node_name, anonymous=False)

    input_folder = rospy.get_param(rospy.search_param('input_folder'))

    dataset_processing = MPFailureMetrics(input_folder=input_folder)

    input()

    dataset_processing.play_next_bag_file()

    rospy.spin()


if __name__ == '__main__':
    main()
