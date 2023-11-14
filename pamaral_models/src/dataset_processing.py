#!/usr/bin/env python3

import csv
import numpy as np
import os
import rosbag
import rospy

from sensor_msgs.msg import Image

from pamaral_models.msg import PointList


class DatasetProcessing:

    def __init__(self, input_folder, input_topic, output_folder):
        self.output_folder = output_folder

        # Iterate over all files in the folder
        self.bag_paths = []
        self.csv_paths = []
        for filename in os.listdir(input_folder):
            # Create the absolute path to the file
            bag_path = os.path.join(input_folder, filename)

            # Check if the file path is a file (not a directory)
            if os.path.isfile(bag_path) and bag_path.endswith(".bag") and not bag_path.endswith("orig.bag"):
                self.bag_paths.append(bag_path)
                self.csv_paths.append(os.path.join(self.output_folder, filename[:-4]+".csv"))
        
        self.image_publisher = rospy.Publisher("front_camera/color/image_raw", Image, queue_size=300)
        self.preprocessed_points_sub = rospy.Subscriber(input_topic, PointList, self.preprocessed_points_callback)

    def preprocessed_points_callback(self, msg):
        #if len(msg.points)>0:
        points = [[p.x, p.y, p.z] for p in msg.points]
        points = np.array(points)

        # Open the CSV file in append mode
        with open(self.csv_paths[0], 'a+', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(points)
        
        self.num_messages -= 1
        if self.num_messages == 0:
            self.bag_paths = self.bag_paths[1:]
            self.csv_paths = self.csv_paths[1:]
            self.play_next_bag_file()
    
    def play_next_bag_file(self):
        bag = rosbag.Bag(self.bag_paths[0])

        self.num_messages = bag.get_message_count()

        for topic, msg, t in bag.read_messages():
            self.image_publisher.publish(msg)


def main():
    default_node_name = 'dataset_processing'
    rospy.init_node(default_node_name, anonymous=False)

    input_folder = rospy.get_param(rospy.search_param('input_folder'))
    input_topic = rospy.get_param(rospy.search_param('input_topic'))
    output_folder = rospy.get_param(rospy.search_param('output_folder'))
    os.makedirs(output_folder, exist_ok=True)

    dataset_processing = DatasetProcessing(input_folder=input_folder, input_topic=input_topic, output_folder=output_folder)

    input()

    dataset_processing.play_next_bag_file()

    rospy.spin()


if __name__ == '__main__':
    main()
