#!/usr/bin/env python3

import csv
import numpy as np
import os
import rosbag
import rospy

from sensor_msgs.msg import Image

from pamaral_models.msg import PointList


class DatasetProcessing:

    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.file_paths = []
        self.image_publisher = rospy.Publisher("front_camera/color/image_raw", Image, queue_size=21000)
        self.preprocessed_points_sub = rospy.Subscriber("preprocessed_points", PointList, self.preprocessed_points_callback)

    def preprocessed_points_callback(self, msg):
        points = [[p.x, p.y, p.z] for p in msg.points]
        points = np.array(points)

        # Open the CSV file in append mode
        assert len(self.file_paths) > 0
        file_path = self.file_paths[0][0]

        if self.file_paths[0][1] > 1:
            self.file_paths[0][1] -= 1
        else:
            self.file_paths = self.file_paths[1:]

        with open(file_path, 'a+', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(points)
    
    def play_bag_files(self):
        # Iterate over all files in the folder
        for filename in os.listdir(self.input_folder):
            # Create the absolute path to the file
            bag_path = os.path.join(self.input_folder, filename)

            # Check if the file path is a file (not a directory)
            if os.path.isfile(bag_path) and bag_path.endswith(".bag"):
                # Open the bag file
                bag = rosbag.Bag(bag_path)

                self.file_paths.append([os.path.join(self.output_folder, filename[:-4]+".csv"), bag.get_message_count()])

                for topic, msg, t in bag.read_messages():
                    self.image_publisher.publish(msg)


def main():
    default_node_name = 'dataset_processing'
    rospy.init_node(default_node_name, anonymous=False)

    input_folder = rospy.get_param(rospy.search_param('input_folder'))
    output_folder = rospy.get_param(rospy.search_param('output_folder'))
    os.makedirs(output_folder, exist_ok=True)

    dataset_processing = DatasetProcessing(input_folder=input_folder, output_folder=output_folder)

    dataset_processing.play_bag_files()

    rospy.spin()


if __name__ == '__main__':
    main()
