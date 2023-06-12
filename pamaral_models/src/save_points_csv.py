#!/usr/bin/env python3

import csv
import numpy as np
import os
import rospy

from datetime import datetime

from pamaral_models.msg import PointList


class SavePointsCSV:

    def __init__(self, output_folder):
        self.file_path = output_folder + datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + ".csv"
        self.preprocessed_points_sub = rospy.Subscriber("preprocessed_points", PointList, self.preprocessed_points_callback)

    def preprocessed_points_callback(self, msg):
        points = [[p.x, p.y, p.z] for p in msg.points]
        points = np.array(points)

        # Open the CSV file in append mode
        with open(self.file_path, 'a+', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(points)


def main():
    default_node_name = 'save_points_csv'
    rospy.init_node(default_node_name, anonymous=False)

    output_folder = rospy.get_param(rospy.search_param('output_folder'))
    # os.makedirs(output_folder)

    SavePointsCSV(output_folder=output_folder)

    rospy.spin()


if __name__ == '__main__':
    main()
