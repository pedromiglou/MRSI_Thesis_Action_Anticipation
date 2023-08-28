#!/usr/bin/env python3

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # set to empty string to force CPU usage

import numpy as np
import rospy
import tensorflow as tf

from pamaral_models.msg import PointList


class ObjectClassifier:
    def __init__(self, model_path):
        # load keras model
        self.model = tf.keras.models.load_model(model_path)

        self.preprocessed_points_sub = rospy.Subscriber("preprocessed_points", PointList, self.preprocessed_points_callback)
        

    def preprocessed_points_callback(self, msg):
        if len(msg.points)>0:
            points = [[p.x, p.y, p.z] for p in msg.points]
            points = np.array(points)

            print(points.shape)

            # make prediction using loaded model
            prediction = self.model.predict(tf.expand_dims(points, axis=0))

            print(prediction)


def main():
    default_node_name = 'flags_validator'
    rospy.init_node(default_node_name)

    model_path = rospy.get_param(rospy.search_param('model_path'))

    ObjectClassifier(model_path)

    rospy.spin()
    

if __name__ == '__main__':
    main()
