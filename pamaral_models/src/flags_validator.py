#!/usr/bin/env python3

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # set to empty string to force CPU usage

import rospy
import tensorflow as tf
from pamaral_models.srv import FlagsValidator, FlagsValidatorResponse
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras


import keras.backend as K

def weighted_binary_crossentropy(y_true, y_pred):
    class_weights = [0.54065041, 6.65 ]
    y_true = tf.cast(y_true, dtype=tf.float32)
    # Flatten the predictions and true labels to compute the cross-entropy loss
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    # Compute the binary cross-entropy loss
    bce = K.binary_crossentropy(y_true_f, y_pred_f)
    # Multiply the loss by the class weights
    weight_vector = y_true_f * class_weights[1] + (1. - y_true_f) * class_weights[0]
    weighted_bce = weight_vector * bce
    return K.mean(weighted_bce)

class MLNode:
    def __init__(self,tokenizer_path,model_path):
        # load tokenizer from file
        f = open(tokenizer_path, "r")
        self.tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(f.read())
        f.close()
        keras.utils.get_custom_objects()['weighted_binary_crossentropy'] = weighted_binary_crossentropy
        self.model = tf.keras.models.load_model(model_path)
        rospy.Service('flags_validator', FlagsValidator, self.flags_validator)

    def flags_validator(self, req):
        # preprocess input data
        if len(req.color1)==0:
            return None
        
        input_data = req.color1
        if len(req.color2)>0:
            input_data += "," + req.color2

            if len(req.color3)>0:
                input_data += "," + req.color3
        
        print(input_data)
        
        input_data = self.tokenizer.texts_to_sequences([input_data])

        input_data = np.array(pad_sequences(input_data, maxlen = 3))
        
        # make prediction using loaded model
        prediction = self.model.predict(input_data)

        print(prediction)
        
        return FlagsValidatorResponse(valid = prediction[0][0] > 0.5)


if __name__ == '__main__':
    rospy.init_node('flags_validator')
    tokenizer_path = rospy.get_param(rospy.search_param('tokenizer_path'))
    model_path = rospy.get_param(rospy.search_param('model_path'))
    MLNode(tokenizer_path,model_path)
    rospy.spin()
