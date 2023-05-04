#!/usr/bin/env python3

import rospy
import tensorflow as tf
from pamaral_models.srv import FlagsValidator, FlagsValidatorResponse
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


class MLNode:
    def __init__(self,tokenizer_path,model_path):
        # load tokenizer from file
        f = open(tokenizer_path, "r")
        self.tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(f.read())
        f.close()
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
