#!/usr/bin/env python3

import json
import rospy

from base_controller import BaseController
from pamaral_object_grasping_pattern_recognition.msg import ObjectPrediction


class RuleBasedController(BaseController):
    def __init__(self, position_list, rules_path):
        super().__init__(position_list)

        f = open(rules_path, "r")
        self.rules = [[{"grasping": rule["grasping"]}, rule["next_block"]] for rule in json.load(f)]
        f.close()

        self.state_dict["grasping"] = ""

        self.last_prediction = None
        self.object_class_sub = rospy.Subscriber("object_class", ObjectPrediction, self.object_class_callback)
    
    def check_rules(self):
        for rule in self.rules:
            if rule[0].items <= self.state_dict.items():
                self.current_block = rule[1]
                return
        
        self.current_block = None
    
    def object_class_callback(self, msg):
        if msg.object_class.data == self.last_prediction:
            print(msg.object_class.data)
            self.state_dict["grasping"] = self.last_prediction
        
        self.last_prediction = msg.object_class.data
    
    def idle_state(self):
        if self.current_block is None:
            self.check_rules()
            rospy.loginfo("Rules checked, current_block: "+str(self.current_block))

            if self.state == "idle" and self.current_block is not None:
                self.state = "picking_up"


def main():
    default_node_name = 'rule_based_controller'
    rospy.init_node(default_node_name, anonymous=False)

    rules_path = rospy.get_param(rospy.search_param('rules_path'))
    quaternion_poses = rospy.get_param(rospy.search_param('quaternion_poses'))

    rule_based_controller = RuleBasedController(position_list = quaternion_poses, rules_path=rules_path)

    rule_based_controller.loop()

    rospy.spin()


if __name__ == "__main__":
    
    main()
