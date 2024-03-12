#!/usr/bin/env python3

import json
import rospy

from experta import Fact, KnowledgeEngine, Rule


from base_controller import BaseController
from pamaral_object_grasping_pattern_recognition.msg import ObjectPrediction


default_node_name = 'rule_based_controller'
rospy.init_node(default_node_name, anonymous=False)
rules_path = rospy.get_param(rospy.search_param('rules_path'))


class Grasping(Fact):
    """Info about the object being grasped."""
    pass


class RuleBasedController(KnowledgeEngine, BaseController):

    f = open(rules_path, "r")
    rules = json.load(f)
    f.close()

    for i, rule in enumerate(rules):
        grasping = rule["grasping"]
        next_block = rule["next_block"]
        f_str = f"@Rule(Grasping(v='{grasping}'))\n"+f"def rule{i}(self):\n\t"+f"self.current_block='{next_block}'"
        exec(f_str)
    
    def __init__(self, position_list, rules_path):
        KnowledgeEngine.__init__(self)
        self.reset()

        BaseController.__init__(self,position_list)

        self.last_prediction = None
        self.object_class_sub = rospy.Subscriber("object_class", ObjectPrediction, self.object_class_callback)

        # flag so that the controller does not keep making Experta run
        self.engine_ran = False
    
    def object_class_callback(self, msg):
        if msg.object_class.data == self.last_prediction:
            print(msg.object_class.data)
            self.declare(Grasping(v=self.last_prediction))
        
        self.last_prediction = msg.object_class.data
    
    def idle_state(self):
        self.run()
        rospy.loginfo("Engine ran, current_block: "+str(self.current_block))

        if self.state == "idle" and self.current_block is not None:
            self.state = "picking_up"

    # def idle_state(self):
    #     if self.current_block is None and not self.engine_ran:
    #         self.run()
    #         self.engine_ran = True
    #         rospy.loginfo("Engine ran, current_block: "+str(self.current_block))

    #         if self.state == "idle" and self.current_block is not None:
    #             self.state = "picking_up"
    
    def putting_down_state(self):
        BaseController.putting_down_state(self)

        self.reset()

        self.engine_ran = False
    
    # def stop_wrong_guess_state(self):
    #     self.engine_ran = False

    #     rospy.loginfo("Refused "+self.current_block)

    #     BaseController.stop_wrong_guess_state(self)


def main():
    quaternion_poses = rospy.get_param(rospy.search_param('quaternion_poses'))

    rule_based_controller = RuleBasedController(position_list = quaternion_poses, rules_path=rules_path)

    rule_based_controller.loop()

    rospy.spin()


if __name__ == "__main__":
    
    main()
