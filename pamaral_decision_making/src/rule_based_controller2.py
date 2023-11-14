#!/usr/bin/env python3

import json
import rospy

from experta import Fact, KnowledgeEngine, Rule

from base_controller import BaseController
from pamaral_perception_system.msg import CentroidList


default_node_name = 'rule_based_controller'
rospy.init_node(default_node_name, anonymous=False)
rules_path = rospy.get_param(rospy.search_param('rules_path'))


class Blocks(Fact):
    """Info about the blocks in the workspace."""
    pass


class Refused(Fact):
    """Info about the blocks the user refused."""
    pass


class RuleBasedController(KnowledgeEngine, BaseController):

    f = open(rules_path, "r")
    rules = json.load(f)
    f.close()

    for i, rule in enumerate(rules):
        blocks = str(tuple(f"{b}"for b in rule["blocks"]))
        refused = "'_'" if len(rule["refused"]) == 0 else "'"+rule["refused"]+"'"
        next_block = rule["next_block"]
        f_str = f"@Rule(Blocks(v={blocks}) & Refused(v={refused}))\n"+f"def rule{i}(self):\n\t"+f"self.current_block='{next_block}'"
        exec(f_str)
    
    def __init__(self, position_list, rules_path):
        KnowledgeEngine.__init__(self)
        self.reset()

        BaseController.__init__(self,position_list)

        # flag so that the controller does not keep making Experta run
        self.engine_ran = False

        self.hand_centroids_subscriber = rospy.Subscriber("hand_centroids", CentroidList, self.hand_centroids_callback)
    
    def hand_centroids_callback(self,msg):
        centroids = msg.points

        if len(self.blocks) % 3 == 0 and self.current_block is None:
            if len(centroids)>0:
                rospy.loginfo("Received centroid -> "+msg.color)
                self.blocks = [msg.color]

                self.reset()
                self.declare(Blocks(v=tuple(self.blocks)))
                self.declare(Refused(v="_"))
                self.run()
                self.engine_ran = True

                if self.state == "idle" and self.current_block is not None:
                    self.state = "picking_up"
    
    def idle_state(self):
        if self.current_block is not None and not self.engine_ran:
            self.run()
            self.engine_ran = True
            rospy.loginfo("Engine ran, current_block: "+self.current_block)

            if self.state == "idle" and self.current_block is not None:
                self.state = "picking_up"
    
    def putting_down_state(self):
        BaseController.putting_down_state(self)

        self.declare(Blocks(v=tuple(self.blocks)))
        self.declare(Refused(v="_"))
        self.engine_ran = False
    
    def stop_wrong_guess_state(self):
        self.declare(Refused(v=self.current_block))
        self.engine_ran = False

        rospy.loginfo("Refused "+self.current_block)

        BaseController.stop_wrong_guess_state(self)


def main():
    quaternion_poses = rospy.get_param(rospy.search_param('quaternion_poses'))

    rule_based_controller = RuleBasedController(position_list = quaternion_poses, rules_path=rules_path)

    rule_based_controller.loop()

    rospy.spin()

    rule_based_controller.arm_gripper_comm.gripper_disconnect()


if __name__ == "__main__":
    
    main()




