#!/usr/bin/env python3

import json
import rospy

from experta import Fact, KnowledgeEngine, Rule

from base_controller import BaseController

class Blocks(Fact):
    """Info about the blocks in the workspace."""
    pass


class Refused(Fact):
    """Info about the blocks the user refused."""
    pass


class RuleBasedController(KnowledgeEngine, BaseController):
    
    def __init__(self, position_list, rules_path):
        KnowledgeEngine.__init__(self)
        self.reset()

        f = open(rules_path, "r")
        rules = json.load(f)
        f.close()

        for i, rule in enumerate(rules):
            blocks = str(tuple(rule["blocks"]))
            refused = str(tuple(rule["refused"]))
            next_block = rule["next_block"]
            exec(   f"@Rule(Blocks(v={blocks}) & Refused(v={refused}))\n"+
                    f"def rule{i}(self):\n\t"+
                        f"self.current_block=[{next_block}]")
        
        BaseController.__init__(self,position_list)
    
    def idle_state(self):
        if self.current_block is not None:
            self.run()

            if self.state == "idle":
                if len(self.current_block)>0:
                    self.state = "picking_up"
                else:
                    self.current_block = None
    
    def putting_down_state(self):
        BaseController.putting_down_state(self)

        self.declare(Blocks(v=tuple(self.blocks)))
    
    def stop_wrong_guess_state(self):
        self.declare(Refused(v=tuple(self.current_block[0])))

        BaseController.stop_wrong_guess_state(self)


def main():
    # ---------------------------------------------------
    # INITIALIZATION
    # ---------------------------------------------------
    default_node_name = 'rule_based_controller'
    rospy.init_node(default_node_name, anonymous=False)

    quaternion_poses = rospy.get_param(rospy.search_param('quaternion_poses'))
    rules_path = rospy.get_param(rospy.search_param('rules_path'))

    rule_based_controller = RuleBasedController(position_list = quaternion_poses, rules_path=rules_path)

    rule_based_controller.loop()

    rospy.spin()

    rule_based_controller.arm_gripper_comm.gripper_disconnect()


if __name__ == "__main__":
    main()
