ros_robot_controller:
	roslaunch larcc_launches script_control_ur10e.launch

arm_movement:
	rosrun pamaral_decision_making_block arm_gripper_movement.py

echo_joints:
	rostopic echo \joint_states
