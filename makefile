ros_base:
	roslaunch larcc_launches script_control_ur10e.launch

arm_movement:
	rosrun robot_movement arm_gripper_movement.py

echo_joints:
	rostopic echo \joint_states
