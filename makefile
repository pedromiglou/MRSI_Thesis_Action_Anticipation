# Launch a Postgres container for probabilities controller
create_postgres:
	docker run -d --name postgres -e POSTGRES_PASSWORD=password -e PGDATA=/var/lib/postgresql/data/pgdata -v /home/miglou/postgres_data:/var/lib/postgresql/data -p 5432:5432 postgres

# Start ROS larcc_drivers
ros_robot_controller:
	roslaunch larcc_bringup script_control_ur10e.launch

# Start node to control the robot with joints
ros_arm_movement_joints:
	roslaunch pamaral_decision_making arm_gripper_movement_joints.launch

# Start node to control the robot with quaternions
ros_arm_movement_quaternions:
	roslaunch pamaral_decision_making arm_gripper_movement_quaternions.launch

###### Utils ######
ros_echo_joints:
	rostopic echo \joint_states

ros_take_photos:
	roslaunch pamaral_perception_block usb_cam.launch
