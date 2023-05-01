ros_robot_controller:
	roslaunch larcc_launches script_control_ur10e.launch

arm_movement:
	roslaunch pamaral_decision_making_block arm_gripper_movement.launch

arm_movement_quaternions:
	roslaunch pamaral_decision_making_block arm_gripper_movement_quaternions.launch

echo_joints:
	rostopic echo \joint_states

take_photos:
	roslaunch pamaral_perception_block usb_cam.launch

create_postgres:
	docker run -d --name postgres -e POSTGRES_PASSWORD=password -e PGDATA=/var/lib/postgresql/data/pgdata -v /home/miglou/postgres_data:/var/lib/postgresql/data -p 5432:5432 postgres
