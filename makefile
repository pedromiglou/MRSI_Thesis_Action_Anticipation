ros_robot_controller:
	roslaunch larcc_launches script_control_ur10e.launch

arm_movement:
	rosrun pamaral_decision_making_block arm_gripper_movement.py

echo_joints:
	rostopic echo \joint_states

create_postgres:
	docker run -d --name postgres -e POSTGRES_PASSWORD=password -e PGDATA=/var/lib/postgresql/data/pgdata -v /home/miglou/postgres_data:/var/lib/postgresql/data -p 5432:5432 postgres

take_photos:
	roslaunch pamaral_perception_block usb_cam-test.launch
