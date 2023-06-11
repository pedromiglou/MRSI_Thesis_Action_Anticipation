import rospy
import time

from std_srvs.srv import Empty

# Initialize the ROS node
rospy.init_node('service_client_node')

# Wait for the service to become available
rospy.wait_for_service('/front_camera/start_capture')
rospy.wait_for_service('/front_camera/stop_capture')

for i in range(3,0,-1):
    print(f"Starting in {i}")

    time.sleep(1)

service_proxy = rospy.ServiceProxy('/front_camera/start_capture', Empty)

service_proxy()

for i in range(10,0,-1):
    print(f"Stopping in {i}")

    time.sleep(1)

service_proxy = rospy.ServiceProxy('/front_camera/stop_capture', Empty)

service_proxy()