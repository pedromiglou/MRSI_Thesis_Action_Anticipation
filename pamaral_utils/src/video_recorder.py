#!/usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_srvs.srv import Empty, EmptyResponse

class ImageToVideoNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.video_writer = None
        self.is_recording = False

        # Define ROS service to start and stop recording
        self.start_recording_service = rospy.Service('start_recording', Empty, self.start_recording)
        self.stop_recording_service = rospy.Service('stop_recording', Empty, self.stop_recording)

    def image_callback(self, msg):
        try:
            # Convert ROS image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            rospy.logerr("Failed to convert image: %s" % str(e))
            return

        if self.is_recording:
            # Write image to video file
            if self.video_writer is None:
                rospy.logwarn("Video writer object not initialized")
                return
            self.video_writer.write(cv_image)

    def start_recording(self, req):
        if not self.is_recording:
            # Create video writer object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter('output.mp4', fourcc, 30.0, (640, 480))

            # Set flag to start recording
            self.is_recording = True

        return EmptyResponse()

    def stop_recording(self, req):
        if self.is_recording:
            # Release video writer object
            if self.video_writer is None:
                rospy.logwarn("Video writer object not initialized")
                return
            self.video_writer.release()

            # Set flag to stop recording
            self.is_recording = False

        return EmptyResponse()

if __name__ == '__main__':
    rospy.init_node('image_to_video_node')
    image_to_video = ImageToVideoNode()

    rospy.spin()
