#!/usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_srvs.srv import Empty, EmptyResponse
from datetime import datetime


class VideoRecorder:
    def __init__(self, input_topic, filename, fps):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(input_topic, Image, self.image_callback)
        self.filename = filename
        self.fps = fps
        self.video_writer = None
        self.is_recording = False
        self.images = []

        # Define ROS service to start and stop recording
        self.start_recording_service = rospy.Service('start_recording', Empty, self.start_recording)
        self.stop_recording_service = rospy.Service('stop_recording', Empty, self.stop_recording)

    def image_callback(self, msg):
        try:
            # Convert ROS image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(e)
            return

        if self.is_recording:
            # Write image to video file
            if self.video_writer is None:
                rospy.logwarn("Video writer object not initialized")
                return
            
            self.images.append(cv_image)
            #self.video_writer.write(cv_image)
            #print("written image")

    def start_recording(self, req):
        if not self.is_recording:
            # Create video writer object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.filename, fourcc, self.fps, (640, 480))

            # Set flag to start recording
            self.is_recording = True

            print("Recording started.")

        return EmptyResponse()

    def stop_recording(self, req):
        if self.is_recording:
            # Release video writer object
            if self.video_writer is None:
                rospy.logwarn("Video writer object not initialized")
                return
            
            for img in self.images:
                self.video_writer.write(img)
            
            print(self.video_writer.release())

            # Set flag to stop recording
            self.is_recording = False

        return EmptyResponse()


def main():
    # ---------------------------------------------------
    # INITIALIZATION
    # ---------------------------------------------------
    default_node_name = 'video_recorder'
    rospy.init_node(default_node_name, anonymous=False)

    input_topic = rospy.get_param(rospy.search_param('input_image_topic'))
    output_folder = rospy.get_param(rospy.search_param('output_video_folder'))
    fps = rospy.get_param(rospy.search_param('fps'))

    # Create a new folder
    now = datetime.now()
    filename = output_folder + now.strftime("%d_%m_%Y_%H:%M:%S")+ ".mp4"
    print(filename)

    VideoRecorder(input_topic=input_topic, filename=filename, fps=float(fps))

    rospy.spin()


if __name__ == '__main__':
    main()
