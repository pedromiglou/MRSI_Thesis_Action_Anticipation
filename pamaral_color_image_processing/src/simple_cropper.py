#!/usr/bin/env python3

import rospy

from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class SimpleCropper:

    def __init__(self, input_topic, output_topic):
        self.bridge = CvBridge()
        self.publisher = rospy.Publisher(output_topic, Image, queue_size=1)
        self.subscriber = rospy.Subscriber(input_topic, Image, self.image_callback)

    def image_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        except:
            print("Error reading color image")
            return
        
        # crop to ROI
        img = img[34:449, 237:457]

        # publish cropped image
        self.publisher.publish(self.bridge.cv2_to_imgmsg(img, "bgr8"))


def main():
    # ---------------------------------------------------
    # INITIALIZATION
    # ---------------------------------------------------
    default_node_name = 'simple_cropper'
    rospy.init_node(default_node_name, anonymous=False)

    input_topic = rospy.get_param(rospy.search_param('input_image_topic'))
    output_topic = rospy.get_param(rospy.search_param('output_image_topic'))

    SimpleCropper(input_topic, output_topic)

    rospy.spin()


if __name__ == '__main__':
    main()
