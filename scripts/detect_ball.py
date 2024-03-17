###sidney Tsui 
### 3/18/2024
### lab 4

###subsribes to correct topic and receives  image frames from the recorded bag file (same topic as real camera)

###proceses image via techniques using HSV color space 
###produces mono-color image (figure ground reversal)
	#labels pixels corresponding to ball with value equal to 255 and other pixels with 0
	
###publishes processed mono_channel imahe
	###set control loop frequency to 10Hz
	###use name /ball_2D for topic
	###use mono8 for output image encoding in ROS messages
	
###test by playing rosbag file and rqt_gui

#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

img_received = False
# define a 720x1280 3-channel image with all pixels equal to zero
global rgb_img = np.zeros((720, 1280, 3), dtype = "uint8")


# get the image message
def get_image(ros_img):
	#global rgb_img
	global img_received
	# convert to opencv image
	rgb_img = CvBridge().imgmsg_to_cv2(ros_img, "rgb8")
	# raise flag
	img_received = True
	
def process_image(process_img):
	hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	lower_yellow_hsv = np.array([25, 100, 1])
	upper_yellow_hsv = np.array([60, 255, 255])
	yellow_mask = cv2.inRange(hsv, lower_yellow_hsv, upper_yellow_hsv)
	
	#rm_background = np.zeros(
	#print('This image is:', type(yellow_mask), 'with dimension:', yellow_mask.shape)

if __name__ == '__main__':
	# define the node and subcribers and publishers
	rospy.init_node('ball_2D', anonymous = True)
	# define a subscriber to ream images
	img_sub = rospy.Subscriber("/camera/color/image_raw", Image, get_image) 
	# define a publisher to publish images
	img_pub = rospy.Publisher('/ball_2D', Image, queue_size = 1)
	
	# set the loop frequency
	rate = rospy.Rate(10)

	while not rospy.is_shutdown():
		# make sure we process if the camera has started streaming images
		if img_received:
			# flip the image up			
			mono-color-image = get_image(rgb_image)
			# convert it to ros msg and publish it
			img_msg = CvBridge().cv2_to_imgmsg(mono-color-image, encoding="mono8")
			# publish the image
			img_pub.publish(img_msg)
		# pause until the next iteration			
		rate.sleep()



