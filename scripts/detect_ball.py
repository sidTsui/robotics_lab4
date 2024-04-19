#!/usr/bin/env python3
###sidney Tsui 
### 3/18/2024
### lab 4

##sources: in class lectures, provided repository resources, online sources.

###proceses image via techniques using HSV color space 
###produces mono-color image (figure ground reversal)
	#labels pixels corresponding to ball with value equal to 255 and other pixels with 0
	
###publishes processed mono_channel imahe
	###set control loop frequency to 10Hz
	###use name /ball_2D for topic
	###use mono8 for output image encoding in ROS messages
	
###test by playing rosbag file and rqt_gui


import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

img_received = False
# define a 720x1280 3-channel image with all pixels equal to zero
rgb_img = np.zeros((720, 1280, 3), dtype = "uint8")


# get the image message
def get_image(ros_img):
	global rgb_img
	global img_received
	# convert to opencv image
	rgb_img = CvBridge().imgmsg_to_cv2(ros_img, "rgb8")
	# raise flag
	img_received = True
	
def process_image(process_img):
	#convert to HSV 
	hsv = cv2.cvtColor(process_img, cv2.COLOR_RGB2HSV)
	#setting high and low values for HSV 
	lower_yellow_hsv = np.array([20, 1, 1])
	upper_yellow_hsv = np.array([50, 255, 255])
	#allows for only tennis ball to be shohwn
	yellow_mask = cv2.inRange(hsv, lower_yellow_hsv, upper_yellow_hsv)
	#remove other objects detected in background
	rm_background  = np.zeros_like(yellow_mask)
	rect_funct = cv2.rectangle(rm_background, (500, 100), (800, 600), 255, -1) #draws rectangle in black accross middle of image, then inverts
	return cv2.bitwise_and(yellow_mask, rect_funct)
	

if __name__ == '__main__':
	###subsribes to correct topic and receives  image frames from the recorded bag file (same topic as real camera
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
			# create color reverasal image
			mono_color_image = process_image(rgb_img)
			# convert it to ros msg and publish it
			img_msg = CvBridge().cv2_to_imgmsg(mono_color_image, encoding="mono8")
			# publish the image
			img_pub.publish(img_msg)
		# pause until the next iteration			
		rate.sleep()



