#!/usr/bin/env python
# Maleen Jayasurya @ UTS-RI
import roslib
import rospy
import copy
import time
import os, sys
import cv2 as cv 
import numpy as np
from sensor_msgs.msg import Image,CameraInfo
from trajectory_msgs.msg import JointTrajectoryPoint
from cv_bridge import CvBridge, CvBridgeError
import depth_extractor as de
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
from decimal import Decimal

class hand_tracker:
    #constructor
    def __init__(self):
        print('construct')
        rospy.init_node('hand_tracker', anonymous=True, log_level=rospy.DEBUG)	

        self.check_init=0	
        self.t1=0
        self.t2=0
        self.detection_img = Image()
        self.hand_pose=JointTrajectoryPoint()
        self.wp_x1=0
        self.wp_y1=0
        self.wp_z1=0
        self.wp_x2=0
        self.wp_y2=0
        self.wp_z2=0
        self.vx=0
        self.vy=0
        self.vz=0
        self.velocity_noise=0.075
        self.bridge = CvBridge()
        self.K = [] #camera calibration



        mp_hands = mp.solutions.hands
        self.hands= mp_hands.Hands(
                    max_num_hands=1,
                    model_complexity=0,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

        self.rawimg_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.imgsub_cb)
        self.handimg_pub=rospy.Publisher('~image', Image, queue_size=10)
        self.handpose_pub = rospy.Publisher('~hand_traj', JointTrajectoryPoint, queue_size=10)
        self.camera_info_sub = rospy.Subscriber('/usb_cam/camera_info', CameraInfo, self.camera_info_cb)

    #destructor     
    def __del__(self):
        print('Shutting down the hand_tracker object')


    def camera_info_cb(self, camera_info_msg):
        # Update the K matrix
        self.K = np.array(camera_info_msg.K).reshape((3, 3))


    def imgsub_cb(self, img_msg):
        #print (img_msg.header.seq)

        self.t1=np.float64((img_msg.header.stamp.secs)+(Decimal(img_msg.header.stamp.nsecs)/1000000000))

        #Convert ROS image to OpenCV image
        try:
            frame = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            print(e)

        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame_height, frame_width = frame.shape[:2]

        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            self.check_init=self.check_init+1
            
            model_points = np.float32([[-l.x, -l.y, -l.z] for l in results.multi_hand_world_landmarks[0].landmark])                 
            image_points = np.float32([[l.x * frame_width, l.y * frame_height] for l in results.multi_hand_landmarks[0].landmark])
            #for hand_landmarks in results.multi_hand_landmarks:
                # mp_drawing.draw_landmarks(
                #     frame,
                #     hand_landmarks,
                #     mp_hands.HAND_CONNECTIONS,
                #     mp_drawing_styles.get_default_hand_landmarks_style(),
                #     mp_drawing_styles.get_default_hand_connections_style())

            average_hand=np.mean(image_points, axis=0)
            world_points=de.GetWorldPoints(model_points,image_points, self.K )
            
            average_loc=np.mean(world_points, axis=0)
            self.wp_x1=-average_loc[0]
            self.wp_y1=average_loc[1]
            self.wp_z1=-average_loc[2]
            
            delta_t=self.t1-self.t2

            if self.check_init > 1 and delta_t > 0 and delta_t < 1:
                self.v_x=(self.wp_x1-self.wp_x2)/delta_t
                self.v_y=(self.wp_y1-self.wp_y2)/delta_t
                self.v_z=(self.wp_z1-self.wp_z2)/delta_t
            else:
                self.v_x=0
                self.v_y=0
                self.v_z=0

            if np.abs(self.v_x) < self.velocity_noise and np.abs(self.v_y) and np.abs(self.v_z) < self.velocity_noise:
                self.v_x=0
                self.v_y=0
                self.v_z=0
            else:
                self.wp_x2=self.wp_x1
                self.wp_y2=self.wp_y1
                self.wp_z2=self.wp_z1
            
            frame = cv.circle(frame, (int(average_hand[0]),int(average_hand[1])), radius=10, color=(255, 0, 255), thickness=5)
            
            #Publish hand pose
            self.hand_pose.positions = [self.wp_x1,self.wp_y1,self.wp_z1]
            self.hand_pose.velocities = [self.v_x,self.v_y,self.v_z]
            self.hand_pose.time_from_start = img_msg.header.stamp
            self.handpose_pub.publish(self.hand_pose)


        #Convert OpenCV image frame to ROS image and publish
        self.detection_img=self.bridge.cv2_to_imgmsg(frame, "bgr8")
        self.detection_img.header = img_msg.header
        self.handimg_pub.publish(self.detection_img)
            
        FPS=1/(self.t1-self.t2)
        self.t2=self.t1
        FPS = int(FPS)
        #print('FPS: ', FPS)


if __name__ == '__main__':
    
    relay_obj = hand_tracker()

    rospy.spin()
    print ('hand_tracker Node Exit()!')