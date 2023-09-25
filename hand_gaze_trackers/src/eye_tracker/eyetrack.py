#!/usr/bin/env python
# Maleen Jayasurya @ UTS-RI

import rospy
import copy
import time
import os, sys
import mediapipe as mp
from trajectory_msgs.msg import JointTrajectoryPoint
import cv2 as cv 
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from decimal import Decimal
import gaze
import pandas as pd
from pathlib import Path 

mp_face_mesh = mp.solutions.face_mesh

UPDATE_RATE = 20 # Hz


class eyeface_tracker:

    def __init__(self):
        
        print('construct')
        self.init_ros_components()
        self.init_media_pipe()
        self.init_variables()

    def init_variables(self):
        self.bridge = CvBridge()
        self.K = []
        self.eye_gaze = JointTrajectoryPoint()
        self.img_msg = []
        self.t1, self.t2 = 0, 0

    def init_ros_components(self):       
        rospy.init_node('eyeface_tracker', anonymous=True, log_level=rospy.DEBUG)	
    
        self.rawimg_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.imgsub_cb)
        self.eyeimg_pub=rospy.Publisher('~image', Image, queue_size=10) 
        self.camera_info_sub = rospy.Subscriber('/usb_cam/camera_info', CameraInfo, self.camera_info_cb)
        self.gaze_pub = rospy.Publisher('~gaze_traj', JointTrajectoryPoint, queue_size=10)

        # Create a timer that calls the 'publish_image' function every 0.1 seconds (10 Hz)
        rospy.Timer(rospy.Duration(1/UPDATE_RATE), self.tracker_update)

    def init_media_pipe(self):
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.8
        )

    def imgsub_cb(self, img_msg):

        self.img_msg = img_msg

    def camera_info_cb(self, camera_info_msg):
        # Update the K matrix
        self.K = np.array(camera_info_msg.K).reshape((3, 3))

    def tracker_update(self, event):
   
        if len(self.K) == 0 or not self.img_msg:
            #cannot make prediction until K and img is receive
            print('Waiting for image and image_info')
            time.sleep(3)
            return
        
        detection_img = Image()
        
        LEFT_IRIS = [474,475, 476, 477]
        RIGHT_IRIS = [469, 470, 471, 472]

        #Convert ROS image to OpenCV image
        try:
            frame = self.bridge.imgmsg_to_cv2(self.img_msg, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        #Extract eye data from mediapipes face landmarks
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:
            
            frame = cv.flip(frame, 1)
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            img_h, img_w = frame.shape[:2]
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                self.t1 = rospy.get_time()

                frame, x, G, trans=gaze.gaze(frame, results.multi_face_landmarks[0],self.K)

                image_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
                (l_cx, l_cy), l_radius = cv.minEnclosingCircle(image_points[LEFT_IRIS])
                (r_cx, r_cy), r_radius = cv.minEnclosingCircle(image_points[RIGHT_IRIS])
                center_left = np.array([l_cx, l_cy], dtype=np.int32)
                center_right = np.array([r_cx, r_cy], dtype=np.int32)
                cv.circle(frame, center_left, int(l_radius), (255,0,255), 1, cv.LINE_AA)
                cv.circle(frame, center_right, int(r_radius), (255,0,255), 1, cv.LINE_AA)

                if trans==1:
                    self.eye_gaze.positions = [x[0,0], x[1,0], x[2,0]]
                    self.eye_gaze.velocities = [G[0,0], G[1,0], G[2,0]]
                    self.eye_gaze.time_from_start = self.img_msg.header.stamp
                    self.gaze_pub.publish(self.eye_gaze)


                FPS=1/(self.t1-self.t2)
                self.t2=copy.copy(self.t1)
                FPS = int(FPS)
                # print('FPS: ', FPS)


            #Convert OpenCV image frame to ROS image
            #cv.waitKey(1)
            detection_img=self.bridge.cv2_to_imgmsg(frame, "bgr8")
            detection_img.header = self.img_msg.header
            self.eyeimg_pub.publish(detection_img)

if __name__ == '__main__':
    
    relay_obj = eyeface_tracker()

    rospy.spin()
    print ('eyeface_tracker Node Exit()!')


