#!/usr/bin/env python
# Maleen Jayasurya @ UTS-RI

import rospy
import copy
import time
import os, sys
import mediapipe as mp
import cv2 as cv 
import numpy as np
from sensor_msgs.msg import Image,CameraInfo
from std_srvs.srv import Empty
from cv_bridge import CvBridge, CvBridgeError
from decimal import Decimal
import gaze
import pandas as pd
from scipy.optimize import minimize
from pathlib import Path 

filepath1 = Path('~/g.csv') 
filepath2 = Path('~/x.csv')  


UPDATE_RATE = 25 # Hz
mp_face_mesh = mp.solutions.face_mesh

class eyeface_tracker:

    def __init__(self):
        
        rospy.init_node('eyeface_tracker', anonymous=True, log_level=rospy.DEBUG)		
        self.t1=0
        self.t2=0
        self.G=np.empty((0,3))
        self.X_p=np.empty((0,3))
        self.bridge = CvBridge()
        self.scoreA=0
        self.scoreB=0
        self.scoreC=0
        self.scoreD=0
        self.scoreE=0
        self.start_stop = False

        self.rawimg_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.imgsub_cb)
        self.eyeimg_pub=rospy.Publisher('~image', Image, queue_size=10) 
        self.camera_info_sub = rospy.Subscriber('/usb_cam/camera_info', CameraInfo, self.camera_info_cb)
        self.start_stop_srv = rospy.Service('~start_stop', Empty, self.start_stop_cb)
        self.calibration_srv = rospy.Service('~calibrate', Empty, self.calibrate_cb)
        

        self.K = [] #camera calibration
        self.img_msg=[]

        rospy.Timer(rospy.Duration(1/UPDATE_RATE), self.calib_update)

        

    def __del__(self):
        print('Shutting down the eyeface_tracker object')

    def gazehit(self,G, X_p, X_e):

        gazevec=X_e - X_p
        #gazevec=np.array([-66.36185776,  22.36018575,  95.81892301])

        dotscore=np.dot(G, gazevec)

        denom=np.linalg.norm(G)*np.linalg.norm(gazevec)
        score=(dotscore/denom)

        #print('score: ', score)

        # if score > 0.995:
        #     print('gaze hit')
        # else:
        #     print('gaze miss')

        return score

    def camera_info_cb(self, camera_info_msg):
        # Update the K matrix
        self.K = np.array(camera_info_msg.K).reshape((3, 3))


    def start_stop_cb(self, req):

        if self.start_stop:
            self.start_stop = False
            req = Empty()
            self.calibrate_cb(req)
            print('Calibration stopped')
        else:
            print('Calibrating')
            self.start_stop = True

        return []

    def calibrate_cb(self, req):

        df1 = pd.read_csv(filepath1)
        df2 = pd.read_csv(filepath2)
        G = (df1.to_numpy()[:, 1:4])
        x_p = (df2.to_numpy()[:, 1:4])

        # Initial guess for x_e
        initial_guess = np.array([0, 0, 0])

        # Find x_e using SciPy's minimize function
        result = minimize(self.cost_function, initial_guess, args=(x_p, G))
        x_e_optimal = result.x

        print("x_e_optimal: " + str(x_e_optimal))

        
    def calib_update(self, event):

        if len(self.K) == 0 or not self.img_msg:
                #cannot make prediction until K and img is receive
                print('Waiting for image and image_info')
                time.sleep(3)
                return
        
        detection_img = self.img_msg

        LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
        RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ] 
        LEFT_IRIS = [474,475, 476, 477]
        RIGHT_IRIS = [469, 470, 471, 472]

        #Convert ROS image to OpenCV image
        try:
            frame = self.bridge.imgmsg_to_cv2(self.img_msg, "bgr8")
        except CvBridgeError as e:
            print(e)

        #Extract eye data from mediapipes face landmarks
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.8
        ) as face_mesh:
            
            frame = cv.flip(frame, 1)
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            img_h, img_w = frame.shape[:2]
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                frame, x, s, trans=gaze.gaze(frame, results.multi_face_landmarks[0],self.K)

                image_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
                #model_points = np.float32([[-l.x, -l.y, -l.z] for l in results.multi_face_world_landmarks[0].landmark])
                #print(mesh_points[LEFT_IRIS])
                # cv.polylines(frame, [mesh_points[LEFT_IRIS]], True, (0,255,0), 1, cv.LINE_AA)
                # cv.polylines(frame, [mesh_points[RIGHT_IRIS]], True, (0,255,0), 1, cv.LINE_AA)
                (l_cx, l_cy), l_radius = cv.minEnclosingCircle(image_points[LEFT_IRIS])
                (r_cx, r_cy), r_radius = cv.minEnclosingCircle(image_points[RIGHT_IRIS])
                center_left = np.array([l_cx, l_cy], dtype=np.int32)
                center_right = np.array([r_cx, r_cy], dtype=np.int32)
                cv.circle(frame, center_left, int(l_radius), (255,0,255), 1, cv.LINE_AA)
                cv.circle(frame, center_right, int(r_radius), (255,0,255), 1, cv.LINE_AA)

                if self.start_stop :
                    if trans==1:
                        self.G= np.vstack((self.G,[s[0,0], s[1,0], s[2,0]]))
                        df1 = pd.DataFrame(self.G)
                        df1.to_csv(filepath1)
                        self.X_p= np.vstack((self.X_p,[x[0,0], x[1,0], x[2,0]]))
                        df2 = pd.DataFrame(self.X_p)
                        df2.to_csv(filepath2)
          
        #Convert OpenCV image frame to ROS image
        #cv.waitKey(1)
        detection_img=self.bridge.cv2_to_imgmsg(frame, "bgr8")
        detection_img.header = self.img_msg.header
        self.eyeimg_pub.publish(detection_img)

    def imgsub_cb(self, img_msg):
        self.img_msg = img_msg
        


    def cost_function(self, x_e, x_p, G):
        gazevec=x_e-x_p
        dotscore=np.sum(np.multiply(G, gazevec))
        denom=np.linalg.norm(G)*np.linalg.norm(gazevec)
        diff = (dotscore/denom) - 1
        return np.sum(np.linalg.norm(diff)**2)




if __name__ == '__main__':
    
    relay_obj = eyeface_tracker()

    rospy.spin()
    print ('eyeface_tracker Node Exit()!')


