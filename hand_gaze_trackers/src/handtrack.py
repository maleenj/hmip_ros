#!/usr/bin/env python
# Maleen Jayasurya @ UTS-RI
import rospy
import copy
import time
import cv2 as cv 
import numpy as np
from sensor_msgs.msg import Image,CameraInfo
from trajectory_msgs.msg import JointTrajectoryPoint
from cv_bridge import CvBridge, CvBridgeError
from hand_gaze_trackers import depth_extractor as de
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
from decimal import Decimal


UPDATE_RATE = 20 # Hz

class hand_tracker:
    #constructor
    def __init__(self):
        print('construct')
        self.init_ros_components()
        self.init_media_pipe()
        self.init_variables()


    def init_ros_components(self):

        self.velocity_noise_xy = rospy.get_param('~velocity_noise_xy', 0.1)
        self.velocity_noise_z = rospy.get_param('~velocity_noise_z', 0.5)

        rospy.init_node('hand_tracker', anonymous=True, log_level=rospy.DEBUG)
        self.rawimg_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.imgsub_cb)
        self.camera_info_sub = rospy.Subscriber('/usb_cam/camera_info', CameraInfo, self.camera_info_cb)
        self.handimg_pub = rospy.Publisher('~image', Image, queue_size=10)
        self.handpose_pub = rospy.Publisher('~hand_traj', JointTrajectoryPoint, queue_size=10)

        # Create a timer that calls the 'publish_image' function every 0.1 seconds (10 Hz)
        rospy.Timer(rospy.Duration(1/UPDATE_RATE), self.tracker_update)

    def init_media_pipe(self):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0
        )

    def init_variables(self):
        self.t1, self.t2 = 0, 0
        self.detection_img = Image()
        self.hand_pose = JointTrajectoryPoint()
        self.wp1 = np.array([0, 0, 0])
        self.wp2 = np.array([0, 0, 0])
        self.v = np.array([0, 0, 0])
        self.bridge = CvBridge()
        self.check_init = 0
        self.img_msg  = '' #img msg
        self.K = [] #camera calibration

    #destructor     
    def __del__(self):
        print('Shutting down the hand_tracker object')


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
        
        # Convert ROS image message to OpenCV format.
        try:
            frame = self.bridge.imgmsg_to_cv2(self.img_msg, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        # Flip the image for visual symmetry.
        frame = cv.flip(frame, 1)

        # Convert BGR frame to RGB, which is needed for further processing.
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        results = self.hands.process(rgb_frame)

        frame_height, frame_width = rgb_frame.shape[:2]

        if results.multi_hand_landmarks:
            # Calculate the current timestamp in seconds.
            self.t1 = rospy.get_time()
            self.check_init+=1
            
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

            world_points=de.GetWorldPoints(model_points,image_points,self.K)
            
            average_loc=1*np.mean(world_points, axis=0)
            self.wp1 = np.array([-average_loc[0],average_loc[1],-average_loc[2]])
                        
            delta_t=self.t1-self.t2

            if self.check_init > 0 and delta_t > 0 and delta_t < 1:
                self.v=(self.wp1-self.wp2)/delta_t                
            else:
                self.v=[0,0,0]

            if np.abs(self.v[0]) < self.velocity_noise_xy and np.abs(self.v[1]) < self.velocity_noise_xy and np.abs(self.v[2]) < self.velocity_noise_z:
                self.v=[0,0,0]
            else:
                self.wp2=self.wp1
            
            frame = cv.circle(frame, (int(average_hand[0]),int(average_hand[1])), radius=10, color=(255, 0, 255), thickness=5)
            
            #                 ^
            #                 |Y
            #                 |
            #   hand    <----- cam
            #             Z    \
            #                   \
            #                    X   

    
            self.hand_pose.positions = self.wp1
            self.hand_pose.velocities = self.v
            self.hand_pose.time_from_start = self.img_msg.header.stamp
            self.handpose_pub.publish(self.hand_pose)

            FPS=1/(self.t1-self.t2)
            self.t2=copy.copy(self.t1)
            FPS = int(FPS)
            #print('FPS: ', FPS)

        else:
            self.t2 = rospy.get_time()
            self.hand_pose.positions = self.wp1
            self.hand_pose.velocities = [0,0,0]
            self.hand_pose.time_from_start = self.img_msg.header.stamp
            self.handpose_pub.publish(self.hand_pose)

        #Convert OpenCV image frame to ROS image and publish
        self.detection_img=self.bridge.cv2_to_imgmsg(frame, "bgr8")
        self.detection_img.header = self.img_msg.header

        if self.detection_img:
            #self.detection_img.header.stamp = rospy.Time.now()
            self.handimg_pub.publish(self.detection_img)

    

if __name__ == '__main__':
    
    relay_obj = hand_tracker()

    rospy.spin()
    print ('hand_tracker Node Exit()!')