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
from hand_gaze_trackers import depth_extractor_old as de
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
from prediction_msgs.msg import Prediction


model_path = '/home/maleen/models/gesture_recognizer.task'

UPDATE_RATE = 20 # Hz

class gesture_tracker:
    #constructor
    def __init__(self):
        print('construct')
        self.init_ros_components()
        self.init_variables()
        self.init_media_pipe()

    def init_ros_components(self):

        rospy.init_node('gesture_tracker', anonymous=True, log_level=rospy.DEBUG)

        self.mindistance = rospy.get_param('~mindistance', 0.05)
        self.object_data = rospy.get_param('~object_locations', [])

        self.rawimg_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.imgsub_cb)
        self.camera_info_sub = rospy.Subscriber('/usb_cam/camera_info', CameraInfo, self.camera_info_cb)
        self.handimg_pub = rospy.Publisher('~image', Image, queue_size=10)
        self.prediction_pub = rospy.Publisher('~prediction', Prediction, queue_size=10)
        #self.handpose_pub = rospy.Publisher('~hand_traj', JointTrajectoryPoint, queue_size=10)

        # Create a timer that calls the 'publish_image' function every 0.1 seconds (10 Hz)
        rospy.Timer(rospy.Duration(1/UPDATE_RATE), self.tracker_update)

    def init_media_pipe(self):
        # Create an GestureRecognizer object.
        BaseOptions = mp.tasks.BaseOptions
        GestureRecognizer = mp.tasks.vision.GestureRecognizer
        GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Create a gesture recognizer instance with the video mode:
        options = GestureRecognizerOptions(
            num_hands=1,
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)


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

    def object_distances(self, locations, wp):

        distance=np.linalg.norm(locations-wp)

        return distance


    def imgsub_cb(self, img_msg):

        self.img_msg = img_msg

    def camera_info_cb(self, camera_info_msg):
        # Update the K matrix
        self.K = np.array(camera_info_msg.K).reshape((3, 3))

    def tracker_update(self, event):

        score_list = []
  
        
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

        # Vertical flip for pointing up gestures
       
        frame= cv.flip(frame, 0)

        # Convert BGR frame to RGB, which is needed for further processing.
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        results = self.recognizer.recognize(mp_image)

        frame_height, frame_width = rgb_frame.shape[:2]

        if results.gestures:
          
            if results.handedness[0][0].category_name == 'Right': # and results.gestures[0][0].category_name == 'Pointing_Up':
                
                hand_landmarks = results.hand_landmarks
                world_landmarks = results.hand_world_landmarks
           
                model_points = np.float32([[-l.x, -l.y, -l.z] for l in world_landmarks[0]])  
                image_points = np.float32([[l.x * frame_width, l.y * frame_height] for l in hand_landmarks[0]])

                average_hand=np.mean(image_points, axis=0)

                world_points=de.GetWorldPoints(model_points,image_points,frame_height, frame_width)
                #world_points=de.GetWorldPoints(model_points,image_points,self.K)
            
                average_loc=1*np.mean(world_points, axis=0)
                self.wp = np.array([-average_loc[0],average_loc[1],-average_loc[2]])
                        
                frame = cv.circle(frame, (int(average_hand[0]),int(average_hand[1])), radius=10, color=(255, 0, 255), thickness=5)
                #print('wp: ', self.wp)

                #                 ^
                #                 |Y
                #                 |
                #   hand    <----- cam
                #             Z    \
                #                   \
                #                    X   

    
                # self.hand_pose.positions = self.wp
                # self.hand_pose.time_from_start = self.img_msg.header.stamp
                # self.handpose_pub.publish(self.hand_pose)

                for obj in self.object_data:
                    
                    score = self.object_distances(np.array(obj['h']), self.wp)
                    score_list.append(score)

                #print(score_list)

                if score_list:
                    # Find the maximum score and corresponding object ID
                    min_score_index = np.argmin(score_list)
                    min_score = score_list[min_score_index]
                    min_score_id = self.object_data[min_score_index]['id']
                
                    # Print the ID and score if the maximum score is greater than 0.5
                    if min_score < self.mindistance:
                        print(f"{min_score_id}: {min_score}")
                        output_vector = np.zeros_like(score_list)
                        output_vector[min_score_index] = 1

                        #publish the best prediction
                        prediction = Prediction()
                        prediction.header.stamp = rospy.Time.now()
                        
                        prediction.probabilities = output_vector
                        prediction.labels = [obj['id'] for obj in self.object_data]
                        prediction.best = min_score_index
                        prediction.best_label = min_score_id
                        prediction.best_probability = min_score

                        self.prediction_pub.publish(prediction)



            # FPS=1/(self.t1-self.t2)
            # self.t2=copy.copy(self.t1)
            # FPS = int(FPS)
            # #print('FPS: ', FPS)

        #Convert OpenCV image frame to ROS image and publish
        frame= cv.flip(frame, 0)
        frame = cv.flip(frame, 1)
        self.detection_img=self.bridge.cv2_to_imgmsg(frame, "bgr8")
        self.detection_img.header = self.img_msg.header

        if self.detection_img:
            #self.detection_img.header.stamp = rospy.Time.now()
            self.handimg_pub.publish(self.detection_img)

    

if __name__ == '__main__':
    
    relay_obj = gesture_tracker()

    rospy.spin()
    print ('hand_tracker Node Exit()!')