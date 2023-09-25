#!/usr/bin/env python
# Maleen Jayasurya @ UTS-RI

import rospy
import copy
import time
import os, sys
import mediapipe as mp
import cv2 as cv 
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from decimal import Decimal
from pathlib import Path 
from trajectory_msgs.msg import JointTrajectoryPoint
from prediction_msgs.msg import Prediction

class predict_gaze:

    # Initialize the class
    def __init__(self):
        
        rospy.init_node('predict_gaze', anonymous=True, log_level=rospy.DEBUG)	

        # Get ROS parameters
        self.gaze_hit_threshold = rospy.get_param('~gaze_hit_threshold', 0.995)
        self.object_data = rospy.get_param('~object_locations', [])

        # Subscribe to hand trajectory topic
        self.handtraj_sub = rospy.Subscriber('/gaze_tracker/gaze_traj', JointTrajectoryPoint, self.gazetraj_cb)        
        self.prediction_pub = rospy.Publisher('~prediction', Prediction, queue_size=10)
      
  
    def __del__(self):
        print('Shutting down the eyeface_tracker object')


    # Callback for handling hand trajectory
    def gazetraj_cb(self, gazetraj):
          
        G = gazetraj.velocities
        x  = gazetraj.positions

        score_list = []
        for obj in self.object_data:
            if len(obj['g']) != len(G) or len(obj['g']) != len(x):                    
                print('Object data has incorrect dimensions')
                return

            score = self.gazehit(np.array(G), np.array(x), np.array(obj['g']))

            score_list.append(score)

        if score_list:
            # Find the maximum score and corresponding object ID
            max_score_index = np.argmax(score_list)
            max_score = score_list[max_score_index]
            max_score_id = self.object_data[max_score_index]['id']
        
            # Print the ID and score if the maximum score is greater than 0.5
            if max_score > self.gaze_hit_threshold:
                print(f"{max_score_id}: {max_score}")
                output_vector = np.zeros_like(score_list)
                output_vector[max_score_index] = 1

                #publish the best prediction
                prediction = Prediction()
                prediction.header.stamp = gazetraj.time_from_start 
                
                prediction.probabilities = output_vector
                prediction.labels = [obj['id'] for obj in self.object_data]
                prediction.best = max_score_index
                prediction.best_label = max_score_id
                prediction.best_probability = max_score

                self.prediction_pub.publish(prediction)

                
    def gazehit(self,G, X_p, X_e):

        gazevec=X_e - X_p

        dotscore=np.dot(G, gazevec)

        denom=np.linalg.norm(G)*np.linalg.norm(gazevec)
        score=(dotscore/denom)

        return score
    

if __name__ == '__main__':
    
    relay_obj = predict_gaze()

    rospy.spin()
    print ('predict_gaze Node Exit()!')


