#!/usr/bin/env python
# Maleen Jayasurya @ UTS-RI
import roslib;
import rospy
import copy
import time
import os, sys
import numpy as np
from geometry_msgs.msg import Vector3Stamped, Pose, Point
from sensor_msgs.msg import JointState
from prediction_msgs.msg import Prediction
from std_msgs.msg import String
from scipy.stats import halfnorm
from trajectory_msgs.msg import JointTrajectoryPoint
from threading import Lock
from tf.transformations import quaternion_from_euler

UPDATE_RATE = 30 # Hz
MINIMAL_PREDICTION_TIME = float(0.5) # seconds

# Define the attention_perception class
class attention_perception:

    # Initialize the class
    def __init__(self):
        
        rospy.init_node('attention_perception', anonymous=True, log_level=rospy.DEBUG)	

        self.velocity_noise = rospy.get_param('~velocity_noise', 0.05)
        self.object_data = rospy.get_param('~object_locations', [])

        self.setup_parameters()

        self.eye_predict_lock = Lock()
        self.hand_predict_lock = Lock()

        # Subscribe to hand trajectory topic
        self.hand_cb = rospy.Subscriber('/predict_hand_vf/prediction', Prediction, self.predict_hand_cb)        
        self.gaze_cb = rospy.Subscriber('/predict_gaze/prediction', Prediction, self.predict_gaze_cb)        
        self.hand_track_cb = rospy.Subscriber('/hand_tracker/hand_traj', JointTrajectoryPoint, self.hand_track_cb)        

        self.handprediction_pub = rospy.Publisher('~hand_prediction', Prediction, queue_size=10)
        self.gazeprediction_pub = rospy.Publisher('~gaze_prediction', Prediction, queue_size=10)
        self.combprediction_pub = rospy.Publisher('~comb_prediction', Prediction, queue_size=10)


        # Create a timer that calls the 'publish_image' function every 0.1 seconds (10 Hz)
        rospy.Timer(rospy.Duration(1/UPDATE_RATE), self.prediction_update)

    def setup_parameters(self):

        #init vector fields hand    
        self.vf_class_means = np.ones_like(self.object_data)*1.0        
        self.vf_class_stds= np.ones_like(self.object_data)*0.2
        self.vf_posteriors = np.zeros_like(self.object_data)
        
        self.initialise_gazestats()

        self.hand_prediction = Prediction()
        self.gaze_prediction = Prediction()

        self.prediction_str = ''
        self.hand_posterior_threshold = -0.09075062077057232
        self.gaze_posterior_threshold = -1.3862943611198906
        self.combined_posterior_threshold = -2.434
        
        self.time_since_reset = rospy.get_time()
        self.hand_time = rospy.get_rostime()
        self.gaze_time = rospy.get_rostime()
        self.reset_eye_time_threshold = 0.1

        self.start_accumulating_predictions = False
        self.timeout_threshold = 2.0 #maximum prediction time is 3seconds
        self.final_prediction_list = []

    def initialise_gazestats(self):    
        self.gaze_counts=np.ones_like(self.object_data)
        self.gaze_posteriors=np.zeros_like(self.object_data)
        self.gaze_sge=0

    # Destructor
    def __del__(self):
        print('Shutting down the hand_tracker object')

    #hand trajectory callback to reset gaze counts
    def hand_track_cb(self, handtraj):
      with self.eye_predict_lock:
        
        #check if velocities are zero and reset gaze states
        if np.abs(handtraj.velocities[0]) != 0 or np.abs(handtraj.velocities[1]) != 0 or np.abs(handtraj.velocities[2]) != 0:
            self.time_since_reset = rospy.get_time()

        if rospy.get_time() - self.time_since_reset > self.reset_eye_time_threshold:    
            # print(handtraj.time_from_start.to_sec())
            self.initialise_gazestats()

    #hand prediction callback
    def predict_hand_cb(self, prediction):
        
      with self.hand_predict_lock:
        #process hand prediction
        if prediction.probabilities:

            #update vf posteriors
            # print(prediction.probabilities)
            self.update_vf_posteriors(prediction.probabilities)

            best_index = np.argmax(self.vf_posteriors)
            best_score = self.vf_posteriors[best_index]
 
            # print(self.vf_posteriors)
            if best_score > self.hand_posterior_threshold:
                
                # update hand prediction object
                #self.hand_time=rospy.get_rostime()
                self.hand_prediction = prediction
                self.hand_prediction.header.stamp = rospy.get_rostime()
                self.hand_prediction.header.frame_id = 'hand'
                self.hand_prediction.probabilities = copy.copy(self.vf_posteriors)
                self.hand_prediction.best = best_index
                self.hand_prediction.best_probability = best_score
                self.hand_prediction.best_label = self.object_data[best_index]['id']
       

    #gaze prediction callback                
    def predict_gaze_cb(self, prediction):

      with self.eye_predict_lock:           
        self.gaze_counts=self.gaze_counts+np.array(prediction.probabilities)
        self.gaze_posteriors=self.gaze_counts/np.sum(self.gaze_counts)
        #print('posteriors: ', self.gaze_posteriors)

        self.gaze_sge = -np.sum(self.gaze_posteriors * np.log2(np.array(self.gaze_posteriors, dtype=float)))/(np.log2(len(self.gaze_posteriors)))
        #print('sge: ', self.gaze_sge)
        self.gaze_posteriors=(1-self.gaze_sge)*np.log(np.array(self.gaze_posteriors, dtype=float))
        best_index = np.argmax(self.gaze_posteriors)
        best_score = self.gaze_posteriors[best_index]
        
        
        if best_score > self.gaze_posterior_threshold:
            
            #self.gaze_time=rospy.get_rostime()
            self.gaze_prediction = prediction
            self.gaze_prediction.header.stamp = rospy.get_rostime()
            self.gaze_prediction.header.frame_id = 'gaze'
            self.gaze_prediction.probabilities = copy.copy(self.gaze_posteriors)
            self.gaze_prediction.best = best_index
            self.gaze_prediction.best_probability = best_score
            self.gaze_prediction.best_label = self.object_data[best_index]['id']
            

    #update prediction based on hand and gaze prediction
    def prediction_update(self, event):
    
        time_now = rospy.get_rostime()

        if self.hand_prediction and self.gaze_prediction :

            #calculate time to each prediction
            time_to_hand = time_now - self.hand_prediction.header.stamp
            time_to_gaze = time_now - self.gaze_prediction.header.stamp 

            final_prediction = Prediction()

            success = False
            #combine predictions
            if(time_to_hand.to_sec() < MINIMAL_PREDICTION_TIME and time_to_gaze.to_sec() < MINIMAL_PREDICTION_TIME):
                final_prediction.probabilities = self.hand_prediction.probabilities + self.gaze_prediction.probabilities

                best_index = np.argmax(final_prediction.probabilities)
                best_score = final_prediction.probabilities[best_index]

                if best_score > self.combined_posterior_threshold:
                                        
                    final_prediction.header.stamp = rospy.get_rostime()
                    final_prediction.header.frame_id = 'combined'
                    final_prediction.best = best_index
                    final_prediction.best_probability = best_score
                    final_prediction.best_label = self.object_data[best_index]['id']
                    # print('combined: ', final_prediction.best_label)
                    self.handprediction_pub.publish(self.hand_prediction)
                    self.gazeprediction_pub.publish(self.gaze_prediction)
                    self.combprediction_pub.publish(final_prediction)
                    success = True    

            elif time_to_hand.to_sec() < MINIMAL_PREDICTION_TIME:
                # print("hand: " + self.hand_prediction.best_labecakt   l)
                final_prediction = self.hand_prediction
                self.handprediction_pub.publish(self.hand_prediction)
                self.combprediction_pub.publish(final_prediction)
                success = True

            elif time_to_gaze.to_sec() < MINIMAL_PREDICTION_TIME:
                # print("gaze: " + self.gaze_prediction.best_label)
                final_prediction = self.gaze_prediction
                self.gazeprediction_pub.publish(self.gaze_prediction)
                self.combprediction_pub.publish(final_prediction)
                success = True

            else:
                # print('No prediction available')   
                pass

            #convert best_index to geometry Pose
            if success:
                if self.start_accumulating_predictions:
                    p = copy.copy(final_prediction)
                    self.final_prediction_list.append(p)
                     
                    #stop when velocties is 0 TODO
                    #if len(self.final_prediction_list) > 1:
                        
                self.combprediction_pub.publish(final_prediction)
        
            #publish final prediction if available
            # if success:
                
            #     self.combprediction_pub.publish(final_prediction)

    def update_vf_posteriors(self,vf_metric):
        prior_c = 0.5

        #Scores for Vector Field
        i=0
        for x in vf_metric:
            self.vf_posteriors[i] = np.log(prior_c*self.half_normal_pdf(x, self.vf_class_means[i], self.vf_class_stds[i]))
            i=i+1
                    
    def half_normal_pdf(self, x, mean, std):

        # Half normal probability density function with adjusted mean
        return halfnorm.pdf(-x+mean, loc=0, scale=std)

if __name__ == '__main__':

    attention_perception_ = attention_perception()
    
    rospy.spin()
    print ('attention_perception Node Exit()!')