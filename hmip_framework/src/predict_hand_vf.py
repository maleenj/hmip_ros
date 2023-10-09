#!/usr/bin/env python
# Maleen Jayasurya @ UTS-RI
import rospy
import numpy as np
from prediction_msgs.msg import Prediction
from trajectory_msgs.msg import JointTrajectoryPoint
from decimal import Decimal
from hmip_framework import vf3D_class as vf3D
from hmip_framework import dynamic_target as dt
from hmip_framework import vf_recognize as vr


# Define the predict_hand_vf class
class predict_hand_vf:

    # Initialize the class
    def __init__(self):
        
        rospy.init_node('predict_hand_vf', anonymous=True, log_level=rospy.DEBUG)	

        # Get ROS parameters
        self.beta = rospy.get_param('~beta', 1)
        self.disc_param = rospy.get_param('~disc_param', 0.05)
        self.velocity_noise = rospy.get_param('~velocity_noise', 0.07)
        self.object_data = rospy.get_param('~object_locations', [])

        # Initialize attributes
        self.n=[1,1,1] #50
        self.object_count=[]
        self.wp = np.array([0,0,0])
        self.v = np.array([0,0,0])
        self.trajloc=np.empty((0,3))
        self.trajvel=np.empty((0,3))
        self.trajtime=np.empty((0,1))
        self.time_since_reset = rospy.get_time()
        self.reset_time_threshold = 0.1

        self.score_list = []
        self.max_score_index = -1
        self.max_score = 0
        self.max_score_id = ''

        # Initialize vector fields
        self.object_vfs = [] 
        self.initialise_vectorfields(self.object_data)

        # Subscribe to hand trajectory topic
        self.handtraj_sub = rospy.Subscriber('/hand_tracker/hand_traj', JointTrajectoryPoint, self.handtraj_cb)        
        self.prediction_pub = rospy.Publisher('~prediction', Prediction, queue_size=10)
      
    # Destructor
    def __del__(self):
        print('Shutting down the hand_tracker object')


    # Callback for handling hand trajectory
    def handtraj_cb(self, handtraj):

        # Perform length check on message arrays
        if len(handtraj.positions) != len(self.wp) or len(handtraj.velocities) != len(self.v):
            print('handtraj_cb: handtraj.positions and handtraj.velocities are not the same length as self.wp and self.v')
            return
        
        # Update trajectory and velocity
        self.wp = np.array(handtraj.positions)
        self.v = np.array(handtraj.velocities)

        # Calculate time
        trajtimenow=np.float64((handtraj.time_from_start.secs)+(Decimal(handtraj.time_from_start.nsecs)/1000000000))
   
        # Code logic for conditions on velocity   
        # if all(self.v < self.velocity_noise):

        if np.abs(handtraj.velocities[0]) != 0 or np.abs(handtraj.velocities[1]) != 0 or np.abs(handtraj.velocities[2]) != 0:
            self.time_since_reset = rospy.get_time()

        if rospy.get_time() - self.time_since_reset > self.reset_time_threshold:    

        #if np.abs(self.v[0]) < self.velocity_noise and np.abs(self.v[1]) < self.velocity_noise and np.abs(self.v[2]) < self.velocity_noise : 
            # Reinitialize if velocity is below noise threshold
            self.initialise_vectorfields(self.object_data)

            prediction = Prediction()
            prediction.header.stamp = handtraj.time_from_start 

            # Print the ID and score if the maximum score is greater than 0.5
            # if self.max_score > 0.75:
            #     print(f"{self.max_score_id}: {self.max_score}")

            # else:
                # print("None and the scores are above 0.5:", score_list)

            #publish the best prediction
            
            prediction.probabilities = self.score_list
            prediction.labels = [obj['id'] for obj in self.object_vfs]
            prediction.best = self.max_score_index
            prediction.best_label = self.max_score_id
            prediction.best_probability = self.max_score


            self.prediction_pub.publish(prediction)
            self.prediction_pub.publish(prediction)

        else:

            # Otherwise, append new trajectory and velocity data
            self.trajtime=np.append(self.trajtime, trajtimenow)
            self.trajloc=np.vstack((self.trajloc, self.wp))
            self.trajvel=np.vstack((self.trajvel, self.v))
        
            #print('self.trajloc', self.trajloc)
            #print('first row', self.trajloc[0])

            # Calculate the distance mesh based on the current hand position
            distancemesh=self.vf.calc_distances(self.wp[0], self.wp[1], self.wp[2])
            # Calculate the alpha values using sigmoid function based on the distance mesh and beta parameter
            alpha=self.vf.calc_alpha_sigmoid(distancemesh, self.beta)
            # Update the main vector field with the new hand position, velocity and calculated alpha
            u,v,w,behind=self.vf.updateVF(self.v[0],self.v[1],self.v[2], self.wp[0], self.wp[1], self.wp[2],  alpha) #

            # Loop through each object's vector field to update it and calculate the similarity score
            self.score_list = []
            for obj in self.object_vfs:
                 # Update object's vector field using current trajectory data
                updated_vf = dt.dynamicVF3D(self.trajloc, self.trajvel, self.trajtime, obj['h'][0],obj['h'][1],obj['h'][2], obj['vf'], self.beta)
                
                obj['vf'] = updated_vf

                # Calculate the similarity score between the main vector field and object's updated vector field 
                obj['score'] = vr.vf_recognise3D(self.vf, updated_vf, behind)
                
                self.score_list.append(obj['score'])


            # Find the maximum score and corresponding object ID
            self.max_score_index = np.argmax(self.score_list)
            self.max_score = self.score_list[self.max_score_index]
            self.max_score_id = self.object_vfs[self.max_score_index]['id']

            # Print the ID and score if the maximum score is greater than 0.5
            if self.max_score > 0.75:
                print(f"{self.max_score_id}: {self.max_score}")

            # else:
                # print("None and the scores are above 0.5:", score_list)

            #publish the best prediction
            prediction = Prediction()
            prediction.header.stamp = handtraj.time_from_start 
            
            prediction.probabilities = self.score_list
            prediction.labels = [obj['id'] for obj in self.object_vfs]
            prediction.best = self.max_score_index
            prediction.best_label = self.max_score_id
            prediction.best_probability = self.max_score


            self.prediction_pub.publish(prediction)

    def initialise_vectorfields(self, object_data):
    
        self.trajloc=np.empty((0,3))
        self.trajvel=np.empty((0,3))
        self.trajtime=np.empty((0,1))
    
        self.vf = vf3D.VectorField3D(self.n[0], self.n[1], self.n[2], self.disc_param)
        # Initialize vector fields for each object location
        self.object_vfs = []
        for obj in object_data:
            obj_id = obj['id']
            p = obj['h']
            vf_obj = vf3D.VectorField3D(self.n[0], self.n[1], self.n[2], self.disc_param)
            self.object_vfs.append({'id': obj_id, 'h': p, 'vf': vf_obj,'score':0})

        # print(self.object_vfs)

if __name__ == '__main__':
    
    relay_obj = predict_hand_vf()

    rospy.spin()
    print ('predict_hand_vf Node Exit()!')