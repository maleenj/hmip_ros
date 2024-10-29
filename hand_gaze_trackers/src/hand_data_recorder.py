#!/usr/bin/env python
import rospy
from trajectory_msgs.msg import JointTrajectoryPoint
import csv
import time
import numpy as np  # Import NumPy for numerical calculations
from std_srvs.srv import Empty, EmptyResponse

class DataRecorder:
    def __init__(self):
        rospy.init_node('hand_data_recorder')
        self.subscriber = rospy.Subscriber('hand_traj', JointTrajectoryPoint, self.callback)
        self.service = rospy.Service('record_hand_data', Empty, self.handle_record_data)
        self.data = []
        self.record = False
        self.max_data_points = 30
        self.max_velocity = 2.0  # Define the maximum velocity threshold

    def calculate_weight(self, velocities):
        # Calculate the velocity magnitude
        vel_magnitude = np.linalg.norm(velocities)
        # Calculate weight based on velocity
        if vel_magnitude > self.max_velocity:
            return 0
        else:
            return 1 - (vel_magnitude / self.max_velocity)

    def callback(self, msg):
        if self.record and len(self.data) < self.max_data_points:
            # Calculate weight for this data point
            weight = self.calculate_weight(np.array([msg.velocities[0], msg.velocities[1], msg.velocities[2]]))
            self.data.append([
                msg.positions[0], msg.positions[1], msg.positions[2],
                msg.velocities[0], msg.velocities[1], msg.velocities[2],
                rospy.get_time(), weight
            ])
            # Check if required data points are collected
            if len(self.data) >= self.max_data_points:
                self.record = False
                self.save_to_csv()

    def handle_record_data(self, req):
        self.record = True  # Start recording
        rospy.loginfo("Recording started")
        return EmptyResponse()

    def save_to_csv(self):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = 'hand_trajectory_data_{}.csv'.format(timestamp)
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['X', 'Y', 'Z', 'Vx', 'Vy', 'Vz', 'Timestamp', 'Weight'])
            writer.writerows(self.data)
        self.data = []  # Clear data after saving
        rospy.loginfo("Data recorded and saved to CSV as {}".format(filename))

if __name__ == '__main__':
    recorder = DataRecorder()
    rospy.spin()
