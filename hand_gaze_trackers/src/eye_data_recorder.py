#!/usr/bin/env python
import rospy
from trajectory_msgs.msg import JointTrajectoryPoint
import csv
import time
import numpy as np
from std_srvs.srv import Empty, EmptyResponse

class EyeDataRecorder:
    def __init__(self):
        rospy.init_node('eye_data_recorder')
        self.subscriber = rospy.Subscriber('gaze_traj', JointTrajectoryPoint, self.callback)
        self.service = rospy.Service('record_eye_data', Empty, self.handle_record_data)
        self.gaze_data = []  # List to hold gaze direction vectors
        self.data = []  # List to hold complete data
        self.record = False
        self.max_data_points = 30
        self.radius = 1.0
        self.n_points = 200
        self.discrete_points = self.discretize_sphere(self.radius, self.n_points)

    def discretize_sphere(self, radius, n_points):
        indices = np.arange(0, n_points, dtype=float) + 0.5
        phi = np.arccos(1 - 2*indices/n_points)
        theta = np.pi * (1 + 5**0.5) * indices
        x, y, z = radius * np.cos(theta) * np.sin(phi), radius * np.sin(theta) * np.sin(phi), radius * np.cos(phi)
        return np.stack((x, y, z), axis=-1)

    def calculate_sge(self):
        fixation_counts = np.zeros(len(self.discrete_points))
        for gaze in self.gaze_data:
            distances = np.dot(self.discrete_points, gaze)
            nearest_point = np.argmax(distances)
            fixation_counts[nearest_point] += 1

        total_fixations = np.sum(fixation_counts)
        if total_fixations == 0:
            return 0  # No data to process

        proportions = fixation_counts / total_fixations
        proportions_with_zeros = proportions.copy()
        proportions_with_zeros[proportions_with_zeros == 0] = 1
        return -np.sum(proportions * np.log2(proportions_with_zeros))

    def callback(self, msg):
        if self.record and len(self.data) < self.max_data_points:
            gaze_vector = np.array([msg.positions[0], msg.positions[1], msg.positions[2]])
            gaze_vector /= np.linalg.norm(gaze_vector)  # Normalize the gaze vector
            self.gaze_data.append(gaze_vector)
            self.data.append([
                msg.positions[0], msg.positions[1], msg.positions[2],
                msg.velocities[0], msg.velocities[1], msg.velocities[2],
                rospy.get_time()
            ])
            if len(self.data) >= self.max_data_points:
                self.record = False
                self.save_to_csv()  # Save data once the maximum data points are reached

    def handle_record_data(self, req):
        self.record = True  # Start recording
        self.gaze_data = []  # Reset gaze data collection
        self.data = []  # Reset complete data collection
        rospy.loginfo("Recording started")
        return EmptyResponse()

    def save_to_csv(self):
        sge = self.calculate_sge()
        weight = 1 - sge
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = 'eye_trajectory_data_{}.csv'.format(timestamp)
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['X', 'Y', 'Z', 'Vx', 'Vy', 'Vz', 'Timestamp', 'Weight'])
            for row in self.data:
                writer.writerow(row + [weight])
        rospy.loginfo("Data recorded and saved to CSV as {}".format(filename))

if __name__ == '__main__':
    recorder = EyeDataRecorder()
    rospy.spin()
