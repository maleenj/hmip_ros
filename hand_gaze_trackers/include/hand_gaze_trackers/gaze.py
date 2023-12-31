import cv2
import numpy as np
from hand_gaze_trackers.helpers import relative, relativeT
import pandas as pd
from pathlib import Path 


def gaze(frame, points, camera_matrix):

    trans=0

    filepath = Path('/home/maleen/out.csv')  

    """
    The gaze function gets an image and face landmarks from mediapipe framework.
    The function draws the gaze direction into the frame.
    """

    '''
    2D image points.
    relative takes mediapipe points that is normalized to [-1, 1] and returns image points
    at (x,y) format
    '''
    image_points = np.array([
        relative(points.landmark[4], frame.shape),  # Nose tip
        relative(points.landmark[152], frame.shape),  # Chin
        relative(points.landmark[263], frame.shape),  # Left eye left corner
        relative(points.landmark[33], frame.shape),  # Right eye right corner
        relative(points.landmark[287], frame.shape),  # Left Mouth corner
        relative(points.landmark[57], frame.shape)  # Right mouth corner
    ], dtype="double")

    '''
    2D image points.
    relativeT takes mediapipe points that is normalized to [-1, 1] and returns image points
    at (x,y,0) format
    '''
    image_points1 = np.array([
        relativeT(points.landmark[4], frame.shape),  # Nose tip
        relativeT(points.landmark[152], frame.shape),  # Chin
        relativeT(points.landmark[263], frame.shape),  # Left eye, left corner
        relativeT(points.landmark[33], frame.shape),  # Right eye, right corner
        relativeT(points.landmark[287], frame.shape),  # Left Mouth corner
        relativeT(points.landmark[57], frame.shape)  # Right mouth corner
    ], dtype="double")

    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0, -63.6, -12.5),  # Chin
        (-43.3, 32.7, -26),  # Left eye, left corner
        (43.3, 32.7, -26),  # Right eye, right corner
        (-28.9, -28.9, -24.1),  # Left Mouth corner
        (28.9, -28.9, -24.1)  # Right mouth corner
    ])

    '''
    3D model eye points
    The center of the eye ball
    '''
    Eye_ball_center_right = np.array([[-29.05], [32.7], [-39.5]])
    Eye_ball_center_left = np.array([[29.05], [32.7], [-39.5]])  # the center of the left eyeball as a vector.

    '''
    camera matrix estimation
    '''
    # focal_length = frame.shape[1]
    focal_length_x = camera_matrix[0][0]
    focal_length_y = camera_matrix[1][1]    
    center = (frame.shape[1] / 2, frame.shape[0] / 2)
    camera_matrix =  np.array(
        [[focal_length_x, 0, center[0]],
         [0, focal_length_y, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    # 2d pupil location
    left_pupil = relative(points.landmark[468], frame.shape)
    right_pupil = relative(points.landmark[473], frame.shape)

    # Transformation between image point to world point
    _, transformation, _ = cv2.estimateAffine3D(image_points1, model_points)  # image to world transformation

    if transformation is not None:  # if estimateAffine3D secsseded
        # project pupil image point into 3d world point 
        trans=1
        pupil_world_cordL = transformation @ np.array([[left_pupil[0], left_pupil[1], 0, 1]]).T
        pupil_world_cordR = transformation @ np.array([[right_pupil[0], right_pupil[1], 0, 1]]).T

        X_p= np.array((pupil_world_cordL+pupil_world_cordR)/2)

        #print('X_p', X_p)

        # 3D gaze point (10 is arbitrary value denoting gaze distance)
        SL = Eye_ball_center_left + (pupil_world_cordL - Eye_ball_center_left) * 10
        SR = Eye_ball_center_right + (pupil_world_cordR - Eye_ball_center_right) * 10

        S=(SL+SR)/2

        #print('S', S)


        # Project a 3D gaze direction onto the image plane.
        (eye_pupil2D_L, _) = cv2.projectPoints((int(SL[0]), int(SL[1]), int(SL[2])), rotation_vector,
                                             translation_vector, camera_matrix, dist_coeffs)
        (eye_pupil2D_R, _) = cv2.projectPoints((int(SR[0]), int(SR[1]), int(SR[2])), rotation_vector,
                                        translation_vector, camera_matrix, dist_coeffs)
        
        # project 3D head pose into the image plane
        (head_pose, _) = cv2.projectPoints((int(pupil_world_cordL[0]), int(pupil_world_cordL[1]), int(40)),
                                           rotation_vector,
                                           translation_vector, camera_matrix, dist_coeffs)
        # correct gaze for head rotation
        gazeL = left_pupil + (eye_pupil2D_L[0][0] - left_pupil) - (head_pose[0][0] - left_pupil)
        gazeR = right_pupil + (eye_pupil2D_R[0][0] - right_pupil) - (head_pose[0][0] - right_pupil)


        # # Draw gaze line into screen
        # p1 = (int(left_pupil[0]), int(left_pupil[1]))
        # p2 = (int(gazeL[0]), int(gazeL[1]))

        # p3 = (int(right_pupil[0]), int(right_pupil[1]))
        # p4 = (int(gazeR[0]), int(gazeR[1]))

        p5 = (int((right_pupil[0]+left_pupil[0])/2), int(right_pupil[1]))
        p6 = (int((gazeR[0]+gazeL[0])/2), int((gazeR[1]+gazeL[1])/2))

        #cv2.line(frame, p1, p2, (0, 0, 255), 2)
        #cv2.line(frame, p3, p4, (0, 0, 255), 2)
        cv2.line(frame, p5, p6, (0, 0, 255), 2)

        #print('SL', SL)
        #print('SR', gazeR)

       

        # df = pd.DataFrame({'x': ['Raphael', 'Donatello'],
        #            'y': ['red', 'purple'],
        #            'z': ['sai', 'bo staff']})

        # df1 = pd.DataFrame(G)
        # df1.to_csv(filepath)

    else:
        #print('transformation is None')
        trans=0
        X_p= np.array([[0,0,0]])
        S= np.array([[0,0,0]])


    return frame , X_p, S, trans





