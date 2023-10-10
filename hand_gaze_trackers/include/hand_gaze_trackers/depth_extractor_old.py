import cv2
import numpy as np

def GetWorldPoints(model_points,image_points, frame_height, frame_width):

    # pseudo camera internals
    channels = 3

    focal_length = frame_width * 0.75
    center = (frame_width/2, frame_height/2)
    camera_matrix = np.array(
                            [[focal_length, 0, center[0]],
                            [0, focal_length, center[1]],
                            [0, 0, 1]], dtype = "double"
                            )
    distortion = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, distortion, flags=cv2.SOLVEPNP_SQPNP)


    transformation = np.eye(4)  # needs to 4x4 because you have to use homogeneous coordinates
    transformation[0:3, 3] = translation_vector.squeeze()
    # the transformation consists only of the translation, because the rotation is accounted for in the model coordinates. Take a look at this (https://codepen.io/mediapipe/pen/RwGWYJw to see how the model coordinates behave - the hand rotates, but doesn't translate

    # transform model coordinates into homogeneous coordinates
    model_points_hom = np.concatenate((model_points, np.ones((21, 1))), axis=1)

    # apply the transformation
    world_points = model_points_hom.dot(np.linalg.inv(transformation).T)

    return world_points