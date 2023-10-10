import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
import numpy as np
import time
import cv2
from include.DepthExtractor import GetWorldPoints
from matplotlib import pyplot as plt



cap = cv2.VideoCapture(0)


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
recognizer = vision.GestureRecognizer.create_from_options(options)


results = []


A=np.array([ 0.03560906, -0.01108182,  0.20221273])
B=np.array([-0.06751596, -0.0060403,   0.17442674])
C=np.array([-0.05898311, -0.05961555,  0.22554329])
D=np.array([ 0.04770224, -0.05381817,  0.23134669])



def object_distances(locations, wp):

    distance=np.linalg.norm(locations-wp)

    return distance

while cap.isOpened():
    
    success, image = cap.read()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image= cv2.flip(image, 0)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    frame_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 

    recognition_result = recognizer.recognize(mp_image)

    if recognition_result:

        #print(recognition_result)

        hand_landmarks = recognition_result.hand_landmarks
        world_landmarks = recognition_result.hand_world_landmarks

        #print(world_landmarks)

        if recognition_result.gestures:
           

            if recognition_result.handedness[0][0].category_name == 'Right' and recognition_result.gestures[0][0].category_name == 'Pointing_Up':
    
                #print(time.time(), recognition_result.handedness[0][0].category_name)

                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks[0]
                ])

                    #print(hand_landmarks_proto)

                image_points = np.float32([[l.x * frame_width, l.y * frame_height] for l in hand_landmarks_proto.landmark])

                model_points = np.float32([[-l.x, -l.y, -l.z] for l in world_landmarks[0]])  

                world_points=GetWorldPoints(model_points,image_points, frame_height, frame_width)

                average_loc=np.mean(world_points, axis=0)

                wp = np.array([-average_loc[0],average_loc[1],-average_loc[2]])

                average_hand=np.mean(image_points, axis=0)
                image = cv2.circle(image, (int(average_hand[0]),int(average_hand[1])), radius=10, color=(255, 0, 255), thickness=5)

                scoreA=object_distances(A, wp)
                scoreB=object_distances(B, wp)
                scoreC=object_distances(C, wp)
                scoreD=object_distances(D, wp)

                score=np.array([scoreA,scoreB, scoreC, scoreD])
                minscore=np.argmin(score)
                countnum=1
                #print('Sore is :' ,score)
                mindist=0.05
      
                if minscore ==0 and scoreA<mindist:
                    print('is A and sore is :' ,score)
                elif minscore ==1 and scoreB<mindist:
                    print('is B and sore is :' ,score)
                elif minscore ==2 and scoreC<mindist:
                    print('is C and sore is :' ,score)
                elif minscore ==3 and scoreD<mindist:
                    print('is D and sore is :' ,score)
                else:
                    print('None and sore is :' ,score)
                
            # elif recognition_result.handedness[0][0].category_name == 'Left':


            #     hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            #     hand_landmarks_proto.landmark.extend([
            #         landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks[0]
            #     ])

            #         #print(hand_landmarks_proto)

            #     image_points = np.float32([[l.x * frame_width, l.y * frame_height] for l in hand_landmarks_proto.landmark])

            #     #print(image_points)
            #     average_hand=np.mean(image_points, axis=0)
            #     image = cv2.circle(image, (int(average_hand[0]),int(average_hand[1])), radius=10, color=(255, 0, 255), thickness=5)

        
        # top_gesture = recognition_result.gestures[0][0]
        # # hand_landmarks = recognition_result.hand_landmarks
        # # results.append((top_gesture, hand_landmarks))

        # gestures = [top_gesture for (top_gesture, _) in results]
        # title = f"{gestures.category_name} ({gestures.score:.2f})"

        # multi_hand_landmarks_list = [multi_hand_landmarks for (_, multi_hand_landmarks) in results]


        # for hand_landmarks in multi_hand_landmarks_list[i]:
        #     hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        #     hand_landmarks_proto.landmark.extend([
        #         landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        #     ])

        #     mp_drawing.draw_landmarks(
        #         image,
        #         hand_landmarks_proto,
        #         mp_hands.HAND_CONNECTIONS,
        #         mp_drawing_styles.get_default_hand_landmarks_style(),
        #         mp_drawing_styles.get_default_hand_connections_style())
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image= cv2.flip(image, 0)
    image= cv2.flip(image, 1)
    
    cv2.imshow('MediaPipe Hands', image)

    
    #time delay for 24 hz
    #time.sleep(0.041666666666666664)
    cv2.waitKey(1)


cap.release()     


