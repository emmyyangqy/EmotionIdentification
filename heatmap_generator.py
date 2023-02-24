import math
import numpy as np
#from google.colab.patches import cv2_imshow
import mediapipe as mp
import glob
import cv2
import matplotlib.pyplot as plt
import os
from mlxtend.preprocessing import minmax_scaling

#emotion = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]

emotion = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]

# Set upper and lower thresholds
upper_threshold = 100
lower_threshold = 0.35

name = f'data/CK+ - zeroL{lower_threshold} U{upper_threshold}'
os.makedirs(name, exist_ok=True)


for emotion_name in emotion:
    emotion_dir = os.path.join(name, emotion_name)
    os.makedirs(emotion_dir, exist_ok=True)

    emotion_name1 = emotion_name

    mat = np.load('a.npy') 

    mp_face_mesh = mp.solutions.face_mesh

    # Load drawing_utils and drawing_styles
    mp_drawing = mp.solutions.drawing_utils 
    mp_drawing_styles = mp.solutions.drawing_styles

    DESIRED_HEIGHT = 480
    DESIRED_WIDTH = 480

    total_points = 468
    Matrix = np.zeros(shape=(total_points, 3))

    img_path = glob.glob(f"data/CK+48 - withneutral/{emotion_name1}/*.png")

    neutrallist = []

    imgnumber = 0

    for image_name in img_path:

        imgnumber += 1
        img = cv2.imread(image_name)

        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5) as face_mesh:

            results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            # Draw face landmarks of each face.
            print(f'Face landmarks of', image_name)
            if not results.multi_face_landmarks:
                continue
            annotated_image = img.copy()
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())

        
            i = 0

            while i < total_points:
                Matrix[i][0]=results.multi_face_landmarks[0].landmark[i].x
                Matrix[i][1]=results.multi_face_landmarks[0].landmark[i].y
                Matrix[i][2]=results.multi_face_landmarks[0].landmark[i].z
                i += 1

            Dist_Matrix = np.zeros(shape=(total_points, total_points))
            b=0

            while b < total_points:
                a=0
                while a < total_points:
                    delta_x_squared = math.pow((Matrix[a][0]-Matrix[b][0]), 2)
                    delta_y_squared = math.pow((Matrix[a][1]-Matrix[b][1]), 2)
                    delta_z_squared = math.pow((Matrix[a][2]-Matrix[b][2]), 2)
                    Dist_Matrix[a][b]= math.sqrt(delta_x_squared + delta_y_squared + delta_z_squared)
                    a += 1
                b += 1

            # MinMax
            Dist_Matrix_scaled = minmax_scaling(Dist_Matrix, columns=list(range(0, 468)))
            
            # Thresholds
            Dist_Matrix_scaled = np.where(Dist_Matrix_scaled < lower_threshold, 0, Dist_Matrix_scaled)
            Dist_Matrix_scaled = np.where(Dist_Matrix_scaled > upper_threshold, upper_threshold, Dist_Matrix_scaled)
            
            # MinMax
            Dist_Matrix_scaled = minmax_scaling(Dist_Matrix_scaled, columns=list(range(0, 468)))

            fig, ax = plt.subplots(figsize=(1, 1))
            fig.subplots_adjust(bottom=0.25)
            baselinesubstacted = Dist_Matrix_scaled-mat
            plt.imshow(baselinesubstacted[:,0,:], cmap = "hot", interpolation='nearest')

            plt.savefig(f'/Users/Emmyl/ML/Emotion-detection/src/{name}/{emotion_name1}/'+f'{emotion_name1}{imgnumber}.png')
            plt.close('all')

