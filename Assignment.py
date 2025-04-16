import cv2
import mediapipe as mp
import numpy as np 
import time
import statistics as st
import os
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float
#from calculateSD import calculateSD #does not work on macOS

def update_right_eye_points(index):
    match index:
        case 33:
            points_right_eye[0] = Point(lm.x, lm.y)
        case 160:
            points_right_eye[1] = Point(lm.x, lm.y)
        case 158:
            points_right_eye[2] = Point(lm.x, lm.y)
        case 133:
            points_right_eye[3] = Point(lm.x, lm.y)
        case 153:
            points_right_eye[4] = Point(lm.x, lm.y)
        case 144:
            points_right_eye[5] = Point(lm.x, lm.y)
        case 145:
            points_right_eye[6] = Point(lm.x, lm.y)
        case 159:
            points_right_eye[7] = Point(lm.x, lm.y)
            
def update_left_eye_points(index):
    match index:
        case 362:
            points_left_eye[0] = Point(lm.x, lm.y)
        case 385:
            points_left_eye[1] = Point(lm.x, lm.y)
        case 387:
            points_left_eye[2] = Point(lm.x, lm.y)
        case 263:
            points_left_eye[3] = Point(lm.x, lm.y)
        case 373:
            points_left_eye[4] = Point(lm.x, lm.y)
        case 380:
            points_left_eye[5] = Point(lm.x, lm.y)
        case 374:
            points_left_eye[6] = Point(lm.x, lm.y)
        case 386:
            points_left_eye[7] = Point(lm.x, lm.y)
    
def normalization(current_value, side):
    if side == 'left':
        return (current_value - min_left_EAR) / (max_left_EAR - min_left_EAR)
    else:
        return (current_value - min_right_EAR) / (max_right_EAR - min_right_EAR)    

def count_down_timer(timer, start, text):
    end = time.time()
    result = int(end - start)
    if result <= timer:
        result = int(timer - result)
        cv2.putText(image, text + str(result+1) + 's', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    if result > timer:
        return 'DONE'
    
def Calibration_timer(start):
    global min_left_EAR, max_left_EAR, min_right_EAR, max_right_EAR, treshold_left_80, treshold_right_80, treshold_left_20, treshold_right_20
    end = time.time()
    result = end - start
    if result <= 6:
        # Save calibration samples
        EAR_calibration()
    else:
        print("Calibrazione terminata!")
        min_left_EAR = min(EAR_left_values)
        max_left_EAR = max(EAR_left_values)
        min_right_EAR = min(EAR_right_values)
        max_right_EAR = max(EAR_right_values)
        print('min left: ', min_left_EAR)
        print('max left: ', max_left_EAR)
        print('min right: ', min_right_EAR)
        print('max right: ', max_right_EAR)
        treshold_left_80 = max_left_EAR * 80 / 100
        treshold_right_80 = max_right_EAR * 80 / 100
        treshold_left_20 = max_left_EAR * 20 / 100
        treshold_right_20 = max_right_EAR * 20 / 100
        treshold_left_80 = normalization(treshold_left_80, 'left')
        treshold_right_80 = normalization(treshold_right_80, 'right')
        treshold_left_20 = normalization(treshold_left_20, 'left')
        treshold_right_20 = normalization(treshold_right_20, 'right')
        print('treshold_left_80: ', treshold_left_80)
        print('treshold_right_80: ', treshold_right_80)
        print('treshold_left_20: ', treshold_left_20)
        print('treshold_right_20: ', treshold_right_20)
        return True
            
def calculate_EAR(points):
    EAR = (abs(points[1].y - points[5].y) + abs(points[2].y - points[4].y)) / (2*(abs(points[0].x - points[3].x)))
    return EAR

def calculate_PERCLOS_80(t):
    print('TIMERS: ', t[0], t[1], t[2], t[3])
    P80 = (t[2] - t[1]) / (t[3] - t[0]) if t[3] != t[0] else 0
    return P80

def EAR_calibration():
    # Metto tutti i valori dell'EAR in due liste, una per occhio, e poi calcolo il massimo e il minimo
    # Calculate the closure of the right eye
    right = calculate_EAR(points_right_eye)
    EAR_right_values.append(right)
    # Calculate the closure of the left eye
    left = calculate_EAR(points_left_eye)
    EAR_left_values.append(left)
    


# ========= Variabili Mediapipe =============
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing_styles = mp.solutions.drawing_styles
mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)

# ======== Variabili globali ===========
numbers_frames_closure_eye = 0
EAR_left_values = []
EAR_right_values = []
calibration_done= False
first_count_down_timer = ''
second_count_down_timer = ''
start = time.time()
min_left_EAR = 0
max_left_EAR = 0
min_right_EAR = 0
max_right_EAR = 0
treshold_left_80 = 0
treshold_right_80 = 0
start_closure_timer = True
t = [0, 0, 0, 0]
previous_ear_left = 0
previous_ear_right = 0
# PERCLOS timers
start_t1 = 0
start_t2 = 0
start_t3 = 0
start_t4 = 0
start_timer_t1 = True
start_timer_t2 = True
start_timer_t3 = True
start_timer_t4 = True
# Treshold PERCLOS
treshold_left_20 = 0
treshold_right_20 = 0
# Variable to understand if i performed the 4 phases of PERCLOS consecutively
blink_phase = 1

while cap.isOpened():
    
    
    points_left_eye = [None] * 8
    points_right_eye = [None] * 8
    totalTime = 0
    # Face orientation
    face_2d = []
    face_3d = []
    right_eye_2d = []
    right_eye_3d = []
    left_eye_2d = []
    left_eye_3d = []
    
    success, image = cap.read()

    # Flip the image horizontally for a later selfie-view display
    # Also convert the color space from BGR to RGB
    if image is None:
        break
        #continue
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # To improve performace
    image.flags.writeable = False
    
    # Get the result
    results = face_mesh.process(image)

    # To improve performance
    image.flags.writeable = True

    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape

    # Left eye indices list
    LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
    # Right eye indices list
    RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]


    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):

                # if idx == 263:
                #     cv2.putText(image, "(263)", (int(lm.x * img_w), int(lm.y * img_h)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
                
                if idx in LEFT_EYE:
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=1, color=(255, 0, 0), thickness=-1)

                # Saving new points position in the lists
                update_left_eye_points(idx)
                update_right_eye_points(idx)
                
                # GET FACE ORIENTATION
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    # Gte the 2D coordinates
                    face_2d.append([x, y])
                    # Get the 3D coordinates
                    face_3d.append([x, y, lm.z])
                # RIGHT EYE POSITION
                if idx == 468 or idx == 33 or idx == 145 or idx == 133 or idx == 159:
                    if idx == 468:
                        right_pupil_2d = (lm.x * img_w, lm.y * img_h)
                        right_pupil_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    # Gte the 2D coordinates
                    right_eye_2d.append([x, y])
                    # Get the 3D coordinates
                    right_eye_3d.append([x, y, lm.z])
                # LEFT EYE POSITION
                if idx == 473 or idx == 362 or idx == 374 or idx == 263 or idx == 386:
                    if idx == 473:
                        left_pupil_2d = (lm.x * img_w, lm.y * img_h)
                        left_pupil_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    # Gte the 2D coordinates
                    left_eye_2d.append([x, y])
                    # Get the 3D coordinates
                    left_eye_3d.append([x, y, lm.z])
                    
    
        # CALIBRATION
        if not calibration_done:
            cv2.putText(image, 'CALIBRATION PHASE', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
            calibration_done = Calibration_timer(start)
            first_count_down_timer = count_down_timer(3, start, 'Keep your eyes OPENED for ')
            if first_count_down_timer == 'DONE' and second_count_down_timer != 'DONE':
                second_count_down_timer = count_down_timer(6, start, 'Keep your eyes CLOSED for ')

        # AFTER CALIBRATION
        else:
            # Calculate the closure of the left eye
            result_left = calculate_EAR(points_left_eye)
            ear_left = normalization(result_left, 'left')
            
            # Calculate the closure of the right eye
            result_right = calculate_EAR(points_right_eye)
            ear_right = normalization(result_right, 'right')
            
            # ------------------------------------------------------------
            # PERCLOS 80
            # ------------------------------------------------------------          
            # Starting timer of t1
            if blink_phase == 1 and ((previous_ear_left > ear_left and ear_left >= treshold_left_80) or (previous_ear_right > ear_right and ear_right >= treshold_right_80)):
                if start_timer_t1 == True:
                    start_timer_t1 = False
                    start_t1 = time.time()
                end = time.time()
                t[0] = end - start_t1
                # print('PHASE 1: ', t[0])
            else: 
                start_timer_t1 = True
                start_t1 = 0
                # Change phase t1 -> t2
                if blink_phase == 1 and ((previous_ear_left > ear_left and ear_left < treshold_left_80) or (previous_ear_right > ear_right and ear_right < treshold_right_80)):
                    blink_phase = 2
                
            # Starting timer of t2
            if blink_phase == 2 and ((previous_ear_left > ear_left and treshold_left_80 > ear_left >= treshold_left_20) or (previous_ear_right > ear_right and treshold_right_80 > ear_right >= treshold_right_20)):
                if start_timer_t2 == True:
                    start_timer_t2 = False
                    start_t2 = time.time()
                end = time.time()
                t[1] = end - start_t2
                # print('PHASE 2: ', t[1])
            else: 
                start_timer_t2 = True
                start_t2 = 0
                # Change phase t2 -> t3
                if blink_phase == 2 and ((previous_ear_left > ear_left and ear_left < treshold_left_20) or (previous_ear_right > ear_right and ear_right < treshold_right_20)):
                    blink_phase = 3
                
            # Starting timer of t3
            if blink_phase == 3 and ((ear_left < treshold_left_20) or (ear_right < treshold_right_20)):
                if start_timer_t3 == True:
                    start_timer_t3 = False
                    start_t3 = time.time()
                end = time.time()
                t[2] = end - start_t3
                # print('PHASE 3: ', t[2])
            else: 
                start_timer_t3 = True
                start_t3 = 0
                # Change phase t3 -> t4
                if blink_phase == 3 and ((previous_ear_left < ear_left and ear_left >= treshold_left_20) or (previous_ear_right < ear_right and ear_right >= treshold_right_20)):
                    blink_phase = 4
                
            # Starting timer of t4
            if (blink_phase == 4 or blink_phase == 3) and (((previous_ear_left < ear_left and treshold_left_80 >= ear_left >= treshold_left_20) or ear_left < treshold_left_20) or ((previous_ear_right < ear_right and treshold_right_80 >= ear_right >= treshold_right_20) or ear_right < treshold_right_20)):
                if start_timer_t4 == True:
                    start_timer_t4 = False
                    start_t4 = time.time()
                end = time.time()
                t[3] = end - start_t4
                # print('PHASE 4: ', t[3])
            else: 
                start_timer_t4 = True
                start_t4 = 0
                # Change phase t4 -> PERCLOS calculation
                if blink_phase == 4 and ((previous_ear_left < ear_left and ear_left > treshold_left_80) or (previous_ear_right < ear_right and ear_right > treshold_right_80)):
                    blink_phase = 5
            
            # Calculation of PERCLOS
            if blink_phase == 5:
                print('PHASE 5')
                perclos = calculate_PERCLOS_80(t)
                print('PERCLOS 80: ', perclos)
                t = [0, 0, 0, 0]
                # Reset blink_phase if finished
                blink_phase = 1  
                
            previous_ear_left = ear_left
            previous_ear_right = ear_right
            
            # ------------------------------------------------------------
            # EAR
            # ------------------------------------------------------------
            if (ear_left > treshold_left_80) or (ear_right > treshold_right_80):
                # print('OVER')
                #Se ho gli occhi chiusi starto il contatore
                if start_closure_timer == True:
                    start_closure_timer = False
                    start_timer = time.time()
                end = time.time()
                totalTime = end - start_timer
                if (totalTime >= 3):
                    cv2.putText(image, 'DROWSY DRIVER', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
            else:
                # print('UNDER')
                # Se ho gli occhi aperti azzero il contatore di frame con occhi chiusi
                start_closure_timer = True
                start_timer = 0
            
            # ------------------------------------------------------------
            # Eyes and Head gaze positions
            # ------------------------------------------------------------
            # FACE ORIENTATION
            # Convert into numpy arrays
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)
            left_eye_2d = np.array(left_eye_2d, dtype=np.float64)
            left_eye_3d = np.array(left_eye_3d, dtype=np.float64)
            right_eye_2d = np.array(right_eye_2d, dtype=np.float64)
            right_eye_3d = np.array(right_eye_3d, dtype=np.float64)
            
            # Define the camera matrix
            focal_length = 1*img_w
            cam_matrix = np.array([ [focal_length, 0, img_h/2],
                                      [0, focal_length, img_w/2],
                                      [0, 0, 1] ])
            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)
            
            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            success_left_eye, rot_vec_left_eye, trans_vec_left_eye = cv2.solvePnP(left_eye_3d, left_eye_2d, cam_matrix, dist_matrix)
            success_right_eye, rot_vec_right_eye, trans_vec_right_eye = cv2.solvePnP(right_eye_3d, right_eye_2d, cam_matrix, dist_matrix)
            
            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)
            rmat_left_eye, jac_left_eye = cv2.Rodrigues(rot_vec_left_eye)
            rmat_right_eye, jac_right_eye = cv2.Rodrigues(rot_vec_right_eye)
            
            # Get euler angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
            angles_left_eye, mtxR_left_eye, mtxQ_left_eye, Qx_left_eye, Qy_left_eye, Qz_left_eye = cv2.RQDecomp3x3(rmat_left_eye)
            angles_right_eye, mtxR_right_eye, mtxQ_right_eye, Qx_right_eye, Qy_right_eye, Qz_right_eye = cv2.RQDecomp3x3(rmat_right_eye)
            
            pitch = angles[0] * 1800
            yaw = -angles[1] * 1800
            roll = 180 + (np.arctan2(points_right_eye[0].y - points_left_eye[3].y, points_right_eye[0].x - points_left_eye[3].x) * 180/np.pi)
            if roll > 180:
                roll = roll - 360
                
            pitch_left_eye = angles_left_eye[0] * 1800
            yaw_left_eye = angles_left_eye[1] * 1800
            pitch_right_eye = angles_right_eye[0] * 1800
            yaw_right_eye = angles_right_eye[1] * 1800
            
            # Check if the driver is distracted
            if -30 > yaw > 30 or -30 > yaw_left_eye > 30 or -30 > yaw_right_eye > 30:
                cv2.putText(image, 'DISTRACTED DRIVER', (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                
            # Display directions nose
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] - yaw * 10), int(nose_2d[1] - pitch * 10))
            cv2.line(image, p1, p2, (255, 0, 0), 3)
            # Display directions left eye
            l_eye_projection, l_eye_jacobian = cv2.projectPoints(left_eye_3d, rot_vec_left_eye, trans_vec, cam_matrix, dist_matrix)
            p1_left_eye = (int(left_pupil_2d[0]), int(left_pupil_2d[1]))
            p2_left_eye = (int(left_pupil_2d[0] + yaw_left_eye * 1.25), int(left_pupil_2d[1] - pitch_left_eye * 1.25))
            cv2.line(image, p1_left_eye, p2_left_eye, (0, 255, 0), 3)
            # Display directions right eye
            r_eye_projection, r_eye_jacobian = cv2.projectPoints(right_eye_3d, rot_vec_right_eye, trans_vec, cam_matrix, dist_matrix)
            p1_right_eye = (int(right_pupil_2d[0]), int(right_pupil_2d[1]))
            p2_right_eye = (int(right_pupil_2d[0] + yaw_right_eye * 1.25), int(right_pupil_2d[1] - pitch_right_eye * 1.25))
            cv2.line(image, p1_right_eye, p2_right_eye, (0, 255, 0), 3)
            
        cv2.imshow('output window', image)       

    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release() 