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

# Timer che registra la chiusura dell'occhio per 10 secondi
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
    
    
    points_left_eye = [None] * 7
    points_right_eye = [None] * 7
    totalTime = 0
    
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
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=1, color=(0, 0, 255), thickness=-1)

                # Saving new points position in the lists
                update_left_eye_points(idx)
                update_right_eye_points(idx)
    
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
            print('EAR LEFT: ', ear_left)
            
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
                    cv2.putText(image, 'DROWSY DRIVER', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 1)
            else:
                # print('UNDER')
                # Se ho gli occhi aperti azzero il contatore di frame con occhi chiusi
                start_closure_timer = True
                start_timer = 0

        cv2.imshow('output window', image)       

    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()