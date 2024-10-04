import cv2
import numpy as np
import mediapipe as mp
import threading as th
import time
import os

# Global variables
x = 0  # X axis head pose
y = 0  # Y axis head pose

X_AXIS_CHEAT = 0
Y_AXIS_CHEAT = 0

# Path for saving screenshots
SCREENSHOT_PATH = (r"Z:\screanshots\New folder")

# Trackers for detecting continuous pose
look_left_start = None
look_right_start = None
pose_duration = 3  # seconds to wait before taking screenshot

def save_screenshot(image, filename):
    if not os.path.exists(SCREENSHOT_PATH):
        os.makedirs(SCREENSHOT_PATH)
    filepath = os.path.join(SCREENSHOT_PATH, filename)
    cv2.imwrite(filepath, image)
    print(f"Screenshot saved as {filepath}")

def pose():
    global x, y, X_AXIS_CHEAT, Y_AXIS_CHEAT, look_left_start, look_right_start

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(1)  # Use the default camera
    mp_drawing = mp.solutions.drawing_utils

    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        return

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Error: Could not read frame.")
            break

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []
        face_ids = [33, 263, 1, 61, 291, 199]

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None)

                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in face_ids:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = 1 * img_w
                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                       [0, focal_length, img_w / 2],
                                       [0, 0, 1]])
                dist_matrix = np.zeros((4, 1), dtype=np.float64)
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                rmat, jac = cv2.Rodrigues(rot_vec)
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                x = angles[0] * 360
                y = angles[1] * 360

                current_time = time.time()

                if y < -10:  # Looking Left
                    if look_left_start is None:
                        look_left_start = current_time  # Start timer
                    elif current_time - look_left_start >= pose_duration:
                        text = "Looking Left"
                        X_AXIS_CHEAT = 1
                        print("Alert: Looking Left for 3 seconds")
                        save_screenshot(image, f"looking_left_{int(current_time)}.png")
                        look_left_start = None  # Reset timer after screenshot
                else:
                    look_left_start = None  # Reset if no longer looking left

                if y > 10:  # Looking Right
                    if look_right_start is None:
                        look_right_start = current_time  # Start timer
                    elif current_time - look_right_start >= pose_duration:
                        text = "Looking Right"
                        X_AXIS_CHEAT = 1
                        print("Alert: Looking Right for 3 seconds")
                        save_screenshot(image, f"looking_right_{int(current_time)}.png")
                        look_right_start = None  # Reset timer after screenshot
                else:
                    look_right_start = None  # Reset if no longer looking right

                text = "Forward" if y >= -10 and y <= 10 else text
                X_AXIS_CHEAT = 0 if y >= -10 and y <= 10 else X_AXIS_CHEAT

                if x < -5:
                    Y_AXIS_CHEAT = 1
                else:
                    Y_AXIS_CHEAT = 0

                display_text = str(int(x)) + "::" + str(int(y)) + " " + text
                cv2.putText(image, display_text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Head Pose Estimation', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    t1 = th.Thread(target=pose)
    t1.start()
    t1.join()
