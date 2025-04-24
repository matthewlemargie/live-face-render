#!/usr/bin/env/ python

import cv2
import mediapipe as mp
import numpy as np
from multiprocessing import shared_memory
import struct

LANDMARK_INDICES = {
    "Top": 10,
    "Bottom": 152,
    "Left": 234,
    "Right": 454
}

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Shared memory setup
SHM_NAME = "landmarks_shm"
size_of_float = 4
# add 1 for null byte
BUFFER_SIZE = 468 * 3 * size_of_float + 1

# Create shared memory
shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=BUFFER_SIZE)

cap = cv2.VideoCapture(0)

try:
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            h, w, _ = frame.shape

            # Flip the frame horizontally and convert to RGB
            # frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame
            results = face_mesh.process(rgb_frame)

            # Extract landmarks
            landmarks_data = []
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    coords = {}
                    for name, idx in LANDMARK_INDICES.items():
                        lm = face_landmarks.landmark[idx]
                        coords[name] = np.array([lm.x, lm.y, lm.z])

                    # Calculate center of the face
                    center = np.mean([coords["Top"], coords["Bottom"], coords["Left"], coords["Right"]], axis=0)

                    # Scale the center to pixel coordinates
                    center_int = (int(center[0] * w), int(center[1] * h))  # Only scale x and y for 2D positioning

                    # Calculate vectors from center
                    vertical_vec = coords["Top"] - coords["Bottom"]
                    horizontal_vec = coords["Right"] - coords["Left"]

                    scale = 1.0 / np.linalg.norm(coords["Right"] - coords["Left"]) 

                    vertical_vec = vertical_vec / np.linalg.norm(vertical_vec)
                    horizontal_vec = horizontal_vec / np.linalg.norm(horizontal_vec)

                    # Calculate cross products for direction vectors
                    # Ensure that the cross product is not too small
                    z_vec = np.cross(horizontal_vec, vertical_vec)
                    z_vec = z_vec / np.linalg.norm(z_vec)

                    x_vec = -np.cross(vertical_vec, z_vec)
                    x_vec = x_vec / np.linalg.norm(x_vec)
                    y_vec = vertical_vec

                    center = np.append(center, 1)
                    x_vec = np.append(x_vec, 0)
                    y_vec = np.append(y_vec, 0)
                    z_vec = np.append(z_vec, 0)

                    A = np.array([x_vec, y_vec, z_vec, center]).T
                    A_inv = np.linalg.inv(A) 

                    for lm in face_landmarks.landmark:
                        point = np.array([lm.x, lm.y, lm.z, 1])
                        transformed = scale * A_inv @ point
                        landmarks_data.append((-transformed[0], transformed[1], -transformed[2]))
                    # for landmark in face_landmarks.landmark:
                        # landmarks_data.append((landmark.x - 0.5, -(landmark.y - 0.5), landmark.z))
                        

            # Serialize data to a binary format
            packed_data = struct.pack(
                f"{len(landmarks_data)*3}f", *[coord for lm in landmarks_data for coord in lm]
            )

            # Write to shared memory
            shm.buf[:len(packed_data)] = packed_data
            shm.buf[len(packed_data)] = 0  # Write a single null byte

except KeyboardInterrupt:
    print("Stopping...")

finally:
    # Cleanup
    cap.release()
    shm.close()
    shm.unlink()

