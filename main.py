#!/usr/bin/env/ python

import cv2
import mediapipe as mp
import numpy as np
from multiprocessing import shared_memory
import struct

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

            # Flip the frame horizontally and convert to RGB
            # frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame
            results = face_mesh.process(rgb_frame)

            # Extract landmarks
            landmarks_data = []
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    for landmark in face_landmarks.landmark:
                        landmarks_data.append((landmark.x - 0.5, -(landmark.y - 0.5), landmark.z))

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

