import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Landmark indices for static facial extremes
LANDMARK_INDICES = {
    "Top": 10,
    "Bottom": 152,
    "Left": 234,
    "Right": 454
}

# Colors for drawing
LANDMARK_COLORS = {
    "Top": (255, 0, 0),
    "Bottom": (0, 0, 255),
    "Left": (0, 255, 255),
    "Right": (255, 0, 255)
}

def angle_between(v1, v2):
    """Returns angle in degrees between two vectors."""
    v1 = np.array(v1)
    v2 = np.array(v2)
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
    angle_rad = np.arccos(dot)
    return np.degrees(angle_rad)

# Start webcam
cap = cv2.VideoCapture(0)

# FaceMesh detector
with mp_face_mesh.FaceMesh(static_image_mode=False,
                           max_num_faces=1,
                           refine_landmarks=True,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw all landmarks
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                )

                # Extract static landmarks
                coords = {}
                for name, idx in LANDMARK_INDICES.items():
                    lm = face_landmarks.landmark[idx]
                    coords[name] = np.array([lm.x, lm.y, lm.z])
                    color = LANDMARK_COLORS[name]
                    cv2.circle(frame, tuple([int(coords[name][0] * w), int(coords[name][1] * h)]), 5, color, -1)
                    cv2.putText(frame, name, (int(coords[name][0] * w) + 5, int(coords[name][1] * h) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Calculate center of the face
                center = np.mean([coords["Top"], coords["Bottom"], coords["Left"], coords["Right"]], axis=0)

                # Scale the center to pixel coordinates
                center_int = (int(center[0] * w), int(center[1] * h))  # Only scale x and y for 2D positioning

                # Calculate vectors from center
                vertical_vec = coords["Top"] - coords["Bottom"]
                horizontal_vec = coords["Right"] - coords["Left"]

                vertical_vec = vertical_vec / np.linalg.norm(vertical_vec)
                horizontal_vec = horizontal_vec / np.linalg.norm(horizontal_vec)

                # Calculate cross products for direction vectors
                # Ensure that the cross product is not too small
                z_vec = np.cross(horizontal_vec, vertical_vec)
                z_vec = z_vec / np.linalg.norm(z_vec)

                x_vec = -np.cross(vertical_vec, z_vec)
                x_vec = x_vec / np.linalg.norm(x_vec)
                y_vec = vertical_vec

                # Limit the scale
                scale = np.min([100, np.linalg.norm(z_vec) * 10])  # Make sure the scale isn't too large

                # Draw lines from center along x, y, z directions
                cv2.line(frame, center_int, tuple((center + x_vec * scale)[:2].astype(int) * np.array([w, h])), (255, 0, 0), 2)
                cv2.line(frame, center_int, tuple((center + y_vec * scale)[:2].astype(int) * np.array([w, h])), (0, 255, 0), 2)
                cv2.line(frame, center_int, tuple((center + z_vec * scale)[:2].astype(int) * np.array([w, h])), (0, 0, 255), 2)

                center = np.append(center, 1)
                x_vec = np.append(x_vec, 0)
                y_vec = np.append(y_vec, 0)
                z_vec = np.append(z_vec, 0)

                A = np.array([x_vec, y_vec, z_vec, center]).T
                A_inv = np.linalg.inv(A) 
                print(A)
                print(A_inv)
                print()

                for lm in face_landmarks.landmark:
                    point = np.array([lm.x, lm.y, lm.z, 1])
                    transformed = A_inv @ point
                    transformed = transformed[:3]
                    

                # Calculate and display angles between the direction vectors
                # angle_x_y = angle_between(x_vec, y_vec)
                # angle_x_z = angle_between(x_vec, z_vec)
                # angle_y_z = angle_between(y_vec, z_vec)

                # Display angles
                # cv2.putText(frame, f"Angle X-Y: {angle_x_y:.2f} deg", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                # cv2.putText(frame, f"Angle X-Z: {angle_x_z:.2f} deg", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                # cv2.putText(frame, f"Angle Y-Z: {angle_y_z:.2f} deg", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow('FaceMesh with Vectors and Angles', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
            break

cap.release()
cv2.destroyAllWindows()

