import cv2
import mediapipe as mp
import math
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

def distancia(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

cap = cv2.VideoCapture(0)

# Variables de referencia para el zoom
base_face_width = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # --- PUNTOS CLAVE PARA ESCALA (acercamiento/alejamiento) ---
            left_cheek = face_landmarks.landmark[234]   # mejilla izquierda
            right_cheek = face_landmarks.landmark[454]  # mejilla derecha
            face_width = distancia(left_cheek, right_cheek)

            if base_face_width is None:
                base_face_width = face_width  # referencia inicial

            zoom_factor = face_width / base_face_width  # >1 si te acercas, <1 si te alejas

            # --- OJOS ---
            left_top = face_landmarks.landmark[159]
            left_bottom = face_landmarks.landmark[145]
            right_top = face_landmarks.landmark[386]
            right_bottom = face_landmarks.landmark[374]

            left_center = (
                int(face_landmarks.landmark[468].x * w),
                int(face_landmarks.landmark[468].y * h)
            )
            right_center = (
                int(face_landmarks.landmark[473].x * w),
                int(face_landmarks.landmark[473].y * h)
            )

            eye_left_dist = distancia(left_top, left_bottom)
            eye_right_dist = distancia(right_top, right_bottom)

            # Normalizar apertura
            min_eye = 0.005
            max_eye = 0.025
            norm_left = np.clip((eye_left_dist - min_eye) / (max_eye - min_eye), 0, 1)
            norm_right = np.clip((eye_right_dist - min_eye) / (max_eye - min_eye), 0, 1)

            # Tamaño base según apertura + zoom
            eye_size_left = int((20 + 15 * norm_left) * zoom_factor)
            eye_size_right = int((20 + 15 * norm_right) * zoom_factor)

            # Dibujar ojos
            cv2.circle(frame, left_center, eye_size_left, (255, 255, 255), -1)
            cv2.circle(frame, right_center, eye_size_right, (255, 255, 255), -1)
            cv2.circle(frame, left_center, max(5, int(10 * norm_left * zoom_factor)), (0, 0, 0), -1)
            cv2.circle(frame, right_center, max(5, int(10 * norm_right * zoom_factor)), (0, 0, 0), -1)

            # --- BOCA ---
            upper_lip = face_landmarks.landmark[13]
            lower_lip = face_landmarks.landmark[14]
            mouth_center = (
                int(upper_lip.x * w),
                int(upper_lip.y * h)
            )

            mouth_dist = distancia(upper_lip, lower_lip)
            min_mouth = 0.01
            max_mouth = 0.08
            norm_mouth = np.clip((mouth_dist - min_mouth) / (max_mouth - min_mouth), 0, 1)

            mouth_height = int((10 + 30 * norm_mouth) * zoom_factor)
            cv2.ellipse(frame, mouth_center, (int(40 * zoom_factor), mouth_height), 0, 0, 360, (0, 0, 255), -1)

            # (opcional) texto debug
            cv2.putText(frame, f"Zoom: {zoom_factor:.2f}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Cara Animada con Zoom y Movimiento Suave", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()


