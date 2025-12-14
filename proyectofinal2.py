import cv2
import mediapipe as mp
import glfw
from OpenGL.GL import *
import math
import time

# -----------------------------
# MediaPipe
# -----------------------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------------
# GLFW
# -----------------------------
glfw.init()
window = glfw.create_window(900, 700, "Filtro Robot HUD", None, None)
glfw.make_context_current(window)

glClearColor(0.02, 0.02, 0.05, 1)
glOrtho(0, 900, 700, 0, -1, 1)

# -----------------------------
# Cámara
# -----------------------------
cap = cv2.VideoCapture(0)
start_time = time.time()

# -----------------------------
# OpenGL helpers
# -----------------------------
def circle(x, y, r, color, filled=True):
    glColor3f(*color)
    glBegin(GL_TRIANGLE_FAN if filled else GL_LINE_LOOP)
    for i in range(360):
        a = math.radians(i)
        glVertex2f(x + math.cos(a)*r, y + math.sin(a)*r)
    glEnd()

def line(x1, y1, x2, y2, color):
    glColor3f(*color)
    glBegin(GL_LINES)
    glVertex2f(x1, y1)
    glVertex2f(x2, y2)
    glEnd()

# -----------------------------
# Suavizados
# -----------------------------
smooth_scale = 1
smooth_mouth = 1

# -----------------------------
# Loop
# -----------------------------
while not glfw.window_should_close(window):
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    glClear(GL_COLOR_BUFFER_BIT)

    if result.multi_face_landmarks:
        face = result.multi_face_landmarks[0]

        def lm(i):
            p = face.landmark[i]
            return int(p.x * 900), int(p.y * 700)

        # Landmarks clave
        le = lm(33)
        re = lm(263)
        nose = lm(1)

        mouth_top = lm(13)
        mouth_bot = lm(14)
        tongue = lm(17)  # lengua

        # Escala rostro
        face_w = abs(le[0] - re[0])
        target_scale = face_w / 140
        smooth_scale += (target_scale - smooth_scale) * 0.1

        # Boca
        mouth_open = abs(mouth_top[1] - mouth_bot[1])
        smooth_mouth += ((mouth_open / 18) - smooth_mouth) * 0.2

        # Tiempo
        t = time.time() - start_time

        # -----------------------------
        # 🤖 HUD ANIMADO
        # -----------------------------
        circle(nose[0], nose[1], 130 * smooth_scale, (0, 1, 0), False)

        for i in range(0, 360, 30):
            ang = math.radians(i + t * 60)
            x1 = nose[0] + math.cos(ang) * 110 * smooth_scale
            y1 = nose[1] + math.sin(ang) * 110 * smooth_scale
            x2 = nose[0] + math.cos(ang) * 140 * smooth_scale
            y2 = nose[1] + math.sin(ang) * 140 * smooth_scale
            line(x1, y1, x2, y2, (0, 1, 1))

        # -----------------------------
        # 👀 OJOS ROBOT
        # -----------------------------
        for eye in [le, re]:
            circle(eye[0], eye[1], 14 * smooth_scale, (0, 0.8, 1))
            circle(eye[0], eye[1], 6 * smooth_scale, (0, 0, 0))

        # -----------------------------
        # 😎 MODO LÁSER
        # -----------------------------
        laser_on = mouth_open > 25

        if laser_on:
            for eye in [le, re]:
                line(
                    eye[0], eye[1],
                    eye[0], eye[1] - 300,
                    (1, 0, 0)
                )

        # -----------------------------
        # 👅 LENGUA ROBÓTICA
        # -----------------------------
        tongue_out = tongue[1] > mouth_bot[1] + 10

        if tongue_out:
            glColor3f(1, 0.3, 0.3)
            glBegin(GL_QUADS)
            glVertex2f(tongue[0] - 10, tongue[1])
            glVertex2f(tongue[0] + 10, tongue[1])
            glVertex2f(tongue[0] + 10, tongue[1] + 40)
            glVertex2f(tongue[0] - 10, tongue[1] + 40)
            glEnd()

    glfw.swap_buffers(window)
    glfw.poll_events()

cap.release()
glfw.terminate()
