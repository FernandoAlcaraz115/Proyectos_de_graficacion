from OpenGL.GL import *
from OpenGL.GLU import gluPerspective, gluLookAt, gluNewQuadric, gluCylinder, gluSphere
import glfw
import cv2
import numpy as np
import sys
import math
import mediapipe as mp
import random

# --- CONFIGURACIÓN MEDIAPIPE ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Variables de Cámara
camera_angle_x = 0.5
camera_angle_y = 0.5 
camera_distance = 80.0 
window = None

# --- GEOMETRÍA (Tus funciones originales compactadas) ---
def init():
    glClearColor(0.53, 0.81, 0.92, 1.0) # Azul Cielo
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING); glEnable(GL_LIGHT0); glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glMatrixMode(GL_PROJECTION)
    gluPerspective(60, 1.33, 0.1, 500.0)
    glMatrixMode(GL_MODELVIEW)

def draw_cylinder():
    glPushMatrix(); glColor3f(0.6, 0.0, 0.0); glTranslatef(0.0, -1.0, 0.0); glRotatef(-90, 1, 0, 0)
    gluCylinder(gluNewQuadric(), 0.2, 0.2, 5.0, 32, 32); glPopMatrix()

def draw_board():
    glPushMatrix(); glColor3f(1.0, 1.0, 1.0); glTranslatef(0.0, 3.5, -0.5); glScalef(1.5, 1.0, 0.1)
    glBegin(GL_QUADS)
    for f in [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]: glVertex3f(*f)
    glEnd(); glPopMatrix()

def draw_hoop(): # Aro simple
    glPushMatrix(); glColor3f(1.0, 0.0, 0.0); glTranslatef(0.0, 3.0, -0.6); glRotatef(90, 1, 0, 0)
    gluCylinder(gluNewQuadric(), 0.5, 0.5, 0.1, 32, 32); glPopMatrix()

def draw_light_pole():
    glPushMatrix(); glColor3f(0.2, 0.2, 0.2); glRotatef(-90, 1, 0, 0)
    gluCylinder(gluNewQuadric(), 0.15, 0.1, 8.0, 16, 16); glPopMatrix()
    glPushMatrix(); glColor3f(1.0, 1.0, 0.0); glTranslatef(0, 8.0, 0)
    gluSphere(gluNewQuadric(), 0.5, 16, 16); glPopMatrix()

def draw_cloud():
    glPushMatrix(); glColor3f(1.0, 1.0, 1.0); q = gluNewQuadric()
    for p in [(0,0.5,0),(-0.8,0,0),(0.8,0,0),(0,0,-0.8),(0,0,0.8)]:
        glPushMatrix(); glTranslatef(*p); gluSphere(q, 1.0, 32, 32); glPopMatrix()
    glPopMatrix()

def draw_snowman():
    glPushMatrix(); glTranslatef(0, 0.0, 0.0); glRotatef(180, 0.0, 1.0, 0.0); glColor3f(1.0, 1.0, 1.0)
    gluSphere(gluNewQuadric(), 1.5, 32, 32); glTranslatef(0.0, 2.0, 0.0); gluSphere(gluNewQuadric(), 1.2, 32, 32)
    glTranslatef(0.0, 1.5, 0.0); gluSphere(gluNewQuadric(), 1.0, 32, 32)
    glPushMatrix(); glColor3f(0.0, 0.0, 0.0); glTranslatef(-0.4, 0.5, 0.8); gluSphere(gluNewQuadric(), 0.1, 16, 16)
    glTranslatef(0.8, 0.0, 0.0); gluSphere(gluNewQuadric(), 0.1, 16, 16); glPopMatrix()
    glPushMatrix(); glColor3f(1.0, 0.5, 0.0); glTranslatef(0.0, 0.5, 1.1); gluSphere(gluNewQuadric(), 0.2, 16, 16); glPopMatrix()
    glPopMatrix()

# --- ARBOLES (Estandarizados para que no sean gigantes) ---
def draw_tree_simple():
    glPushMatrix()
    # Tronco
    glColor3f(0.55, 0.27, 0.07); glRotatef(-90, 1, 0, 0)
    gluCylinder(gluNewQuadric(), 0.5, 0.5, 2.0, 16, 16)
    # Hojas
    glTranslatef(0, 0, 2.0); glColor3f(0.13, 0.55, 0.13)
    gluCylinder(gluNewQuadric(), 2.0, 0.0, 4.0, 16, 16) # Pino
    glPopMatrix()

def draw_tree_round():
    glPushMatrix()
    glColor3f(0.55, 0.27, 0.07); glRotatef(-90, 1, 0, 0)
    gluCylinder(gluNewQuadric(), 0.4, 0.4, 2.5, 16, 16)
    glTranslatef(0, 0, 2.5); glColor3f(0.2, 0.8, 0.2)
    gluSphere(gluNewQuadric(), 1.8, 16, 16)
    glPopMatrix()

# --- CASAS (Estandarizadas) ---
def draw_house_type1(): # Casa Cuadrada
    glPushMatrix()
    # Base
    glBegin(GL_QUADS); glColor3f(0.8, 0.5, 0.2) # Paredes Naranja
    for f in [(-2,0,2),(2,0,2),(2,3,2),(-2,3,2), (-2,0,-2),(2,0,-2),(2,3,-2),(-2,3,-2), 
              (-2,0,-2),(-2,0,2),(-2,3,2),(-2,3,-2), (2,0,-2),(2,0,2),(2,3,2),(2,3,-2)]: glVertex3f(*f)
    glEnd()
    # Techo
    glBegin(GL_TRIANGLES); glColor3f(0.6, 0.1, 0.1) # Techo Rojo
    for f in [(-2.2,3,2.2),(2.2,3,2.2),(0,5,0), (-2.2,3,-2.2),(2.2,3,-2.2),(0,5,0),
              (-2.2,3,-2.2),(-2.2,3,2.2),(0,5,0), (2.2,3,-2.2),(2.2,3,2.2),(0,5,0)]: glVertex3f(*f)
    glEnd()
    # Puerta
    glBegin(GL_QUADS); glColor3f(0.4, 0.2, 0.1)
    glVertex3f(-0.5,0,2.01); glVertex3f(0.5,0,2.01); glVertex3f(0.5,2,2.01); glVertex3f(-0.5,2,2.01)
    glEnd()
    glPopMatrix()

def draw_house_type2(): # Casa con Segundo Piso
    glPushMatrix()
    # Base
    glBegin(GL_QUADS); glColor3f(0.7, 0.7, 0.7) # Paredes Grises
    for f in [(-2,0,2),(2,0,2),(2,2,2),(-2,2,2), (-2,0,-2),(2,0,-2),(2,2,-2),(-2,2,-2), 
              (-2,0,-2),(-2,0,2),(-2,2,2),(-2,2,-2), (2,0,-2),(2,0,2),(2,2,2),(2,2,-2)]: glVertex3f(*f)
    glEnd()
    # Piso 2
    glBegin(GL_QUADS); glColor3f(0.8, 0.8, 0.8) 
    for f in [(-1.8,2,-1.8),(1.8,2,-1.8),(1.8,3.5,-1.8),(-1.8,3.5,-1.8), (-1.8,2,1.8),(1.8,2,1.8),(1.8,3.5,1.8),(-1.8,3.5,1.8),
              (-1.8,2,-1.8),(-1.8,2,1.8),(-1.8,3.5,1.8),(-1.8,3.5,-1.8), (1.8,2,-1.8),(1.8,2,1.8),(1.8,3.5,1.8),(1.8,3.5,-1.8)]: glVertex3f(*f)
    glEnd()
    # Ventanas Grandes
    glBegin(GL_QUADS); glColor3f(0.2, 0.6, 0.9)
    glVertex3f(-1.5,2.2,1.81); glVertex3f(-0.2,2.2,1.81); glVertex3f(-0.2,3.0,1.81); glVertex3f(-1.5,3.0,1.81)
    glVertex3f(0.2,2.2,1.81); glVertex3f(1.5,2.2,1.81); glVertex3f(1.5,3.0,1.81); glVertex3f(0.2,3.0,1.81)
    glEnd()
    glPopMatrix()

def draw_ground():
    glBegin(GL_QUADS); glColor3f(0.2, 0.2, 0.2) # Asfalto/Suelo oscuro para resaltar colores
    glVertex3f(-300,0,300); glVertex3f(300,0,300); glVertex3f(300,0,-300); glVertex3f(-300,0,-300); glEnd()
    # Pasto central
    glBegin(GL_QUADS); glColor3f(0.1, 0.5, 0.1) 
    glVertex3f(-15,0.1,15); glVertex3f(15,0.1,15); glVertex3f(15,0.1,-15); glVertex3f(-15,0.1,-15); glEnd()


# -------------------------------------------------------------------------
# DIBUJO DE ESCENA - AQUÍ ESTÁ LA MAGIA DEL PUEBLO COMPLETO
# -------------------------------------------------------------------------
def draw_scene():
    global camera_angle_x, camera_angle_y, camera_distance
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    
    # Cámara
    cam_x = camera_distance * math.sin(camera_angle_y) * math.cos(camera_angle_x)
    cam_y = camera_distance * math.cos(camera_angle_y)
    cam_z = camera_distance * math.sin(camera_angle_y) * math.sin(camera_angle_x)
    if cam_y < 2.0: cam_y = 2.0
    gluLookAt(cam_x, cam_y, cam_z, 0, 0, 0, 0, 1, 0)
    
    t = glfw.get_time()

    draw_ground()
    
    # --- 1. GENERACIÓN PROCEDURAL DE CIUDAD (GRID) ---
    # Esto crea muchas casas ordenadas sin tener que escribirlas una por una
    block_size = 15 # Tamaño de cada "cuadra"
    city_radius = 4 # Cuántas cuadras a cada lado (4x4 = 16 cuadras hacia cada lado)
    
    for x in range(-city_radius, city_radius + 1):
        for z in range(-city_radius, city_radius + 1):
            
            # ZONA DEL PARQUE CENTRAL (No dibujar casas aquí)
            if -1 <= x <= 1 and -1 <= z <= 1:
                continue

            glPushMatrix()
            glTranslatef(x * block_size, 0, z * block_size)
            
            # Decidir qué dibujar basado en la posición para variedad
            if (x + z) % 3 == 0:
                draw_house_type1()
                # Arbolito al lado de la casa
                glPushMatrix(); glTranslatef(3.5, 0, 2); glScalef(0.6, 0.6, 0.6); draw_tree_simple(); glPopMatrix()
            elif (x + z) % 3 == 1:
                glRotatef(90, 0, 1, 0)
                draw_house_type2()
                # Poste de luz en la esquina
                glPushMatrix(); glTranslatef(4, 0, 4); glScalef(0.5, 0.5, 0.5); draw_light_pole(); glPopMatrix()
            else:
                # Bosquecillo (Lote baldío con árboles)
                glPushMatrix(); glTranslatef(-2, 0, -2); draw_tree_round(); glPopMatrix()
                glPushMatrix(); glTranslatef(2, 0, 1); glScalef(1.2,1.2,1.2); draw_tree_simple(); glPopMatrix()
                glPushMatrix(); glTranslatef(0, 0, -3); glScalef(0.8,0.8,0.8); draw_tree_round(); glPopMatrix()

            glPopMatrix()

    # --- 2. EL PARQUE CENTRAL (Tus objetos especiales) ---
    
    # Muñeco de Nieve (Animado saltando)
    glPushMatrix()
    glTranslatef(0, 0, 0) # En el puro centro
    jump = abs(math.sin(t*3)) * 1.5
    glTranslatef(0, jump, 0)
    glScalef(0.8, 0.8, 0.8)
    draw_snowman()
    glPopMatrix()
    
    # Cancha de Basquet (Izquierda)
    glPushMatrix()
    glTranslatef(-8, 0, 0)
    draw_cylinder(); draw_board(); 
    # Aro animado girando
    glPushMatrix(); glRotatef(t*100, 0, 1, 0); draw_hoop(); glPopMatrix()
    glPopMatrix()
    
    # Cancha de Basquet (Derecha)
    glPushMatrix()
    glTranslatef(8, 0, 0); glRotatef(180, 0, 1, 0)
    draw_cylinder(); draw_board(); draw_hoop()
    glPopMatrix()
    
    # Algunos árboles grandes en el parque
    glPushMatrix(); glTranslatef(0, 0, -8); glScalef(1.5, 1.5, 1.5); draw_tree_round(); glPopMatrix()
    glPushMatrix(); glTranslatef(0, 0, 8); glScalef(1.5, 1.5, 1.5); draw_tree_round(); glPopMatrix()
    
    # --- 3. OBJETOS MÓVILES (NUBES) ---
    # 20 Nubes orbitando toda la ciudad
    for i in range(20):
        glPushMatrix()
        angle = t * 0.05 + (i * 0.314)
        r = 50.0 + (i % 3) * 15.0 # Distintos radios
        h = 25.0 + math.sin(t + i) * 2.0
        
        glTranslatef(math.cos(angle)*r, h, math.sin(angle)*r)
        
        # Orientar la nube hacia el centro (opcional, visualmente mejor)
        glRotatef(-math.degrees(angle), 0, 1, 0)
        
        glScalef(3, 3, 3)
        draw_cloud()
        glPopMatrix()

    glfw.swap_buffers(window)

# --- LOOP PRINCIPAL ---
def main_loop():
    global camera_angle_x, camera_angle_y, camera_distance, window
    cap = cv2.VideoCapture(0)
    
    while not glfw.window_should_close(window):
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            
            index_tip = hand.landmark[8]
            thumb_tip = hand.landmark[4]
            
            # Rotación (Índice)
            camera_angle_x = index_tip.x * 2 * math.pi
            camera_angle_y = max(0.01, min(math.pi - 0.01, index_tip.y * math.pi))
            
            # Zoom (Pinza)
            pinch = math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
            # Mapeo: 0.05 (cerca) -> Zoom 20, 0.3 (lejos) -> Zoom 150
            # Invertimos la lógica: Pellizco cerrado = Lejos, Abierto = Cerca
            target_zoom = 150 - (pinch * 400)
            camera_distance = max(20, min(150, camera_distance * 0.9 + target_zoom * 0.1))
            
            cv2.putText(frame, f"Zoom: {int(camera_distance)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        
        # Teclas WASD para mover si la mano cansa
        if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS: camera_distance -= 1
        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS: camera_distance += 1

        cv2.imshow("Control MediaPipe", frame)
        draw_scene()
        glfw.poll_events()
        if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    cap.release(); cv2.destroyAllWindows(); glfw.terminate()

def main():
    global window
    if not glfw.init(): sys.exit()
    window = glfw.create_window(1024, 768, "Ciudad 3D Generada", None, None)
    if not window: glfw.terminate(); sys.exit()
    glfw.make_context_current(window)
    glViewport(0, 0, 1024, 768)
    init()
    main_loop()

if __name__ == "__main__":
    main()