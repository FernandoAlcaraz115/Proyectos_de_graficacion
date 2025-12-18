import cv2
import mediapipe as mp
import numpy as np
import time
import random

# --- 1. CONFIGURACIÓN ---
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# Configuración de la malla facial para que se vea "Tech"
estilo_conexiones = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 255)) # Cian
estilo_puntos = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 255))

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True, # IMPORTANTE: Nos da puntos más detallados en ojos/iris
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Variables para animación del escáner
scan_y = 0
scan_direction = 1
scan_speed = 5

ANCHO_CAMARA = 800
ALTO_CAMARA = 600

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, ANCHO_CAMARA)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO_CAMARA)

# Función para generar texto aleatorio 
def generar_texto_datos():
    return f"CPU: {random.randint(20, 90)}%  MEM: {random.randint(1024, 4096)}MB  TGT: {random.randint(100, 999)}"

print("Filtro HUD Cyberpunk Iniciado.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    # 1. Preparación
    image = cv2.flip(image, 1)
    # Oscurecer un poco la imagen original para que los gráficos resalten (Efecto Cine)
    image = cv2.convertScaleAbs(image, alpha=0.8, beta=0) 
    
    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    
    # Capa para dibujar los gráficos (overlay)
    overlay = image.copy()

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        # --- MEJORA 1: DIBUJAR LA MALLA FACIAL (WIREFRAME) ---
        # Esto le da el aspecto complejo inmediato
        mp_drawing.draw_landmarks(
            image=overlay,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 100, 0), thickness=1, circle_radius=0)
        )
        
        # --- MEJORA 2: OJOS DE ROBOT (IRIS) ---
        # Usamos los landmarks refinados del iris (468-472 izquierda, 473-477 derecha)
        p_ojo_izq = face_landmarks.landmark[468]
        p_ojo_der = face_landmarks.landmark[473]
        
        cx_izq, cy_izq = int(p_ojo_izq.x * w), int(p_ojo_izq.y * h)
        cx_der, cy_der = int(p_ojo_der.x * w), int(p_ojo_der.y * h)
        
        # Efecto "Glow" (Círculos concéntricos)
        for radio in range(1, 15, 3):
            alpha = 1 - (radio / 20)
            color = (0, 255, 255) # Amarillo/Cian
            cv2.circle(overlay, (cx_izq, cy_izq), radio, color, 1)
            cv2.circle(overlay, (cx_der, cy_der), radio, color, 1)
            
        # Línea de conexión entre ojos (Targeting system)
        cv2.line(overlay, (cx_izq + 20, cy_izq), (cx_der - 20, cy_der), (0, 255, 255), 1)

        # --- MEJORA 3: BARRA DE ESCANEO VERTICAL ---
        # Calculamos los límites de la cara para el escaneo
        top_head = int(face_landmarks.landmark[10].y * h)
        chin = int(face_landmarks.landmark[152].y * h)
        
        # Inicializar escáner si está fuera de rango
        if scan_y < top_head or scan_y > chin:
            scan_y = top_head
            
        # Mover escáner
        scan_y += scan_speed * scan_direction
        if scan_y > chin: scan_direction = -1
        elif scan_y < top_head: scan_direction = 1
        
        # Dibujar línea de escaneo
        cv2.line(overlay, (int(face_landmarks.landmark[234].x * w) - 50, scan_y), 
                          (int(face_landmarks.landmark[454].x * w) + 50, scan_y), (0, 0, 255), 2)
        
        # --- MEJORA 4: ANIMACIÓN DE MANDÍBULA (Interfaz lateral) ---
        # Detectar apertura de boca
        boca_sup = face_landmarks.landmark[13].y * h
        boca_inf = face_landmarks.landmark[14].y * h
        apertura = boca_inf - boca_sup
        
        # Si abre la boca, mostrar alerta
        if apertura > 20: # Umbral de boca abierta
            cv2.putText(overlay, "WARNING: MOUTH OPEN", (cx_izq - 50, chin + 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Dibujar recuadro alrededor de la boca
            cv2.rectangle(overlay, (int(face_landmarks.landmark[61].x*w), int(boca_sup)-10),
                                   (int(face_landmarks.landmark[291].x*w), int(boca_inf)+10), (0,0,255), 2)

        # --- MEJORA 5: DATOS FLOTANTES (HUD) ---
        # Texto al lado del rostro que sigue el movimiento
        pos_texto_x = int(face_landmarks.landmark[454].x * w) + 20
        pos_texto_y = int(face_landmarks.landmark[454].y * h)
        
        # Solo actualizamos el texto cada ciertos frames para que sea legible
        if int(time.time() * 10) % 5 == 0:
            texto_hud = generar_texto_datos()
        else:
            # Mantener texto (truco simple, recalcular o usar variable global)
            texto_hud = f"SYS.ANALYSIS.RUNNING... {int(time.time())}"
            
        cv2.putText(overlay, "SYSTEM: ONLINE", (pos_texto_x, pos_texto_y), 
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        cv2.putText(overlay, texto_hud, (pos_texto_x, pos_texto_y + 20), 
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

    # Mezclar la capa de gráficos con la imagen original para dar efecto de transparencia (pantalla)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

    cv2.imshow('Proyecto 2: Filtro HUD Cyberpunk', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
