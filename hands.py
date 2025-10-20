import cv2
import mediapipe as mp
import numpy as np

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Definir el estilo de dibujo para el esqueleto de las manos
DRAWING_SPEC = mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2) # Amarillo/Cyan

# Colores para los puntos de referencia
COLOR_MANO_IZQUIERDA = (255, 0, 0)  # Azul (BGR)
COLOR_MANO_DERECHA = (0, 0, 255)   # Rojo (BGR)
COLOR_LINEA_VERDE = (0, 255, 0)    # Verde (BGR)
GROSOR_LINEA = 5
RADIO_PUNTO_REFERENCIA = 10

# Umbral para detectar si el dedo índice está "levantado"
# Compara la altura de la punta del índice con la articulación MCP del mismo dedo.
# Si la punta está significativamente más alta, se considera levantado.
INDEX_FINGER_UP_THRESHOLD = 0.05 # Ajusta este valor si es necesario (0.0 a 1.0)

# Inicializar la cámara
cap = cv2.VideoCapture(0)

# Variables para almacenar los puntos de referencia para la línea
punto_mano_izquierda_actual = None
punto_mano_derecha_actual = None

# Variables para el modo "cuerda" (dedos índices levantados)
modo_cuerda_activo = False
punto_cuerda_izquierda = None
punto_cuerda_derecha = None


print("Programa iniciado. Muestra tus dos manos.")
print("Para ver la distancia entre manos, simplemente muéstralas.")
print("Para el 'modo cuerda', levanta solo los dedos índices de ambas manos.")
print("Presiona 'q' para salir.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("No se pudo acceder a la cámara.")
        break

    image = cv2.flip(image, 1) # Efecto espejo
    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)

    # Reiniciar puntos en cada frame
    punto_mano_izquierda_actual = None
    punto_mano_derecha_actual = None
    
    # Reiniciar el estado del modo cuerda
    modo_cuerda_candidato = True # Asumimos que podemos entrar en modo cuerda, a menos que falle la detección

    if results.multi_hand_landmarks and results.multi_handedness:
        # Diccionario para almacenar los landmarks por lateralidad
        detected_hands = {'Left': None, 'Right': None}

        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[i].classification[0].label
            detected_hands[handedness] = hand_landmarks

            # Dibujar el esqueleto de cada mano
            mp_drawing.draw_landmarks(
                image, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                DRAWING_SPEC, 
                DRAWING_SPEC
            )
            
            # Lógica para detectar si el dedo índice está "levantado"
            # Un dedo está levantado si su punta (TIP) está por encima de su MCP (Metacarpophalangeal)
            index_finger_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
            index_finger_mcp_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y

            # Si la punta del índice NO está significativamente más alta que el nudillo,
            # o si hay otros dedos que no son el índice que estén levantados,
            # entonces no estamos en modo cuerda.
            if not (index_finger_tip_y < index_finger_mcp_y - INDEX_FINGER_UP_THRESHOLD):
                modo_cuerda_candidato = False
            
            # Verificar que solo el índice esté levantado (simplificación: otros dedos no están más arriba que sus nudillos)
            # Esto es una simplificación, se podría hacer más robusto verificando la posición relativa de todos los dedos
            
            # Pulgar
            thumb_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
            thumb_mcp_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y
            if thumb_tip_y < thumb_mcp_y - INDEX_FINGER_UP_THRESHOLD:
                modo_cuerda_candidato = False
            
            # Medio
            middle_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
            middle_mcp_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
            if middle_tip_y < middle_mcp_y - INDEX_FINGER_UP_THRESHOLD:
                modo_cuerda_candidato = False

            # Anular
            ring_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
            ring_mcp_y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y
            if ring_tip_y < ring_mcp_y - INDEX_FINGER_UP_THRESHOLD:
                modo_cuerda_candidato = False

            # Meñique
            pinky_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y
            pinky_mcp_y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y
            if pinky_tip_y < pinky_mcp_y - INDEX_FINGER_UP_THRESHOLD:
                modo_cuerda_candidato = False
                

        # Verificar si ambas manos fueron detectadas para proceder con la lógica
        if detected_hands['Left'] and detected_hands['Right']:
            
            # Determinar si el modo cuerda está activo
            if modo_cuerda_candidato:
                modo_cuerda_activo = True
            else:
                modo_cuerda_activo = False

            if modo_cuerda_activo:
                # MODO CUERDA: Usar puntas de dedos índices
                index_left_x = int(detected_hands['Left'].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w)
                index_left_y = int(detected_hands['Left'].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h)
                punto_mano_izquierda_actual = (index_left_x, index_left_y)

                index_right_x = int(detected_hands['Right'].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w)
                index_right_y = int(detected_hands['Right'].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h)
                punto_mano_derecha_actual = (index_right_x, index_right_y)
                
                texto_modo = "MODO CUERDA"
                color_modo_texto = (0, 255, 255) # Amarillo
            else:
                # MODO DISTANCIA NORMAL: Usar nudillos del medio
                mcp_left_x = int(detected_hands['Left'].landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * w)
                mcp_left_y = int(detected_hands['Left'].landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * h)
                punto_mano_izquierda_actual = (mcp_left_x, mcp_left_y)

                mcp_right_x = int(detected_hands['Right'].landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * w)
                mcp_right_y = int(detected_hands['Right'].landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * h)
                punto_mano_derecha_actual = (mcp_right_x, mcp_right_y)
                
                texto_modo = "MODO DISTANCIA"
                color_modo_texto = (255, 255, 255) # Blanco
            
            # Dibujar los puntos de referencia coloreados
            cv2.circle(image, punto_mano_izquierda_actual, RADIO_PUNTO_REFERENCIA, COLOR_MANO_IZQUIERDA, -1)
            cv2.circle(image, punto_mano_derecha_actual, RADIO_PUNTO_REFERENCIA, COLOR_MANO_DERECHA, -1)

            # Dibujar la línea verde
            cv2.line(image, punto_mano_izquierda_actual, punto_mano_derecha_actual, COLOR_LINEA_VERDE, GROSOR_LINEA)

            # Calcular y mostrar la distancia
            distancia_pixeles = np.sqrt(
                (punto_mano_izquierda_actual[0] - punto_mano_derecha_actual[0])**2 +
                (punto_mano_izquierda_actual[1] - punto_mano_derecha_actual[1])**2
            )
            
            mid_x = int((punto_mano_izquierda_actual[0] + punto_mano_derecha_actual[0]) / 2)
            mid_y = int((punto_mano_izquierda_actual[1] + punto_mano_derecha_actual[1]) / 2)
            
            texto_distancia = f"{int(distancia_pixeles)} px"
            cv2.putText(image, texto_distancia, (mid_x - 50, mid_y - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2) # Texto amarillo

            # Mostrar el modo actual
            cv2.putText(image, texto_modo, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_modo_texto, 2)

        else:
            # Si no hay 2 manos detectadas, el modo cuerda no puede estar activo.
            modo_cuerda_activo = False
            cv2.putText(image, "Esperando 2 manos...", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    else:
        # Si no hay manos detectadas, el modo cuerda no puede estar activo.
        modo_cuerda_activo = False
        cv2.putText(image, "Esperando manos...", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


    cv2.imshow('Deteccion de Manos Interactiva', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()