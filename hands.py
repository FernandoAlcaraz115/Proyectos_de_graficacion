import cv2
import mediapipe as mp
import numpy as np

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Necesitamos detectar dos manos
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.5
)
# Inicializar la utilidad para dibujar
mp_drawing = mp.solutions.drawing_utils
# Definir el estilo de dibujo para las conexiones (esqueleto)
DRAWING_SPEC = mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2) # Amarillo/Cyan

# Inicializar la cámara
cap = cv2.VideoCapture(0) # 0 para la cámara predeterminada

# Variables para almacenar los puntos
mano_izquierda_punto = None
mano_derecha_punto = None

print("Programa iniciado. Muestra tus dos manos para ver la línea de distancia y el esqueleto de las manos.")
print("Presiona 'q' para salir.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("No se pudo acceder a la cámara.")
        break

    # 1. Preparación de la Imagen
    image = cv2.flip(image, 1)
    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 2. Procesamiento de Manos
    results = hands.process(image_rgb)

    # Reiniciar puntos en cada frame
    mano_izquierda_punto = None
    mano_derecha_punto = None

    if results.multi_hand_landmarks and results.multi_handedness:
        # Iterar sobre las manos detectadas
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Obtener la lateralidad (izquierda o derecha) de la mano
            handedness = results.multi_handedness[i].classification[0].label
            
            # **AÑADIDO: Dibujar el "esqueleto" (conexiones de los landmarks)**
            mp_drawing.draw_landmarks(
                image, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                DRAWING_SPEC,  # Estilo para los landmarks
                DRAWING_SPEC   # Estilo para las conexiones
            )
            
            # Usaremos el nudillo del dedo medio (punto 9) como referencia central.
            center_point = mp_hands.HandLandmark.MIDDLE_FINGER_MCP # Punto 9

            # Obtener coordenadas de píxeles para el punto central
            x_norm = hand_landmarks.landmark[center_point].x
            y_norm = hand_landmarks.landmark[center_point].y
            
            x_pixel = int(x_norm * w)
            y_pixel = int(y_norm * h)
            
            # Opcional: dibujar un círculo en el punto de referencia
            cv2.circle(image, (x_pixel, y_pixel), 8, (255, 0, 255), -1) # Magenta

            # 3. Asignar Puntos según Lateralidad
            if handedness == 'Left':
                mano_izquierda_punto = (x_pixel, y_pixel)
            elif handedness == 'Right':
                mano_derecha_punto = (x_pixel, y_pixel)

    # 4. Dibujo de la Línea y Cálculo de Distancia (Lógica principal)
    if mano_izquierda_punto and mano_derecha_punto:
        p1 = mano_izquierda_punto
        p2 = mano_derecha_punto

        # Dibujar la línea verde
        cv2.line(image, p1, p2, (0, 255, 0), 5) # Verde, grosor 5

        # Calcular la distancia euclidiana
        distancia_pixeles = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        # Mostrar la distancia
        mid_x = int((p1[0] + p2[0]) / 2)
        mid_y = int((p1[1] + p2[1]) / 2)
        
        texto_distancia = f"{int(distancia_pixeles)} px"
        cv2.putText(image, texto_distancia, (mid_x, mid_y - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2) # Texto amarillo

    # 5. Mostrar la Imagen
    cv2.imshow('Detector de Distancia entre Manos con Esqueleto', image)

    # Salir con la tecla 'q'
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()