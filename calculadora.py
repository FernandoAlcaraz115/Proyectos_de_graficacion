import cv2
import mediapipe as mp
import numpy as np

# --- Configuración de MediaPipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # Solo necesitamos una mano para interactuar
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# --- Configuración de la Interfaz y Lógica de la Calculadora ---
# Variables de la calculadora
display = ""
operador = None
valor_a = None
estado_calculadora = "INPUT_A" # INPUT_A, OPERATOR, INPUT_B

# Definición de la cuadrícula de la calculadora: [etiqueta, fila, columna]
botones_layout = [
    ['7', 1, 1], ['8', 1, 2], ['9', 1, 3], ['/', 1, 4],
    ['4', 2, 1], ['5', 2, 2], ['6', 2, 3], ['*', 2, 4],
    ['1', 3, 1], ['2', 3, 2], ['3', 3, 3], ['-', 3, 4],
    ['C', 4, 1], ['0', 4, 2], ['=', 4, 3], ['+', 4, 4],
]

# Parámetros para el dibujo (ajustados al tamaño de la ventana)
ANCHO_VENTANA, ALTO_VENTANA = 800, 600
TAMANO_CELDA = 100 # Tamaño de cada botón

# Posición de inicio de la cuadrícula (para centrarla un poco)
POS_X_INICIO = (ANCHO_VENTANA - 4 * TAMANO_CELDA) // 2
POS_Y_INICIO = 150 

# Variables de interacción gestual
indice_anterior_x, indice_anterior_y = 0, 0
SELECCION_ACTIVA = False # Bandera para saber si el "rascado" comenzó
UMBRAL_RASPADO = 20 # Distancia mínima para considerar un "rascado"

# Función para realizar los cálculos
def calcular(val_a, op, val_b):
    try:
        if op == '+': return str(val_a + val_b)
        if op == '-': return str(val_a - val_b)
        if op == '*': return str(val_a * val_b)
        if op == '/':
            if val_b == 0: return "Error: /0"
            return str(val_a / val_b)
    except:
        return "Error"

# --- Bucle Principal del Programa ---
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    # 1. Preparación de la Imagen
    image = cv2.flip(image, 1)
    image = cv2.resize(image, (ANCHO_VENTANA, ALTO_VENTANA))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 2. Procesamiento de Manos
    results = hands.process(image_rgb)
    
    # Reiniciar la selección si no se detecta el dedo
    dedo_indice_actual = None 

    if results.multi_hand_landmarks:
        # Solo procesamos la primera mano detectada
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Obtener las coordenadas de la punta del dedo índice (Landmark 8)
        h, w, _ = image.shape
        x_norm = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
        y_norm = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
        
        x_pixel = int(x_norm * w)
        y_pixel = int(y_norm * h)
        dedo_indice_actual = (x_pixel, y_pixel)

        # 3. Lógica de Interacción (Rascado/Selección)
        if SELECCION_ACTIVA:
            # Si ya estábamos rascando, verificamos si el dedo se movió lo suficiente
            dx = abs(x_pixel - indice_anterior_x)
            dy = abs(y_pixel - indice_anterior_y)
            distancia_movida = np.sqrt(dx**2 + dy**2)
            
            if distancia_movida > UMBRAL_RASPADO:
                # El "rascado" se ha completado, y la selección es exitosa
                
                # 4. Encontrar el Botón Seleccionado
                for etiqueta, fila, columna in botones_layout:
                    bx1 = POS_X_INICIO + (columna - 1) * TAMANO_CELDA
                    by1 = POS_Y_INICIO + (fila - 1) * TAMANO_CELDA
                    bx2 = bx1 + TAMANO_CELDA
                    by2 = by1 + TAMANO_CELDA
                    
                    # Verificamos si la posición del dedo anterior o actual está dentro del botón
                    if (bx1 < indice_anterior_x < bx2 and by1 < indice_anterior_y < by2) or \
                       (bx1 < x_pixel < bx2 and by1 < y_pixel < by2):
                        
                        # Botón Seleccionado: Lógica de la Calculadora
                        if etiqueta.isdigit():
                            if estado_calculadora in ["INPUT_A", "INPUT_B"]:
                                display = display + etiqueta
                        elif etiqueta in ['+', '-', '*', '/']:
                            if estado_calculadora == "INPUT_A" and display:
                                valor_a = float(display)
                                operador = etiqueta
                                display = ""
                                estado_calculadora = "INPUT_B"
                        elif etiqueta == '=':
                            if estado_calculadora == "INPUT_B" and display:
                                valor_b = float(display)
                                resultado = calcular(valor_a, operador, valor_b)
                                display = resultado
                                valor_a = float(resultado) # Prepara para operación continua
                                estado_calculadora = "OPERATOR"
                                operador = None
                        elif etiqueta == 'C':
                            # Limpiar
                            display = ""
                            operador = None
                            valor_a = None
                            estado_calculadora = "INPUT_A"
                        
                        # Detener el rascado inmediatamente después de una selección
                        SELECCION_ACTIVA = False
                        break # Salir del bucle de botones
        
        # Si el rascado no está activo, verificamos si se debe iniciar
        elif not SELECCION_ACTIVA:
            # Actualizamos las coordenadas iniciales para el próximo rascado
            indice_anterior_x, indice_anterior_y = x_pixel, y_pixel
            
            # Verificamos si el dedo índice está sobre algún botón para empezar la SELECCIÓN
            for _, fila, columna in botones_layout:
                bx1 = POS_X_INICIO + (columna - 1) * TAMANO_CELDA
                by1 = POS_Y_INICIO + (fila - 1) * TAMANO_CELDA
                bx2 = bx1 + TAMANO_CELDA
                by2 = by1 + TAMANO_CELDA
                
                if bx1 < x_pixel < bx2 and by1 < y_pixel < by2:
                    # El dedo está sobre un botón, activamos la selección
                    SELECCION_ACTIVA = True
                    break

    # Si no se detecta la mano, se reinicia la selección
    else:
        SELECCION_ACTIVA = False


    # --- 5. Dibujo de la Interfaz (Overlay) ---
    
    # Dibujar la pantalla (Display) de la calculadora
    cv2.rectangle(image, (POS_X_INICIO, 50), (POS_X_INICIO + 4 * TAMANO_CELDA, 130), (50, 50, 50), -1)
    cv2.putText(image, display if display else "0", (POS_X_INICIO + 10, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    # Dibujar los botones
    for etiqueta, fila, columna in botones_layout:
        x1 = POS_X_INICIO + (columna - 1) * TAMANO_CELDA
        y1 = POS_Y_INICIO + (fila - 1) * TAMANO_CELDA
        x2 = x1 + TAMANO_CELDA
        y2 = y1 + TAMANO_CELDA
        
        color_fondo = (30, 30, 30) # Gris oscuro
        color_texto = (200, 200, 200) # Gris claro
        
        # Resaltar botón si el dedo está sobre él
        if dedo_indice_actual and x1 < dedo_indice_actual[0] < x2 and y1 < dedo_indice_actual[1] < y2:
             color_fondo = (0, 150, 0) # Verde para pre-selección
        
        # Dibujar el botón
        cv2.rectangle(image, (x1, y1), (x2, y2), color_fondo, -1)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2) # Borde blanco
        
        # Dibujar el texto
        (text_w, text_h), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)
        text_x = x1 + (TAMANO_CELDA - text_w) // 2
        text_y = y1 + (TAMANO_CELDA + text_h) // 2
        cv2.putText(image, etiqueta, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color_texto, 2)

    # Dibujar el punto del dedo índice para feedback visual
    if dedo_indice_actual:
        color_dedo = (0, 255, 255) if SELECCION_ACTIVA else (255, 0, 0)
        cv2.circle(image, dedo_indice_actual, 10, color_dedo, -1)

    # Mostrar la imagen
    cv2.imshow('Calculadora Gestual', image)

    # Salir con la tecla 'q'
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()