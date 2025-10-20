import cv2
import mediapipe as mp
import numpy as np
import time # <-- IMPORTANTE: Librería para manejar el tiempo

# --- Configuración de MediaPipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# --- Configuración de la Interfaz y Lógica de la Calculadora ---
display = ""
operador = None
valor_a = None
estado_calculadora = "INPUT_A"

# Definición de la cuadrícula
botones_layout = [
    ['7', 1, 1], ['8', 1, 2], ['9', 1, 3], ['/', 1, 4],
    ['4', 2, 1], ['5', 2, 2], ['6', 2, 3], ['*', 2, 4],
    ['1', 3, 1], ['2', 3, 2], ['3', 3, 3], ['-', 3, 4],
    ['C', 4, 1], ['0', 4, 2], ['=', 4, 3], ['+', 4, 4],
]

# Parámetros para el dibujo
ANCHO_VENTANA, ALTO_VENTANA = 800, 600
TAMANO_CELDA = 100
POS_X_INICIO = (ANCHO_VENTANA - 4 * TAMANO_CELDA) // 2
POS_Y_INICIO = 150 

# --- Variables de Interacción por Toque Sostenido (Hover-and-Hold) ---
TIEMPO_SELECCION_REQUERIDO = 2.0 # Segundos requeridos para seleccionar
tiempo_inicio_hover = None      # Guarda el tiempo cuando el hover comenzó
boton_hover_actual = None       # Guarda el botón que está siendo seleccionado
boton_seleccionado_en_frame = None # Etiqueta del botón seleccionado en el frame actual

# Función para realizar los cálculos (igual que antes)
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

# Función de Lógica de la Calculadora (para manejar el estado)
def procesar_seleccion(etiqueta, display, valor_a, operador, estado_calculadora):
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
            try:
                valor_a = float(resultado) # Prepara para operación continua
            except ValueError:
                valor_a = None
            estado_calculadora = "OPERATOR"
            operador = None
    elif etiqueta == 'C':
        display = ""
        operador = None
        valor_a = None
        estado_calculadora = "INPUT_A"
    
    return display, valor_a, operador, estado_calculadora

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
    
    dedo_indice_actual = None 
    boton_hover_encontrado = None # El botón que está *actualmente* bajo el dedo

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Obtener coordenadas de la punta del dedo índice (Landmark 8)
        h, w, _ = image.shape
        x_norm = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
        y_norm = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
        
        x_pixel = int(x_norm * w)
        y_pixel = int(y_norm * h)
        dedo_indice_actual = (x_pixel, y_pixel)

        # 3. Lógica de Toque Sostenido (Hover-and-Hold)
        
        # Identificar qué botón está bajo el dedo
        for etiqueta, fila, columna in botones_layout:
            bx1 = POS_X_INICIO + (columna - 1) * TAMANO_CELDA
            by1 = POS_Y_INICIO + (fila - 1) * TAMANO_CELDA
            bx2 = bx1 + TAMANO_CELDA
            by2 = by1 + TAMANO_CELDA
            
            if bx1 < x_pixel < bx2 and by1 < y_pixel < by2:
                boton_hover_encontrado = etiqueta
                break
        
        # Actualizar el estado del temporizador
        if boton_hover_encontrado:
            if boton_hover_encontrado != boton_hover_actual:
                # El dedo se movió a un nuevo botón: Reiniciar el temporizador
                boton_hover_actual = boton_hover_encontrado
                tiempo_inicio_hover = time.time()
                boton_seleccionado_en_frame = None # Prevenir doble selección
            else:
                # El dedo sigue en el mismo botón: Comprobar si se cumple el tiempo
                tiempo_transcurrido = time.time() - tiempo_inicio_hover
                if tiempo_transcurrido >= TIEMPO_SELECCION_REQUERIDO and boton_seleccionado_en_frame != boton_hover_actual:
                    # ¡Tiempo cumplido! Seleccionar el botón
                    display, valor_a, operador, estado_calculadora = procesar_seleccion(
                        boton_hover_actual, display, valor_a, operador, estado_calculadora)
                    
                    boton_seleccionado_en_frame = boton_hover_actual # Marcar como seleccionado para este evento
        else:
            # El dedo no está sobre ningún botón: Reiniciar el estado
            boton_hover_actual = None
            tiempo_inicio_hover = None
            boton_seleccionado_en_frame = None

    else:
        # No se detecta la mano: Reiniciar el estado
        boton_hover_actual = None
        tiempo_inicio_hover = None
        boton_seleccionado_en_frame = None

    # --- 4. Dibujo de la Interfaz (Overlay) ---
    
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
        
        color_fondo = (30, 30, 30)
        color_texto = (200, 200, 200)
        
        # Lógica de resaltado y progreso
        if etiqueta == boton_hover_actual and tiempo_inicio_hover is not None:
            # Calcular el progreso de la selección
            tiempo_transcurrido = time.time() - tiempo_inicio_hover
            progreso = min(tiempo_transcurrido / TIEMPO_SELECCION_REQUERIDO, 1.0)
            
            # Resaltar botón (cambiar de color según el progreso)
            color_resaltado = (0, int(255 * progreso), 0) # De negro a verde brillante
            cv2.rectangle(image, (x1, y1), (x2, y2), color_resaltado, -1)
            
            # Dibujar un círculo de progreso en la esquina (feedback visual)
            centro_x = x1 + TAMANO_CELDA // 2
            centro_y = y1 + TAMANO_CELDA // 2
            radio_progreso = int(30 * progreso)
            cv2.circle(image, (x1 + 10, y1 + 10), radio_progreso, (0, 255, 255), -1) # Pequeño círculo amarillo

        elif etiqueta == boton_seleccionado_en_frame:
             # Resaltado instantáneo si fue seleccionado en este frame
             cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), -1)

        else:
            # Dibujar el botón normal
            cv2.rectangle(image, (x1, y1), (x2, y2), color_fondo, -1)
            
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2) # Borde blanco
        
        # Dibujar el texto
        (text_w, text_h), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)
        text_x = x1 + (TAMANO_CELDA - text_w) // 2
        text_y = y1 + (TAMANO_CELDA + text_h) // 2
        cv2.putText(image, etiqueta, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color_texto, 2)

    # Dibujar el punto del dedo índice
    if dedo_indice_actual:
        cv2.circle(image, dedo_indice_actual, 10, (255, 0, 255), -1) # Magenta

    # Mostrar la imagen
    cv2.imshow('Calculadora Gestual (Toque Sostenido)', image)

    # Salir con la tecla 'q'
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()