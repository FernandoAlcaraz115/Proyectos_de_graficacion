import cv2
import numpy as np
import math

# --- VARIABLES GLOBALES ---
color_seleccionado = False
hsv_min = np.array([0, 0, 0])
hsv_max = np.array([0, 0, 0])

modo_dibujo = "TRAZADO" 
x_anterior, y_anterior = 0, 0
lienzo = None

# Variables para la física del movimiento (El punto "Avanzado")
angulo_actual = 0
escala_actual = 50

# --- 1. SELECCIÓN DE COLOR (Igual que antes) ---
def seleccionar_color(event, x, y, flags, param):
    global hsv_min, hsv_max, color_seleccionado, frame_actual
    if event == cv2.EVENT_LBUTTONDOWN: 
        pixel_bgr = frame_actual[y, x]
        pixel_hsv = cv2.cvtColor(np.uint8([[pixel_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
        print(f"Color seleccionado: {pixel_hsv}")
        
        # Rango de tolerancia
        h_min = max(0, pixel_hsv[0] - 10)
        h_max = min(179, pixel_hsv[0] + 10)
        
        hsv_min = np.array([h_min, 60, 60])   
        hsv_max = np.array([h_max, 255, 255])
        color_seleccionado = True

# --- 2. INICIALIZACIÓN ---
cap = cv2.VideoCapture(0)
cap.set(3, 800); cap.set(4, 600)
cv2.namedWindow("Pizarra Fisica")
cv2.setMouseCallback("Pizarra Fisica", seleccionar_color)

print("--- PROYECTO 1: FÍSICA DE MOVIMIENTO ---")
print("1. Clic en el objeto de color.")
print("2. Presiona 'M' para ver las figuras reaccionar.")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    frame_actual = frame.copy()
    
    if lienzo is None: lienzo = np.zeros_like(frame)
    
    centro_x, centro_y = -1, -1
    contorno = None
    
    if color_seleccionado:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hsv_min, hsv_max)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contornos:
            c = max(contornos, key=cv2.contourArea)
            if cv2.contourArea(c) > 500:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    centro_x = int(M["m10"] / M["m00"])
                    centro_y = int(M["m01"] / M["m00"])
                    contorno = c

    # --- LÓGICA AVANZADA ---
    frame_final = cv2.addWeighted(frame, 0.7, lienzo, 0.3, 0)
    
    if contorno is not None:
        # Dibujar el "Landmark" (Centroide)
        cv2.circle(frame_final, (centro_x, centro_y), 5, (0, 0, 255), -1)
        
        if modo_dibujo == "TRAZADO":
            if x_anterior != 0 and y_anterior != 0:
                # Dibujar línea si no hubo un salto muy grande (evitar líneas locas)
                distancia = np.sqrt((centro_x - x_anterior)**2 + (centro_y - y_anterior)**2)
                if distancia < 100:
                    cv2.line(lienzo, (x_anterior, y_anterior), (centro_x, centro_y), (0, 255, 255), 4)
            x_anterior, y_anterior = centro_x, centro_y
            
        elif modo_dibujo == "FIGURAS":
            # Aquí ocurre la magia del vector de movimiento
            # 1. Calcular el vector delta (diferencia de posición)
            if x_anterior == 0 and y_anterior == 0:
                dx, dy = 0, 0
            else:
                dx = centro_x - x_anterior
                dy = centro_y - y_anterior
            
            # 2. CALCULAR MAGNITUD (Velocidad) -> Controla ESCALAMIENTO
            velocidad = np.sqrt(dx**2 + dy**2)
            # Tamaño base 30 + velocidad * factor
            tamaño_objetivo = 30 + int(velocidad * 3)
            # Suavizado para que no parpadee (interpolación)
            escala_actual = int(escala_actual * 0.8 + tamaño_objetivo * 0.2)
            
            # 3. CALCULAR DIRECCIÓN (Ángulo) -> Controla ROTACIÓN
            # Solo actualizamos el ángulo si hay movimiento significativo
            if velocidad > 2:
                # atan2 devuelve radianes, convertimos a grados
                angulo_rad = math.atan2(dy, dx) 
                angulo_grados = math.degrees(angulo_rad)
                angulo_actual = angulo_grados

            # 4. DIBUJAR RECTÁNGULO ROTADO (Primitiva Avanzada)
            # OpenCV necesita un formato especial para rectángulos rotados
            rect = ((centro_x, centro_y), (escala_actual, escala_actual/2), angulo_actual)
            box = cv2.boxPoints(rect) 
            box = np.int0(box) # Convertir coordenadas a enteros
            
            # Dibujar el rectángulo rotado
            cv2.drawContours(frame_final, [box], 0, (0, 255, 0), 3)
            
            # Dibujar flecha de dirección (Vector)
            cv2.arrowedLine(frame_final, (x_anterior, y_anterior), (centro_x, centro_y), (255, 0, 255), 2)
            
            # Datos en pantalla
            cv2.putText(frame_final, f"Vel: {int(velocidad)} -> Escala: {escala_actual}", (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame_final, f"Dir: {int(angulo_actual)} deg", (10, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Actualizar anterior
            x_anterior, y_anterior = centro_x, centro_y

    else:
        # Si perdemos el objeto, reseteamos las coordenadas anteriores para no dibujar líneas raras
        x_anterior, y_anterior = 0, 0
        if not color_seleccionado:
            cv2.putText(frame_final, "CLIC EN UN OBJETO PARA INICIAR", (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(frame_final, f"MODO: {modo_dibujo} (M)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("Pizarra Fisica", frame_final)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('c'): lienzo = np.zeros_like(frame)
    elif key == ord('m'): 
        modo_dibujo = "FIGURAS" if modo_dibujo == "TRAZADO" else "TRAZADO"
        x_anterior, y_anterior = 0, 0 # Reset al cambiar modo

cap.release()
cv2.destroyAllWindows()