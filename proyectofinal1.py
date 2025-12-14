import cv2
import numpy as np

# --- 1. CONFIGURACIÓN INICIAL ---
# Rango HSV para el color del objeto (ej. Verde Brillante)
# Puedes ajustar estos valores con una herramienta como la de OpenCV o un selector HSV en línea.
# HSV para VERDE (ejemplo):
RANGO_COLOR_INFERIOR = np.array([40, 70, 70])
RANGO_COLOR_SUPERIOR = np.array([80, 255, 255])

# Dimensiones de la ventana
ANCHO_CAMARA = 800
ALTO_CAMARA = 600

# Variables de estado del programa
modo_dibujo = "TRAZADO" # Puede ser "TRAZADO" o "FIGURAS"
x_anterior, y_anterior = 0, 0 # Coordenadas del Landmark en el frame anterior
lienzo = np.zeros((ALTO_CAMARA, ANCHO_CAMARA, 3), dtype=np.uint8) # Lienzo virtual (Matriz NumPy negra)

# Inicializar la cámara
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, ANCHO_CAMARA)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO_CAMARA)

print("--- Pizarra Virtual Iniciada ---")
print("Modo inicial: TRAZADO LIBRE")
print("Presiona 'M' para cambiar de modo (TRAZADO / FIGURAS)")
print("Presiona 'C' para limpiar el lienzo")
print("Presiona 'Q' para salir")

# Función para encontrar el centroide del objeto de color
def encontrar_centroide(mask):
    # Encontrar contornos en la máscara
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Inicializar el centroide
    centro_x, centro_y = -1, -1
    
    if contornos:
        # Encontrar el contorno más grande (asumiendo que es el objeto de control)
        contorno_mas_grande = max(contornos, key=cv2.contourArea)
        
        # Filtrar por área mínima para evitar ruido
        if cv2.contourArea(contorno_mas_grande) > 500: # 500 píxeles es un buen umbral
            
            # Calcular los momentos para encontrar el centroide (Landmark)
            M = cv2.moments(contorno_mas_grande)
            if M["m00"] > 0:
                centro_x = int(M["m10"] / M["m00"])
                centro_y = int(M["m01"] / M["m00"])
                return (centro_x, centro_y, contorno_mas_grande)
    
    return (centro_x, centro_y, None)


# --- 2. BUCLE PRINCIPAL ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Pre-procesamiento
    frame = cv2.flip(frame, 1)
    
    # Convertir a HSV para la segmentación de color (Paso Clave)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Aplicar la umbralización (Color Tracking)
    mask = cv2.inRange(hsv, RANGO_COLOR_INFERIOR, RANGO_COLOR_SUPERIOR)
    
    # Operaciones morfológicas para reducir ruido y cerrar agujeros
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    # 2. Localización del Landmark
    centro_x, centro_y, contorno = encontrar_centroide(mask)
    
    # Crear una imagen combinada (frame + lienzo)
    # Convertir el lienzo a flotante (para la mezcla) y luego a uint8
    frame_con_lienzo = cv2.addWeighted(frame, 0.7, lienzo, 0.3, 0)
    
    # 3. Lógica de Dibujo
    if contorno is not None:
        
        # Dibujar el contorno del objeto y el centroide en el frame en vivo
        cv2.drawContours(frame_con_lienzo, [contorno], -1, (0, 255, 255), 2)
        cv2.circle(frame_con_lienzo, (centro_x, centro_y), 10, (0, 255, 255), -1) # Landmark amarillo
        
        # --- MODO TRAZADO LIBRE (PINTURA) ---
        if modo_dibujo == "TRAZADO":
            if x_anterior != 0 and y_anterior != 0:
                # Dibujar una línea entre el punto anterior y el actual en el lienzo
                cv2.line(lienzo, (x_anterior, y_anterior), (centro_x, centro_y), (0, 0, 255), 5) # Línea Roja
            
            # Actualizar la posición anterior
            x_anterior, y_anterior = centro_x, centro_y
            
        # --- MODO FIGURAS PRIMITIVAS ---
        elif modo_dibujo == "FIGURAS":
            
            # Reiniciar la posición anterior para evitar que el modo TRAZADO continúe
            x_anterior, y_anterior = 0, 0
            
            # Controlar Traslación (Posición) y Escalamiento
            
            # Escalamiento: Usar el área del contorno como magnitud para el escalamiento
            area = cv2.contourArea(contorno)
            escala = int(np.sqrt(area) * 0.5) # Factor de escala basado en el área
            radio_circulo = max(10, escala)
            lado_rectangulo = max(10, escala * 2)
            
            # Dibujar las primitivas centradas en el Landmark, pero en el frame en vivo
            # Círculo: Traslación y Escalamiento (radio)
            cv2.circle(frame_con_lienzo, (centro_x, centro_y), radio_circulo, (255, 0, 0), 2) # Azul
            
            # Rectángulo: Traslación y Escalamiento (tamaño)
            cv2.rectangle(frame_con_lienzo, 
                          (centro_x - lado_rectangulo//2, centro_y - lado_rectangulo//2),
                          (centro_x + lado_rectangulo//2, centro_y + lado_rectangulo//2),
                          (0, 255, 0), 2) # Verde

        
    # Si no se detecta el color, el modo TRAZADO debe pausarse
    else:
        x_anterior, y_anterior = 0, 0
        
    # --- 4. Mostrar y Eventos de Teclado ---
    
    # Mostrar el modo actual
    cv2.putText(frame_con_lienzo, f"MODO: {modo_dibujo}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Mostrar la imagen
    cv2.imshow("Pizarra Virtual", frame_con_lienzo)
    
    # Manejar eventos de teclado
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    
    # Cambiar de Modo
    elif key == ord('m'):
        if modo_dibujo == "TRAZADO":
            modo_dibujo = "FIGURAS"
            print("Cambiado a MODO: FIGURAS PRIMITIVAS")
        else:
            modo_dibujo = "TRAZADO"
            print("Cambiado a MODO: TRAZADO LIBRE")
            
    # Limpiar el lienzo
    elif key == ord('c'):
        lienzo = np.zeros((ALTO_CAMARA, ANCHO_CAMARA, 3), dtype=np.uint8)
        print("Lienzo limpiado.")


# --- 5. Finalización ---
cap.release()
cv2.destroyAllWindows()