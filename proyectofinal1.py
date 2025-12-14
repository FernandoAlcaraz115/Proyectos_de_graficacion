import cv2
import numpy as np

# --- 1. CONFIGURACIÓN INICIAL ---
# Rango HSV para el color del objeto (Rojo Brillante)
# Nota: El rojo abarca dos extremos del espectro HSV, por lo que usamos dos rangos para asegurar la detección
# Aquí solo usaremos un rango simplificado para el rojo oscuro/cercano a 0:
RANGO_COLOR_INFERIOR = np.array([0, 100, 100])  # Tono 0 (Rojo), Saturación y Valor altos
RANGO_COLOR_SUPERIOR = np.array([10, 255, 255]) # Tono hasta 10, Saturación y Valor máximos

# Para una detección de rojo más robusta, a menudo se usa un segundo rango (tonos altos, cerca de 180):
# RANGO_COLOR_INFERIOR_2 = np.array([170, 100, 100])
# RANGO_COLOR_SUPERIOR_2 = np.array([180, 255, 255])
# Pero para simplicidad, trabajaremos con el rango inferior por ahora.

# Variables de estado del programa
modo_dibujo = "TRAZADO" 
x_anterior, y_anterior = 0, 0 
lienzo = None # Se inicializa en None para coincidir con el tamaño real de la cámara

# Inicializar la cámara
cap = cv2.VideoCapture(0)
ANCHO_DESEADO = 800
ALTO_DESEADO = 600
cap.set(cv2.CAP_PROP_FRAME_WIDTH, ANCHO_DESEADO)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO_DESEADO)

print("--- Pizarra Virtual Iniciada (Rastreo Rojo) ---")
print("Presiona 'M' para cambiar de modo (TRAZADO / FIGURAS)")
print("Presiona 'C' para limpiar el lienzo")
print("Presiona 'Q' para salir")

# Función para encontrar el centroide del objeto de color (misma que antes)
def encontrar_centroide(mask):
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centro_x, centro_y = -1, -1
    if contornos:
        contorno_mas_grande = max(contornos, key=cv2.contourArea)
        if cv2.contourArea(contorno_mas_grande) > 500:
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

    # 1. Ajuste de Dimensiones (Solución al error anterior)
    if lienzo is None:
        ALTO_REAL, ANCHO_REAL, _ = frame.shape
        lienzo = np.zeros((ALTO_REAL, ANCHO_REAL, 3), dtype=np.uint8)
        print(f"Dimensiones de la cámara detectadas: {ANCHO_REAL}x{ALTO_REAL}")
        
    # 2. Pre-procesamiento y Segmentación
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Aplicar la umbralización para el color Rojo
    mask = cv2.inRange(hsv, RANGO_COLOR_INFERIOR, RANGO_COLOR_SUPERIOR)
    
    # (Opcional) Si usas dos rangos (Rojo en tonos altos y bajos), combina las máscaras:
    # mask2 = cv2.inRange(hsv, RANGO_COLOR_INFERIOR_2, RANGO_COLOR_SUPERIOR_2)
    # mask = cv2.bitwise_or(mask1, mask2)
    
    # Operaciones morfológicas para limpieza
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    # 3. Localización del Landmark
    centro_x, centro_y, contorno = encontrar_centroide(mask)
    
    # Combinación del frame y el lienzo
    frame_con_lienzo = cv2.addWeighted(frame, 0.7, lienzo, 0.3, 0)
    
    # 4. Lógica de Dibujo
    if contorno is not None:
        
        # Dibujar el contorno del objeto y el centroide en el frame en vivo
        cv2.drawContours(frame_con_lienzo, [contorno], -1, (0, 255, 255), 2)
        cv2.circle(frame_con_lienzo, (centro_x, centro_y), 10, (0, 255, 255), -1) 
        
        if modo_dibujo == "TRAZADO":
            if x_anterior != 0 and y_anterior != 0:
                # El trazo de la línea se dibuja en ROJO (0, 0, 255)
                cv2.line(lienzo, (x_anterior, y_anterior), (centro_x, centro_y), (0, 0, 255), 5) 
            
            x_anterior, y_anterior = centro_x, centro_y
            
        elif modo_dibujo == "FIGURAS":
            
            x_anterior, y_anterior = 0, 0
            
            area = cv2.contourArea(contorno)
            escala = int(np.sqrt(area) * 0.5) 
            radio_circulo = max(10, escala)
            lado_rectangulo = max(10, escala * 2)
            
            # Las primitivas también se pueden hacer rojas si se desea, pero las mantendré diferentes para claridad
            cv2.circle(frame_con_lienzo, (centro_x, centro_y), radio_circulo, (255, 0, 0), 2) 
            cv2.rectangle(frame_con_lienzo, 
                          (centro_x - lado_rectangulo//2, centro_y - lado_rectangulo//2),
                          (centro_x + lado_rectangulo//2, centro_y + lado_rectangulo//2),
                          (0, 255, 0), 2) 
            
    else:
        x_anterior, y_anterior = 0, 0
        
    # 5. Mostrar y Eventos de Teclado
    cv2.putText(frame_con_lienzo, f"MODO: {modo_dibujo}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("Pizarra Virtual", frame_con_lienzo)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    
    elif key == ord('m'):
        if modo_dibujo == "TRAZADO":
            modo_dibujo = "FIGURAS"
            print("Cambiado a MODO: FIGURAS PRIMITIVAS")
        else:
            modo_dibujo = "TRAZADO"
            print("Cambiado a MODO: TRAZADO LIBRE")
            
    elif key == ord('c'):
        lienzo = np.zeros(lienzo.shape, dtype=np.uint8)
        print("Lienzo limpiado.")


# --- 6. Finalización ---
cap.release()
cv2.destroyAllWindows()