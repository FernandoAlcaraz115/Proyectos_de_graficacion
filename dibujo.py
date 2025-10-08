import cv2
import numpy as np

# Colores (BGR) oscuros para la barra
colors = [
    (128, 0, 128),   # Morado oscuro
    (0, 0, 128),     # Rojo oscuro
    (0, 128, 0),     # Verde oscuro
    (0, 128, 128),   # Amarillo mostaza
    (128, 0, 0),     # Azul marino
    (0, 0, 0)        # Negro
]

canvas = None
drawing_color = (0, 0, 0)  
prev_x, prev_y = 0, 0

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.ones_like(frame) * 255

    # Barra de colores arriba
    bar_height = 80
    step = w // len(colors)
    for i, color in enumerate(colors):
        cv2.rectangle(frame, (i * step, 0), ((i + 1) * step, bar_height), color, -1)

    # Convertir a HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Rango para color azul (ajústalo si no detecta bien tu lapicero)
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Reducir ruido
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Detectar contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 800:  # Filtrar ruido
            (x, y, w_box, h_box) = cv2.boundingRect(c)
            center = (x + w_box // 2, y + h_box // 2)

            # Mostrar cursor (circulito verde en la punta detectada)
            cv2.circle(frame, center, 10, (0, 255, 0), 2)

            # Selección de color en la barra
            if center[1] < bar_height:
                selected_index = int(center[0] // step)
                drawing_color = colors[selected_index]
                prev_x, prev_y = 0, 0
            else:
                # Dibujo en el canvas
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = center
                cv2.line(canvas, (prev_x, prev_y), center, drawing_color, 5)
                prev_x, prev_y = center
        else:
            prev_x, prev_y = 0, 0
    else:
        prev_x, prev_y = 0, 0

    # Combinar cámara y canvas
    frame_out = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)

    cv2.imshow("Dibuja con el lapicero azul", frame_out)
    cv2.imshow("Mascara azul", mask)  # Ventana de depuración

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):  # Limpiar
        canvas = np.ones_like(frame) * 255
    elif key == 27:  # Esc para salir
        break

cap.release()
cv2.destroyAllWindows()


