import cv2
import numpy as np
import math

# ===============================
# CONFIGURACIÓN
# ===============================
cap = cv2.VideoCapture(0)
canvas = None

modo = "PINTURA"      # PINTURA o FIGURAS
figura = "CIRCULO"    # CIRCULO o RECTANGULO

prev_x, prev_y = None, None

# Color azul en HSV (AJUSTABLE)
lower_blue = np.array([100, 120, 70])
upper_blue = np.array([130, 255, 255])

# ===============================
# FUNCIÓN PARA OBTENER CENTROIDE
# ===============================
def obtener_landmark(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 500:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return cx, cy
    return None

# ===============================
# LOOP PRINCIPAL
# ===============================
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    landmark = obtener_landmark(mask)

    if landmark:
        x, y = landmark
        cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)

        # ===============================
        # MODO PINTURA
        # ===============================
        if modo == "PINTURA":
            if prev_x is not None:
                cv2.line(canvas, (prev_x, prev_y), (x, y), (255, 0, 0), 5)

            prev_x, prev_y = x, y

        # ===============================
        # MODO FIGURAS
        # ===============================
        elif modo == "FIGURAS":
            size = 40
            if prev_x:
                dist = int(math.hypot(x - prev_x, y - prev_y))
                size = max(20, min(120, dist))

            if figura == "CIRCULO":
                cv2.circle(canvas, (x, y), size, (0, 255, 255), 3)

            elif figura == "RECTANGULO":
                cv2.rectangle(
                    canvas,
                    (x - size, y - size),
                    (x + size, y + size),
                    (255, 0, 255),
                    3
                )

            prev_x, prev_y = x, y

    else:
        prev_x, prev_y = None, None

    # ===============================
    # COMBINAR CAMARA + CANVAS
    # ===============================
    output = cv2.addWeighted(frame, 0.7, canvas, 0.7, 0)

    cv2.putText(output, f"Modo: {modo}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(output, f"Figura: {figura}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Pizarra Virtual", output)
    cv2.imshow("Mascara", mask)

    # ===============================
    # TECLADO
    # ===============================
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break
    elif key == ord('p'):
        modo = "PINTURA"
        prev_x, prev_y = None, None
    elif key == ord('f'):
        modo = "FIGURAS"
        prev_x, prev_y = None, None
    elif key == ord('c'):
        figura = "RECTANGULO" if figura == "CIRCULO" else "CIRCULO"
    elif key == ord('x'):
        canvas = np.zeros_like(canvas)

cap.release()
cv2.destroyAllWindows()
