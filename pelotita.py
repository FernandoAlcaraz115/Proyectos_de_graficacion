# filepath: c:\Users\ferna\OneDrive\Documentos\Programas graficación\pelotita.py
import cv2
import numpy as np
import math

# Configuración de la ventana
ANCHO, ALTO = 800, 600

# Bolita móvil (roja)
radio = 20
pos_x, pos_y = ANCHO // 2, ALTO // 2
vel_x, vel_y = 5.0, 5.0

# Bolita "estática" (azul) que puede esquivar
bradio = 25
bx, by = ANCHO // 4, ALTO // 2
# variables para esquivar
is_dodging = False
dodge_steps = 0
dodge_dx = 0.0
dodge_dy = 0.0
DODGE_STEPS_TOTAL = 18
DODGE_SPEED = 8.0  # píxeles por frame al esquivar
SAFE_MARGIN = 8    # margen extra antes de considerar impacto

def clamp_ball(x, y, r):
    x = max(r, min(ANCHO - r, int(round(x))))
    y = max(r, min(ALTO - r, int(round(y))))
    return x, y

# Crear ventana de OpenCV (permite detectar el cierre por la X)
WIN_NAME = "Bolita Rebotando (cv2)"
cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)

try:
    while True:
        # Crear la imagen (fondo negro)
        imagen = np.zeros((ALTO, ANCHO, 3), dtype=np.uint8)

        # Movimiento de la bolita roja
        pos_x += vel_x
        pos_y += vel_y

        # Rebotes contra los bordes
        if pos_x + radio >= ANCHO or pos_x - radio <= 0:
            vel_x = -vel_x
            pos_x = max(radio, min(ANCHO - radio, pos_x))
        if pos_y + radio >= ALTO or pos_y - radio <= 0:
            vel_y = -vel_y
            pos_y = max(radio, min(ALTO - radio, pos_y))

        # Posición futura de la roja para prever colisión
        future_x = pos_x + vel_x
        future_y = pos_y + vel_y
        dx = future_x - bx
        dy = future_y - by
        distancia_futura = math.hypot(dx, dy)

        # Iniciar esquiva si se aproxima una colisión y la azul no está ya esquivando
        if (distancia_futura <= (radio + bradio + SAFE_MARGIN)) and not is_dodging:
            approach_x = future_x - bx
            approach_y = future_y - by
            norm = math.hypot(approach_x, approach_y) or 1.0
            # Vector perpendicular unitario para esquivar lateralmente
            perpx = -approach_y / norm
            perpy = approach_x / norm
            # Probar sentido para mantener dentro de la ventana
            test_bx = bx + perpx * DODGE_SPEED * DODGE_STEPS_TOTAL
            test_by = by + perpy * DODGE_SPEED * DODGE_STEPS_TOTAL
            if not (bradio <= test_bx <= ANCHO - bradio and bradio <= test_by <= ALTO - bradio):
                perpx *= -1
                perpy *= -1
            dodge_dx = perpx * DODGE_SPEED
            dodge_dy = perpy * DODGE_SPEED
            is_dodging = True
            dodge_steps = DODGE_STEPS_TOTAL

        # Ejecutar esquiva si corresponde
        if is_dodging and dodge_steps > 0:
            bx += dodge_dx
            by += dodge_dy
            dodge_steps -= 1
            bx, by = clamp_ball(bx, by, bradio)
        else:
            is_dodging = False  # reset si ya terminó

        # Dibujar las bolas (B,G,R)
        cv2.circle(imagen, (int(round(pos_x)), int(round(pos_y))), radio, (0, 0, 255), -1)   # roja
        cv2.circle(imagen, (int(round(bx)), int(round(by))), bradio, (255, 0, 0), -1)       # azul

        # Mostrar la ventana
        cv2.imshow(WIN_NAME, imagen)

        # Leer tecla / detectar cierre de ventana
        key = cv2.waitKey(25) & 0xFF
        if key == ord('q') or key == 27:  # q o ESC para salir
            break

        # Si el usuario cerró la ventana con la X, getWindowProperty devuelve < 1
        if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

except KeyboardInterrupt:
    pass
finally:
    cv2.destroyAllWindows()