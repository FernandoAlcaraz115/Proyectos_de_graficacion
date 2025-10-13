import cv2
import numpy as np

# Cargar imagen
img = cv2.imread("cod_zombies.jpg")
cv2.imshow("Original", img)

# Escalar ×2 con bilineal
img_escalada = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

# Rotar 45° con bilineal
h, w = img_escalada.shape[:2]
centro = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(centro, 45, 1)
img_final = cv2.warpAffine(img_escalada, M, (w, h), flags=cv2.INTER_LINEAR)

cv2.imshow("Resultado Final", img_final)

cv2.waitKey(0)
cv2.destroyAllWindows()
