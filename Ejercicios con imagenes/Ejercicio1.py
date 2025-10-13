import cv2
import numpy as np

# Cargar imagen
img = cv2.imread("cod_zombies.jpg") 
cv2.imshow("Original", img)

# Escalar a un factor de 2 con interpolación bilineal
img_escalada = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
cv2.imshow("Escalada x2 (Bilineal)", img_escalada)

# Rotar 45 grados con interpolación bilineal
h, w = img_escalada.shape[:2]
centro = (w // 2, h // 2)
matriz_rot = cv2.getRotationMatrix2D(centro, 45, 1)
img_rotada = cv2.warpAffine(img_escalada, matriz_rot, (w, h), flags=cv2.INTER_LINEAR)
cv2.imshow("Rotada 45° (Bilineal)", img_rotada)

cv2.waitKey(0)
cv2.destroyAllWindows()
