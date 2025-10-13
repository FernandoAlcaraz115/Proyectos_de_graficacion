import cv2
import numpy as np

# Cargar imagen
img = cv2.imread("maxresdefault.jpg")
cv2.imshow("Original", img)

# Obtener dimensiones
h, w = img.shape[:2]

# --- Traslación al centro ---
M_tras = np.float32([[1, 0, w//4], [0, 1, h//4]])  # mueve la imagen hacia el centro
img_tras = cv2.warpAffine(img, M_tras, (w, h), flags=cv2.INTER_LINEAR)

# --- Rotar 90° ---
M_rot = cv2.getRotationMatrix2D((w//2, h//2), 90, 1)
img_rot = cv2.warpAffine(img_tras, M_rot, (w, h), flags=cv2.INTER_LINEAR)

# --- Escalar ×2 con bilineal ---
img_final = cv2.resize(img_rot, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

cv2.imshow("Resultado Final", img_final)

cv2.waitKey(0)
cv2.destroyAllWindows()