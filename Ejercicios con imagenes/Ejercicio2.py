import cv2
import numpy as np
from tkinter import Tk, filedialog

Tk().withdraw()

ruta = filedialog.askopenfilename(title="Selecciona una imagen",
                                  filetypes=[("Archivos de imagen", "*.jpg *.png *.jpeg *.bmp")])
img = cv2.imread(ruta)

if img is None:
    print("❌ No se pudo cargar la imagen.")
    exit()
else:
    print("✅ Imagen cargada correctamente:", ruta)

# Escalar ×2 con filtro bilineal
img_escalada = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

# Rotar 45° con filtro bilineal
h, w = img_escalada.shape[:2]
centro = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(centro, 45, 1)
img_rotada = cv2.warpAffine(img_escalada, M, (w, h), flags=cv2.INTER_LINEAR)

cv2.imshow("Original", img)
cv2.imshow("Resultado Final", img_rotada)
cv2.waitKey(0)
cv2.destroyAllWindows()

