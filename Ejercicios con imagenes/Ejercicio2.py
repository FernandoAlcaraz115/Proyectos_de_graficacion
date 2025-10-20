import cv2
import numpy as np
from tkinter import Tk, filedialog
from PIL import Image

def cargar_imagen():
    Tk().withdraw()
    ruta = filedialog.askopenfilename(
        title="Selecciona una imagen",
        filetypes=[("Archivos de imagen", "*.jpg *.png *.jpeg *.bmp *.jfif *.webp")]
    )
    if not ruta:
        print("❌ No se seleccionó ninguna imagen.")
        exit()
    try:
        img_pil = Image.open(ruta).convert("RGB")
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        print("✅ Imagen cargada correctamente:", ruta)
        return img
    except Exception as e:
        print("❌ Error al abrir la imagen:", e)
        exit()

# --- PROCESO ---
img = cargar_imagen()

# Escalar ×2 con bilineal
img_escalada = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

# Rotar 45° con bilineal
h, w = img_escalada.shape[:2]
centro = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(centro, 45, 1)
img_rotada = cv2.warpAffine(img_escalada, M, (w, h), flags=cv2.INTER_LINEAR)

# Mostrar resultado final
cv2.imshow("Ejercicio 2 - Resultado Final", img_rotada)
cv2.waitKey(0)
cv2.destroyAllWindows()
