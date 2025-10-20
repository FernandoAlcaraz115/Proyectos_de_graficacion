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

h, w = img.shape[:2]

# Traslación al centro
M_tras = np.float32([[1, 0, w // 4], [0, 1, h // 4]])
img_tras = cv2.warpAffine(img, M_tras, (w, h), flags=cv2.INTER_LINEAR)

# Rotar 90°
M_rot = cv2.getRotationMatrix2D((w // 2, h // 2), 90, 1)
img_rot = cv2.warpAffine(img_tras, M_rot, (w, h), flags=cv2.INTER_LINEAR)

# Escalar ×2
img_final = cv2.resize(img_rot, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

# Mostrar resultado final
cv2.imshow("Ejercicio 3 - Resultado Final", img_final)
cv2.waitKey(0)
cv2.destroyAllWindows()
