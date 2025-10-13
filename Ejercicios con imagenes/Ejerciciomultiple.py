import cv2
import numpy as np
from tkinter import Tk, filedialog
from PIL import Image  # ← para abrir imágenes de forma segura

def cargar_imagen():
    """Abre una ventana para seleccionar una imagen y la carga de forma segura."""
    Tk().withdraw()
    ruta = filedialog.askopenfilename(
        title="Selecciona una imagen",
        filetypes=[("Archivos de imagen", "*.jpg *.png *.jpeg *.bmp *.jfif *.webp")]
    )
    
    if not ruta:
        print("❌ No se seleccionó ninguna imagen.")
        exit()

    try:
        # Usar PIL para abrir la imagen y convertirla a formato compatible con OpenCV
        img_pil = Image.open(ruta).convert("RGB")
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        print("✅ Imagen cargada correctamente:", ruta)
        return img
    except Exception as e:
        print("❌ Error al abrir la imagen:", e)
        exit()

def ejercicio1(img):
    img_escalada = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    h, w = img_escalada.shape[:2]
    centro = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(centro, 45, 1)
    img_rotada = cv2.warpAffine(img_escalada, M, (w, h), flags=cv2.INTER_LINEAR)
    cv2.imshow("Ejercicio 1 - Escalada x2 (Bilineal)", img_escalada)
    cv2.imshow("Ejercicio 1 - Rotada 45° (Bilineal)", img_rotada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def ejercicio2(img):
    img_escalada = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    h, w = img_escalada.shape[:2]
    centro = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(centro, 45, 1)
    img_rotada = cv2.warpAffine(img_escalada, M, (w, h), flags=cv2.INTER_LINEAR)
    cv2.imshow("Ejercicio 2 - Resultado Final", img_rotada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def ejercicio3(img):
    h, w = img.shape[:2]
    M_tras = np.float32([[1, 0, w // 4], [0, 1, h // 4]])
    img_tras = cv2.warpAffine(img, M_tras, (w, h), flags=cv2.INTER_LINEAR)
    M_rot = cv2.getRotationMatrix2D((w // 2, h // 2), 90, 1)
    img_rot = cv2.warpAffine(img_tras, M_rot, (w, h), flags=cv2.INTER_LINEAR)
    img_final = cv2.resize(img_rot, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Ejercicio 3 - Resultado Final", img_final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# -------------------------------
# PROGRAMA PRINCIPAL
# -------------------------------
print("📸 Programa de Ejercicios con Imágenes")
print("1️⃣ Ejercicio 1: Escalar x2 + Rotar 45° + Bilineal")
print("2️⃣ Ejercicio 2: Escalar x2 → Rotar 45° → Bilineal")
print("3️⃣ Ejercicio 3: Trasladar al centro → Rotar 90° → Escalar x2 → Bilineal")
print("----------------------------------------------------------")

opcion = input("Selecciona el número del ejercicio que deseas ejecutar (1-3): ")

img = cargar_imagen()

if opcion == "1":
    ejercicio1(img)
elif opcion == "2":
    ejercicio2(img)
elif opcion == "3":
    ejercicio3(img)
else:
    print("⚠️ Opción no válida. Debes ingresar 1, 2 o 3.")

