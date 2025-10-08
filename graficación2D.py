import numpy as np   # Importa la librería NumPy, útil para trabajar con arreglos y operaciones numéricas.
import cv2 as cv     # Importa la librería OpenCV, que se utiliza para procesamiento de imágenes.

# Crea una imagen de 500x500 píxeles, todos con valor 240 (gris claro). 
# La imagen tiene solo un canal (escala de grises) y está inicializada con valores de tipo uint8 (enteros sin signo de 8 bits).
img = np.ones((500, 500), dtype=np.uint8) * 240

# Modifica algunos píxeles específicos en las coordenadas (30, 30) a (30, 35) para que tengan un valor de 1 (casi negro).
# Esto creará una pequeña línea vertical de 6 píxeles en la imagen de color casi negro.
img[30, 30] = 1
img[30, 31] = 1
img[30, 32] = 1
img[30, 33] = 1
img[30, 34] = 1
img[30, 35] = 1

# Muestra la imagen en una ventana con el título 'img'. 
cv.imshow('img', img)

# Espera a que el usuario presione cualquier tecla para continuar.
cv.waitKey()

# Cierra todas las ventanas creadas por OpenCV.
cv.destroyAllWindows()