# funciones para procesar una imagen
#       - Transformar a grises
#       - Operaciones Morfologicas cierre  y umbralizacion

########################################################################

import cv2
from tkinter import Tk
import numpy as np
from tkinter.filedialog import askopenfilename
from vista import escalar_imagen

def procesar_una_imagen(ruta_imagen):

    #convertir imagen de color a escala de grises
    imagengray = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)

    # Aplicar filtro Gaussiano para suavizar la imagen
    imagen_suavizada = cv2.GaussianBlur(imagengray, (5, 5), 0)

    #umbralizar la imagen (Binaria)
    _, umbralizada = cv2.threshold(imagen_suavizada, 40, 255, cv2.THRESH_BINARY)

    # Aplicar operación morfologica de cierre para cerrar huecos
    kernel = np.ones((5, 5), np.uint8)
    umbralizada = cv2.morphologyEx(umbralizada, cv2.MORPH_CLOSE, kernel)

    # Mostrar la imagen umbralizada
    # cv2.namedWindow("Imagen umbralizada", cv2.WINDOW_AUTOSIZE)
    # cv2.imshow("Imagen umbralizada", escalar_imagen(umbralizada))
    print("Imagen Procesada")
    return umbralizada