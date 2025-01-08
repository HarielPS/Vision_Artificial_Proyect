#leer la imagen y regresarla con todo y ruta

########################################################################

import cv2
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def cargar_imagen(folder,ruta_actual):
    Tk().withdraw()

    # Seleccion de la imagen
    ruta_archivo = askopenfilename(
        title="Selecciona una imagen",
        initialdir="/"+folder+ruta_actual,
        filetypes=[("Archivos de imagen", "*.jpg *.png *.jpeg")]
    )

    # Cargar la imagen seleccionada
    if ruta_archivo:
        imagen = cv2.imread(ruta_archivo, cv2.IMREAD_COLOR)
        if imagen is not None:
            print(f"Imagen cargada correctamente desde: {ruta_archivo}")
            return imagen, ruta_archivo
        else:
            print("Error al cargar la imagen. Verifica la ruta o el formato.")
            return None
    else:
        print("No se seleccionó ninguna imagen.")
        return None
