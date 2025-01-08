# Mostrar la imagen en pantalla 
# Escalado de la imagen y datos de caracteristicas en consola

########################################################################
import cv2
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from deteccion import calcular_momentos_hu

def escalar_imagen(imagen):
    # Obtener resolucion de la pantalla
    root = Tk()
    root.withdraw()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Escalar imagen al 80% de la pantalla
    ancho_deseado = int(screen_width * 0.8)
    alto_deseado = int(ancho_deseado * (imagen.shape[0] / imagen.shape[1]))

    if alto_deseado > int(screen_height * 0.8):
        alto_deseado = int(screen_height * 0.8)
        ancho_deseado = int(alto_deseado * (imagen.shape[1] / imagen.shape[0]))

    imagen_redimensionada = cv2.resize(imagen, (ancho_deseado, alto_deseado))

    return imagen_redimensionada

def mostrar_objetos_por_separado(imagen, contornos):
    # Vector de caracteristicas que contiene los momentos de Hu de cada objeto
    momentos_hu_totales = []
    
    # Mostrar informacion de los objetos en consola y calcular los momentos de Hu
    print(f"Se detectaron {len(contornos)} objetos.")
    
    for i, contorno in enumerate(contornos):
        #Obtener caracteristicas basicas del objeto
        x, y, w, h = cv2.boundingRect(contorno)
        area = cv2.contourArea(contorno) 
        perimetro = cv2.arcLength(contorno, True)  
        # Calcular los momentos de Hu para cada objeto
        hu_momentos = calcular_momentos_hu(contorno, imagen)
        
        # Imprimir informacion del objeto
        print(f"Objeto {i+1}:")
        print(f"  - Coordenadas: (x: {x}, y: {y})")
        print(f"  - Ancho: {w}, Alto: {h}")
        print(f"  - Área del objeto: {area}")
        print(f"  - Perímetro del objeto: {perimetro}")
        print(f"  - Momentos de Hu:")
        for j, hu in enumerate(hu_momentos):
            print(f"      Hu[{j+1}]: {hu}")
        print("\n")

        # Almacenar caracteristicas en un diccionario
        caracteristicas_objeto = {
            "coordenadas": (x, y),
            "ancho": w,
            "alto": h,
            "area": area,
            "perimetro": perimetro,
            "momentos_hu": hu_momentos.tolist()
        }
        momentos_hu_totales.append(caracteristicas_objeto)


    # Devolver el vector de caracteristicas con los momentos de Hu
    return momentos_hu_totales
