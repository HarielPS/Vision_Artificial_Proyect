#crear dataset esto a partir de manualmente dar el etiquetado final a todos los objetos y guardarlo en un csv

########################################################################

import os
import cv2
import pandas as pd
from leer import cargar_imagen
from procesar2 import procesar_una_imagen
from deteccion import detectar_objetos, calcular_momentos_hu
from vista import escalar_imagen, mostrar_objetos_por_separado

def procesar_y_etiquetar_imagen(ruta_imagen, etiquetas):
    imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    imgProcesada = procesar_una_imagen(ruta_imagen)
    contornos = detectar_objetos(imgProcesada)
    momentos_hu_totales = mostrar_objetos_por_separado(imagen, contornos)

    datos = []
    for i, obj in enumerate(momentos_hu_totales):
        print(f"Objeto {i + 1}: {obj}")
        etiqueta_num = int(input("Introduce el número de la clase para este objeto: "))
        etiqueta = etiquetas[etiqueta_num]
        obj_data = {
            "Hu[1]": obj["momentos_hu"][0],
            "Hu[2]": obj["momentos_hu"][1],
            "Hu[3]": obj["momentos_hu"][2],
            "Hu[4]": obj["momentos_hu"][3],
            "Hu[5]": obj["momentos_hu"][4],
            "Hu[6]": obj["momentos_hu"][5],
            "Hu[7]": obj["momentos_hu"][6],
            "area": obj["area"],
            "perimetro": obj["perimetro"],
            "objeto": etiqueta,
            "imagen": ruta_imagen
        }
        datos.append(obj_data)
    
    return datos

def main():
    # Obtener la ruta actual y construir la ruta de las imagenes
    ruta_actual = os.getcwd()
    directorio_imagenes = "nuevo"
    ruta_img = os.path.join(ruta_actual, directorio_imagenes)

    etiquetas = {
        1: "cuchara",
        2: "tenedor",
        3: "cuchillo",
        4: "encendedor",
        5: "te",
        6: "otro"
    }
    
    todos_los_datos = []
    
    for archivo in os.listdir(ruta_img):
        if archivo.endswith(('.png', '.jpg', '.jpeg')):
            ruta_imagen = os.path.join(ruta_img, archivo)
            print(f"Procesando {ruta_imagen}...")
            datos_imagen = procesar_y_etiquetar_imagen(ruta_imagen, etiquetas)
            todos_los_datos.extend(datos_imagen)
    
    # Guardar los datos en un archivo CSV
    df = pd.DataFrame(todos_los_datos)
    df.to_csv('dataset.csv', index=False)
    print("Datos guardados en dataset_objetos.csv")

main()
