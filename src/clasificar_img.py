#Introducir una imagen y que te regrese la clasificacion de cada objeto en la imagen

########################################################################

import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd

from deteccion import calcular_momentos_hu, detectar_objetos
from leer import cargar_imagen
from procesar2 import procesar_una_imagen
from vista import mostrar_objetos_por_separado
#import para las imagenes similares
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity



# Funcion para cargar el modelo entrenado
def load_model():
    with open('./nb_model.pkl', 'rb') as f:
        nb_model = pickle.load(f)
    
    return nb_model

# Funsion para clasificar la imagen
def classify_image(imagen, image_path):
    # Procesar la imagen
    imgProcesada = procesar_una_imagen(image_path)
    
    # Detectar objetos en la imagen
    contornos = detectar_objetos(imgProcesada)
    
    # Extraer los momentos de Hu de cada objeto detectado
    momentos_hu_totales = mostrar_objetos_por_separado(imagen, contornos)

    # Crear la lista de caracteristicas para clasificacion
    features = []
    
    # Obtener las 9 caracteristicas para la clasificacion (momentos de Hu, área y perímetro)
    for objeto in momentos_hu_totales:
        # Extraer las caracteristicas de cada objeto
        features_objeto = [
            objeto["momentos_hu"][0],  # Hu[1]
            objeto["momentos_hu"][1],  # Hu[2]
            objeto["momentos_hu"][2],  # Hu[3]
            objeto["momentos_hu"][3],  # Hu[4]
            objeto["momentos_hu"][4],  # Hu[5]
            objeto["momentos_hu"][5],  # Hu[6]
            objeto["momentos_hu"][6],  # Hu[7]
            objeto["area"],            # Área
            objeto["perimetro"]       # Perímetro
        ]
        
        # Agregar las caracteristicas del objeto al vector de caracteristicas
        features.append(features_objeto)
    
    features = np.array(features) 
    
    # Cargar el modelo entrenado
    nb_model = load_model()
    
    # Clasificar con Naive Bayes
    nb_pred = nb_model.predict(features)
    
    # Mostrar el resultado de la clasificacion
    print(f"clasificacion con Naive Bayes: {nb_pred}")
    
    # # Mostrar la imagen con la clasificacion como título
    # plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
    # plt.title(f"clasificacion: {', '.join(map(str, nb_pred))}")
    # plt.axis('off')
    # plt.show()

    # Devolver las predicciones y características
    return nb_pred, features

def conteo(pred):
    # Contamos las ocurrencias de cada etiqueta en la predicción
    pred_count = Counter(pred)
    
    # Creamos un diccionario con el conteo de objetos
    conteo_objetos = {
        'cuchara': pred_count.get('cuchara', 0),
        'tenedor': pred_count.get('tenedor', 0),
        'cuchillo': pred_count.get('cuchillo', 0),
        'encendedor': pred_count.get('encendedor', 0),
        'te': pred_count.get('te', 0),
        'otro': pred_count.get('otro', 0)
    }
    return conteo_objetos


# Función para calcular la similitud entre las imágenes del dataset
def find_most_similar_images(imagen, image_path, dataset_path, top_n=5):
    # Cargar la imagen a clasificar y obtener sus etiquetas
    pred, features_input = classify_image(imagen, image_path)
    conteo_imagen = conteo(pred)

    # Cargar el dataset desde el archivo CSV
    df = pd.read_csv(dataset_path)

    # Creamos un DataFrame con los conteos de objetos por imagen
    similar_images = []
    
    for _, row in df.iterrows():
        # Extraemos el conteo de objetos de cada imagen en el dataset
        conteo_dataset = {
            'cuchara': row['cuchara'],
            'tenedor': row['tenedor'],
            'cuchillo': row['cuchillo'],
            'encendedor': row['encendedor'],
            'te': row['te'],
            'otro': row['otro']
        }
        
        # Convertimos el conteo de objetos de la imagen y del dataset en vectores
        conteo_imagen_vector = np.array(list(conteo_imagen.values()))
        conteo_dataset_vector = np.array(list(conteo_dataset.values()))
        
        # Calculamos la similitud del coseno entre los vectores
        similarity = cosine_similarity([conteo_imagen_vector], [conteo_dataset_vector])[0][0]
        
        # Almacenamos la imagen y su similitud
        similar_images.append((row['imagen'], similarity))

    # Ordenamos las imágenes por la similitud en orden descendente
    similar_images.sort(key=lambda x: x[1], reverse=True)
    
    # Devolvemos las imágenes más similares
    return similar_images[:top_n]
    