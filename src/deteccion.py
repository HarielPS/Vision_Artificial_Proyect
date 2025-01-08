#Deteccion de objetos con momentos de hu

########################################################################

import cv2
import numpy as np


def calcular_momentos_hu(contorno, imagen):
    # Calcular el momento de la imagen para el contorno
    momento = cv2.moments(contorno)
    
    # Extraer los momentos de Hu
    hu_momentos = cv2.HuMoments(momento).flatten()
    
    return hu_momentos

def detectar_objetos(imagen_binaria):
    # # Aplicar una operasion morfologica de cierre para eliminar ruido
    # kernel = np.ones((5, 5), np.uint8) 
    # imagen_binaria = cv2.morphologyEx(imagen_binaria, cv2.MORPH_CLOSE, kernel)
    
    # Obtener los contornos de la imagen binaria
    contornos, _ = cv2.findContours(imagen_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrar los contornos pequeños
    contornos_filtrados = [contorno for contorno in contornos if cv2.contourArea(contorno) > 500] 
    return contornos_filtrados
