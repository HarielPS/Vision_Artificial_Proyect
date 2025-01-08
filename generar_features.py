import pandas as pd
import numpy as np
import pickle

# Ruta del archivo CSV
csv_path = "./dataset_objetos.csv"

# Ruta donde se guardara el archivo pickle
output_pickle_path = "./training_features.pkl"

def generate_training_features(csv_path, output_pickle_path):
    # Cargar el dataset desde el archivo CSV
    try:
        data = pd.read_csv(csv_path)
        print(f"Dataset cargado correctamente desde: {csv_path}")
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en la ruta {csv_path}.")
        return
    except Exception as e:
        print(f"Error al cargar el archivo CSV: {e}")
        return
    
    # Verificar que el CSV contenga las columnas necesarias
    required_columns = [
        "Hu[1]","Hu[2]","Hu[3]","Hu[4]","Hu[5]","Hu[6]","Hu[7]","area","perimetro"
    ]
    
    for col in required_columns:
        if col not in data.columns:
            print(f"Error: Falta la columna requerida '{col}' en el dataset.")
            return
    
    # Extraer solo las características necesarias (sin las etiquetas de clase)
    features = data[required_columns].to_numpy()

    # Guardar las características en un archivo pickle
    try:
        with open(output_pickle_path, "wb") as f:
            pickle.dump(features, f)
        print(f"Archivo de características generado y guardado en: {output_pickle_path}")
    except Exception as e:
        print(f"Error al guardar el archivo pickle: {e}")

if __name__ == "__main__":
    # Llamar a la función para generar el archivo pickle
    generate_training_features(csv_path, output_pickle_path)
