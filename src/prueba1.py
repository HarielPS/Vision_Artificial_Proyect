import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import cv2
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from deteccion import detectar_objetos
from procesar2 import procesar_una_imagen
from vista import mostrar_objetos_por_separado
from leer import cargar_imagen

# Función para cargar el modelo entrenado
def load_model():
    import pickle
    with open('./nb_model.pkl', 'rb') as f:
        nb_model = pickle.load(f)
    return nb_model

# Función para clasificar la imagen
def classify_image(imagen, image_path):
    imgProcesada = procesar_una_imagen(image_path)
    contornos = detectar_objetos(imgProcesada)
    momentos_hu_totales = mostrar_objetos_por_separado(imagen, contornos)

    features = []
    for objeto in momentos_hu_totales:
        features_objeto = [
            objeto["momentos_hu"][i] for i in range(7)
        ] + [
            objeto["area"],
            objeto["perimetro"]
        ]
        features.append(features_objeto)

    features = np.array(features)
    nb_model = load_model()
    nb_pred = nb_model.predict(features)
    return nb_pred, features

# Función para contar objetos
def conteo(pred):
    pred_count = Counter(pred)
    return {
        'cuchara': pred_count.get('cuchara', 0),
        'tenedor': pred_count.get('tenedor', 0),
        'cuchillo': pred_count.get('cuchillo', 0),
        'encendedor': pred_count.get('encendedor', 0),
        'te': pred_count.get('te', 0),
        'otro': pred_count.get('otro', 0)
    }

# Función para encontrar las imágenes más similares
def find_most_similar_images(imagen, image_path, dataset_path, top_n=5):
    pred, features_input = classify_image(imagen, image_path)
    conteo_imagen = conteo(pred)
    df = pd.read_csv(dataset_path)

    similar_images = []
    for _, row in df.iterrows():
        conteo_dataset = {
            'cuchara': row['cuchara'],
            'tenedor': row['tenedor'],
            'cuchillo': row['cuchillo'],
            'encendedor': row['encendedor'],
            'te': row['te'],
            'otro': row['otro']
        }
        conteo_imagen_vector = np.array(list(conteo_imagen.values()))
        conteo_dataset_vector = np.array(list(conteo_dataset.values()))
        similarity = cosine_similarity([conteo_imagen_vector], [conteo_dataset_vector])[0][0]
        similar_images.append((row['imagen'], similarity))

    similar_images.sort(key=lambda x: x[1], reverse=True)
    return similar_images[:top_n]

# Crear interfaz gráfica
def cargar_interfaz():
    def cargar_imagen_click():
        file_path = filedialog.askopenfilename()
        if file_path:
            folder = os.path.dirname(file_path)
            imagen, ruta_imagen = cargar_imagen(folder, os.getcwd())

            if imagen is not None:
                dataset_path = './conteo.csv'
                similares = find_most_similar_images(imagen, ruta_imagen, dataset_path)

                # Mostrar imagen principal
                img_cv = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_cv)
                img_tk = ImageTk.PhotoImage(img_pil.resize((200, 200)))
                label_img.configure(image=img_tk)
                label_img.image = img_tk

                # Mostrar resultados
                for widget in resultados_frame.winfo_children():
                    widget.destroy()

                for idx, (img_path, similarity) in enumerate(similares):
                    img = Image.open(img_path).resize((100, 100))
                    img_tk = ImageTk.PhotoImage(img)
                    lbl_img = Label(resultados_frame, image=img_tk)
                    lbl_img.image = img_tk
                    lbl_img.grid(row=idx, column=0, padx=5, pady=5)

                    lbl_text = Label(resultados_frame, text=f"{similarity * 100:.2f}%")
                    lbl_text.grid(row=idx, column=1, padx=5, pady=5)

    # Configuración de la ventana
    root = tk.Tk()
    root.title("Clasificación de Imágenes")

    # Botón para cargar imagen
    btn_cargar = Button(root, text="Cargar Imagen", command=cargar_imagen_click)
    btn_cargar.pack(pady=10)

    # Etiqueta para mostrar la imagen cargada
    label_img = Label(root)
    label_img.pack(pady=10)

    # Frame para mostrar resultados
    resultados_frame = tk.Frame(root)
    resultados_frame.pack(pady=10)

    root.mainloop()

# Ejecutar la interfaz
def main():
    cargar_interfaz()

if __name__ == "__main__":
    main()
