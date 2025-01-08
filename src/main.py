import os
import tkinter as tk
from PIL import Image, ImageTk 
from tkinter import PhotoImage, filedialog
from PIL import Image, ExifTags

from clasificar_img import find_most_similar_images
from leer import cargar_imagen

# Se obtiene la ruta actual
ruta_actual = os.getcwd()
# folder = "data"
folder = "prueba"
dataset_path = './conteo.csv'

imagenes = {
    # "imagen_1": {"ruta": "/ruta/a/imagen1.jpg", "porcentaje": "20%"},
    # "imagen_2": {"ruta": "/ruta/a/imagen2.png", "porcentaje": "20%"},
    # "imagen_3": {"ruta": "/ruta/a/imagen3.bmp", "porcentaje": "20%"},
    # "imagen_4": {"ruta": "/ruta/a/imagen4.gif", "porcentaje": "20%"},
    # "imagen_5": {"ruta": "/ruta/a/imagen5.jpeg", "porcentaje": "20%"}
}

imagenes_actualizadas = [] 

def corregir_orientacion(imagen):
    try:
        # Obtener los metadatos EXIF
        exif = imagen._getexif()
        if exif is not None:
            # Buscar la etiqueta de orientación
            for etiqueta, valor in exif.items():
                if etiqueta == 274:  # Orientación
                    if valor == 3:
                        imagen = imagen.rotate(180, expand=True)
                    elif valor == 6:
                        imagen = imagen.rotate(270, expand=True)
                    elif valor == 8:
                        imagen = imagen.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # Si no se encuentra la orientación en los EXIF, no hacer nada
        pass
    return imagen

# def funcion_boton():
#     global label_imagen_principal, imagen_principal, imagenes_actualizadas

#     # Cargar una nueva imagen
#     imagen, ruta_imagen = cargar_imagen(folder, ruta_actual)

#     if imagen is not None:
#         try:
#             # Usar Pillow para abrir y convertir la imagen
#             nueva_imagen = Image.open(ruta_imagen)
#             nueva_imagen = nueva_imagen.resize((frame_cuadro.winfo_width(), frame_cuadro.winfo_height()), Image.Resampling.LANCZOS)
#             tk_imagen = ImageTk.PhotoImage(nueva_imagen)
            
#             # Actualizar la imagen principal
#             label_imagen_principal.config(image=tk_imagen)
#             label_imagen_principal.image = tk_imagen  # Mantener referencia
#         except Exception as e:
#             print(f"Error al cargar imagen principal: {e}")
#             label_imagen_principal.config(text="Imagen no encontrada", image="", bg="gray")


#         # Simulación de imágenes similares
#         similar_images = find_most_similar_images(imagen, ruta_imagen, dataset_path)
#         print("Imágenes similares:", similar_images)
#         for idx, (img_path, similarity) in enumerate(similar_images):
#             if idx < len(imagenes_actualizadas):
#                 label_imagen, label_texto = imagenes_actualizadas[idx]
#                 try:
#                     nueva_imagen = Image.open(img_path)
#                     # nueva_imagen = nueva_imagen.resize((300, 300), Image.Resampling.LANCZOS)
#                     nueva_imagen = nueva_imagen.resize((frame_cuadro.winfo_width(), frame_cuadro.winfo_height()), Image.Resampling.LANCZOS)
#                     tk_imagen = ImageTk.PhotoImage(nueva_imagen)
                    
#                     label_imagen.config(image=tk_imagen)
#                     label_imagen.image = tk_imagen 
#                 except Exception as e:
#                     print(f"Error al cargar imagen secundaria: {e}")
#                     label_imagen.config(text="Imagen no encontrada", bg="gray")
#                 label_texto.config(text=f"{similarity * 100:.2f}%")
    
#     print("Botón presionado")

def funcion_boton():
    global label_imagen_principal, imagen_principal, imagenes_actualizadas

    # Cargar una nueva imagen
    imagen, ruta_imagen = cargar_imagen(folder, ruta_actual)

    if imagen is not None:
        try:
            nueva_imagen = Image.open(ruta_imagen)
            nueva_imagen = corregir_orientacion(nueva_imagen) 
            nueva_imagen.thumbnail((frame_cuadro.winfo_width(), frame_cuadro.winfo_height()), Image.Resampling.LANCZOS)
            tk_imagen = ImageTk.PhotoImage(nueva_imagen)
            
            label_imagen_principal.config(image=tk_imagen)
            label_imagen_principal.image = tk_imagen  # Mantener referencia
        except Exception as e:
            print(f"Error al cargar imagen principal: {e}")
            label_imagen_principal.config(text="Imagen no encontrada", image="", bg="gray")

        # Simulación de imágenes similares
        similar_images = find_most_similar_images(imagen, ruta_imagen, dataset_path)
        for idx, (img_path, similarity) in enumerate(similar_images):
            if idx < len(imagenes_actualizadas):
                label_imagen, label_texto = imagenes_actualizadas[idx]
                try:
                    nueva_imagen = Image.open(img_path)
                    nueva_imagen = corregir_orientacion(nueva_imagen) 
                    nueva_imagen.thumbnail((frame_cuadro.winfo_width(), frame_cuadro.winfo_height()), Image.Resampling.LANCZOS)
                    tk_imagen = ImageTk.PhotoImage(nueva_imagen)

                    label_imagen.config(image=tk_imagen)
                    label_imagen.image = tk_imagen 
                except Exception as e:
                    print(f"Error al cargar imagen secundaria: {e}")
                    label_imagen.config(text="Imagen no encontrada", bg="gray")
                label_texto.config(text=f"{similarity * 100:.2f}%")


imagenprincipal = "/public/default.jpg"

withWindow = 1400
heightWindow = 650

# Crear la ventana principal
ventana = tk.Tk()
ventana.title("Clasificador")
ventana.minsize(width=withWindow, height=heightWindow)
ventana.config(padx=35, pady=35)

# Crear el frame izquierdo (30% del ancho) con borde
frame_izquierdo = tk.Frame(ventana, width=int(withWindow * 0.3), height=heightWindow, borderwidth=2, relief="solid")
frame_izquierdo.pack(side="left", fill="both", expand=True)

boton = tk.Button(frame_izquierdo, text="Haz clic aquí", command=funcion_boton)
boton.place(relx=0.5, rely=0.2,) 


# Intentar cargar la imagen principal en frame_izquierdo (ocupando el 25% de su ancho)
try:
    imagen_principal = PhotoImage(file=imagenprincipal)
    label_imagen_principal = tk.Label(frame_izquierdo, image=imagen_principal)
    label_imagen_principal.image = imagen_principal 
    label_imagen_principal.place(relwidth=0.25, relheight=1, relx=0.375, rely=0) 
except:
    # Si no se encuentra la imagen, mostrar un rectángulo gris
    print(f"Error al cargar la imagen predeterminada: ")
    label_imagen_principal = tk.Label(frame_izquierdo, text="Imagen no encontrada", bg="gray")
    label_imagen_principal.place(relwidth=0.3, relheight=0.5, relx=0.375, rely=0.3) 

# Crear el frame derecho (70% del ancho) con borde
frame_derecho = tk.Frame(ventana, width=int(withWindow * 0.7), height=heightWindow, borderwidth=2, relief="solid")
frame_derecho.pack(side="left", fill="both", expand=True)

# Crear la cuadrícula dentro del frame derecho (3 columnas y 2 filas)
imagenes_keys = list(imagenes.keys())
for fila in range(2):
    for col in range(3):
        frame_cuadro = tk.Frame(frame_derecho, borderwidth=2, relief="solid")
        frame_cuadro.grid(row=fila, column=col, padx=5, pady=5, sticky="nsew") 
        
        # Crear widgets para cada celda de la cuadrícula
        label_imagen = tk.Label(frame_cuadro, text="Imagen no cargada", bg="gray", width=20, height=10)
        label_imagen.pack(expand=True, fill="both")

        label_texto = tk.Label(frame_cuadro, text="0%", font=("Arial", 14))
        label_texto.pack(side="bottom", pady=5)

        # Añadir los widgets a la lista
        imagenes_actualizadas.append((label_imagen, label_texto))

        # Verificar si es la ultima celda, si es así dejarla en blanco
        if fila == 1 and col == 2:
            continue
        
        # Obtener la clave de la imagen que corresponde
        if fila * 3 + col < len(imagenes_keys):
            imagen_key = imagenes_keys[fila * 3 + col]
            try:
                # Intentar cargar la imagen
                imagen = PhotoImage(file=imagenes[imagen_key]["ruta"])
                label_imagen = tk.Label(frame_cuadro, image=imagen)
                label_imagen.image = imagen  # Mantener una referencia para que no se borre
                label_imagen.pack(expand=True)
            except:
                # Si no se encuentra la imagen, mostrar un rectangulo gris
                label_rect = tk.Label(frame_cuadro, text="Imagen no encontrada", bg="gray", width=20, height=10)
                label_rect.pack(expand=True)
                
            # Agregar el texto centrado
            label_texto = tk.Label(frame_cuadro, text=imagenes[imagen_key]["porcentaje"], font=("Arial", 14))
            label_texto.pack(side="bottom", pady=5)

# Ajustar proporciones para que los frames tengan el tamaño adecuado al redimensionar
ventana.update()
frame_izquierdo.pack_propagate(False)
frame_derecho.pack_propagate(False)

# Hacer que la cuadrícula en frame_derecho se expanda al redimensionar
for i in range(3):  # 3 columnas
    frame_derecho.grid_columnconfigure(i, weight=1)

for j in range(2):  # 2 filas
    frame_derecho.grid_rowconfigure(j, weight=1)

# Ejecutar la ventana
ventana.mainloop()
