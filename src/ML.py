# Entrena el modelo y lo guarda para su uso

########################################################################

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pickle

# Cargar datasets
img = pd.read_csv('./dataset_objetos.csv')

# Dividir características y etiquetas
x_img = img.iloc[:, :9]
y_img = img.iloc[:, 9]

# Mezclar y dividir en entrenamiento y prueba
X_train_img, X_test_img, y_train_img, y_test_img = train_test_split(
    x_img, y_img, test_size=0.3, random_state=0
)

# Entrenar el modelo de Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train_img, y_train_img)

# Guardar el modelo entrenado
with open('nb_model.pkl', 'wb') as f:
    pickle.dump(nb_model, f)

print("Modelo de Naive Bayes entrenado y guardado exitosamente.")
