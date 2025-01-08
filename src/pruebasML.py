#Pruebas con diferentes modelos 
# con Knn | 10 vecinos | peso: Distancia | 94% 
# con gaussiana | uniforme | 100% 

########################################################################

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
#modelo probabilistico
from sklearn.naive_bayes import GaussianNB, MultinomialNB


# Cargar datasets
img = pd.read_csv('./dataset_objetos.csv')

# Dividir caracteristicas y etiquetas
x_img = img.iloc[:, :9]
y_img = img.iloc[:, 9]

# Mezclar y dividir en entrenamiento y prueba
X_train_img, X_test_img, y_train_img, y_test_img = train_test_split(
    x_img, y_img, test_size=0.3, random_state=0
)

# Funcion para validacion cruzada
def cross_validate_knn(X, y, neighbors, weights, k=5):
    kf = KFold(n_splits=k)
    model = KNeighborsClassifier(n_neighbors=neighbors, weights=weights)
    accuracies = []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx].reset_index(drop=True), X.iloc[val_idx].reset_index(drop=True)
        y_train, y_val = y.iloc[train_idx].reset_index(drop=True), y.iloc[val_idx].reset_index(drop=True)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        accuracies.append(accuracy_score(y_val, y_pred))
    return accuracies

# Generar datos para la tabla 1
results_img = []

# KNN para img.csv
for n_neighbors, weights in [(10, 'distance')]:
    accuracies = cross_validate_knn(X_train_img, y_train_img, neighbors=n_neighbors, weights=weights)
    results_img.append(["img.csv", n_neighbors, weights] + list(accuracies) + [np.mean(accuracies)])

# Crear y mostrar Tabla 1
columns = ["Dataset", "Vecinos", "Pesos", "Pliegue 1", "Pliegue 2", "Pliegue 3", "Pliegue 4", "Pliegue 5", "Promedio"]
tabla_img = pd.DataFrame(results_img, columns=columns)

print("\nTabla 1 - Validación cruzada (img.csv)")
print(tabla_img.to_string(index=False))

# Seleccionar la mejor configuración de KNN
best_config_img = tabla_img.loc[tabla_img['Promedio'].idxmax()]

# Pruebas finales con KNN
best_knn_img = KNeighborsClassifier(n_neighbors=int(best_config_img['Vecinos']), weights=best_config_img['Pesos'])
best_knn_img.fit(X_train_img, y_train_img)
y_pred_img = best_knn_img.predict(X_test_img)
accuracy_img_knn = accuracy_score(y_test_img, y_pred_img)

final_results = [["dataset_objetos.csv", "K-NN", int(best_config_img['Vecinos']), best_config_img['Pesos'], "", accuracy_img_knn]]

#otros modelos
# Naive Bayes para pruebas finales
nb_model = GaussianNB()

nb_model.fit(X_train_img, y_train_img)
y_pred_nb = nb_model.predict(X_test_img)
accuracy_nb = accuracy_score(y_test_img, y_pred_nb)
final_results.append(["dataset_objetos.csv", "Naive Bayes", "", "", "Gaussian", accuracy_nb])


# Crear y mostrar Tabla 2
columns_final = ["Dataset", "Clasificador", "Vecinos", "Pesos", "Distribución", "Accuracy"]
tabla_final = pd.DataFrame(final_results, columns=columns_final)

print("\nTabla 2 - Pruebas finales")
print(tabla_final.to_string(index=False))

# Generar matrices de confucion y reportes
print("\nMatrices de confusión y reportes de clasificación")
ConfusionMatrixDisplay.from_predictions(y_test_img, y_pred_img).plot()
print(classification_report(y_test_img, y_pred_img))
plt.show()
