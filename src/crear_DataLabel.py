import pandas as pd

# Leer el archivo CSV original
df = pd.read_csv('./dataset_objetos.csv')

# Crear un nuevo DataFrame para el conteo
conteo_data = []

# Listado de etiquetas de los objetos
etiquetas = ['cuchara', 'tenedor', 'cuchillo', 'encendedor', 'te', 'otro']

# Agrupar por la columna 'imagen' para contar los objetos por imagen
for imagen, group in df.groupby('imagen'):
    conteo = { 'imagen': imagen }
    
    # Contar la cantidad de cada objeto en la imagen
    for etiqueta in etiquetas:
        conteo[etiqueta] = (group['objeto'] == etiqueta).sum()
    
    conteo_data.append(conteo)

# Crear el DataFrame final con los conteos
conteo_df = pd.DataFrame(conteo_data)

# Guardar el nuevo archivo CSV con los conteos
conteo_df.to_csv('conteo.csv', index=False)

print("Archivo 'conteo.csv' guardado exitosamente.")
