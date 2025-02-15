import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import random
from pymongo import MongoClient
from dotenv import load_dotenv

# Cargar datasets
df_preuLloguer = pd.read_csv('data/preuLloguer.csv', sep=';', on_bad_lines='skip')
df_poblacion = pd.read_csv('data/poblacio.csv', sep=';', on_bad_lines='skip')

# Normalizar nombres de columnas
df_preuLloguer.columns = df_preuLloguer.columns.str.strip().str.lower()
df_poblacion.columns = df_poblacion.columns.str.strip().str.lower()

# Unir los datasets
print("Columnas preuLloguer:", df_preuLloguer.columns)
print("Columnas poblacion:", df_poblacion.columns)
df = df_preuLloguer.merge(df_poblacion, left_on=['nomterritori', 'any'], right_on=['poblacio', 'periode'], how='left')
df.drop(columns=['poblacio', 'periode'], inplace=True, errors='ignore')

# Identificar columnas numéricas y categóricas
numeric_columns = df.select_dtypes(include=[np.number]).columns
categorical_columns = ['nomterritori']

# Manejo de valores faltantes en columnas numéricas
imputer = SimpleImputer(strategy='median')
df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

# Aplicar Label Encoding a las columnas categóricas
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Seleccionar variables predictoras y objetivo
features = ['nomterritori', 'any', 'habitatges']
if 'poblacion' in df.columns:
    features.append('poblacion')

X = df[features]
y = df['renda']

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Evaluar modelo XGBoost
def evaluate_xgb(params):
    model = xgb.XGBRegressor(objective='reg:squarederror', **params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_absolute_error(y_test, y_pred), model


# Optimización con Hill Climbing
def hill_climbing(max_iterations=20):
    current_solution = {'n_estimators': random.randint(50, 400),
                        'max_depth': random.randint(3, 10),
                        'learning_rate': random.uniform(0.01, 0.3)}
    current_mae, best_model = evaluate_xgb(current_solution)

    for _ in range(max_iterations):
        new_solution = current_solution.copy()
        param_to_modify = random.choice(['n_estimators', 'max_depth', 'learning_rate'])

        if param_to_modify == 'n_estimators':
            new_solution['n_estimators'] = max(50, min(300, new_solution['n_estimators'] + random.choice([-50, 50])))
        elif param_to_modify == 'max_depth':
            new_solution['max_depth'] = max(3, min(15, new_solution['max_depth'] + random.choice([-1, 1])))
        elif param_to_modify == 'learning_rate':
            new_solution['learning_rate'] = max(0.01,
                                                min(0.5, new_solution['learning_rate'] + random.uniform(-0.05, 0.05)))

        new_mae, new_model = evaluate_xgb(new_solution)
        if new_mae < current_mae:
            current_solution, current_mae, best_model = new_solution, new_mae, new_model

    return best_model


# Entrenar modelo con la mejor configuración encontrada
best_model = hill_climbing()


# Función para predecir los próximos 5 años
def predict_future_years(model, last_year, num_years=5):
    future_years = []
    # Iteramos sobre las categorías conocidas (en formato de texto) según el label encoder
    known_territories = label_encoders['nomterritori'].classes_

    for year in range(last_year + 1, last_year + num_years + 1):
        for territori in known_territories:
            # Creamos el DataFrame con el nombre del territorio (en formato texto)
            new_data = pd.DataFrame({
                'nomterritori': [territori],
                'any': [year],
                'habitatges': [df['habitatges'].mean()]
            })
            if 'poblacion' in df.columns:
                new_data['poblacion'] = df['poblacion'].mean()

            # Guardamos el nombre original en una variable
            territori_str = new_data['nomterritori'].copy()

            # Creamos un DataFrame para la predicción (sin la columna de texto extra)
            new_data_numeric = new_data.copy()
            new_data_numeric['nomterritori'] = label_encoders['nomterritori'].transform(
                new_data_numeric['nomterritori'].astype(str))

            # Realizamos la predicción usando solo columnas numéricas
            predicted_renda = model.predict(new_data_numeric)

            # Agregamos el resultado y restauramos el nombre original en el DataFrame final
            new_data['renda_predicha'] = predicted_renda
            # Ya tenemos el nombre original en 'nomterritori'
            future_years.append(new_data)

    return pd.concat(future_years, ignore_index=True)

# Obtener último año disponible y hacer predicción
last_year = int(df['any'].max())
predictions = predict_future_years(best_model, last_year)
predictions.to_csv('predictions.csv', index=False, sep=';')

'''
# Guardar en MongoDB
load_dotenv(dotenv_path='psswd.env')  # Fichero con la contraseña de la base de datos
MONGO_URI = os.getenv('MONGO_URI')

client = MongoClient(MONGO_URI)
df_pred = pd.read_csv('predictions.csv', delimiter=';')
data = df_pred.to_dict(orient='records')

collection_name = os.path.splitext(os.path.basename('predictions.csv'))[0]
db = client['hackato-urv']
collection = db[collection_name]
collection.insert_many(data)
print('Datos insertados en la colección', collection_name)

client.close()
'''