import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import random

# Cargar datasets
df_preuLloguer = pd.read_csv('data/preuLloguer.csv', sep=';', on_bad_lines='skip')
df_poblacion = pd.read_csv('data/poblacio.csv', sep=';', on_bad_lines='skip')
df_habTuristic = pd.read_csv('data/habTuristic.csv', sep=';', on_bad_lines='skip')

# Normalizar nombres de columnas
df_preuLloguer.columns = df_preuLloguer.columns.str.strip().str.lower()
df_poblacion.columns = df_poblacion.columns.str.strip().str.lower()
df_habTuristic.columns = df_habTuristic.columns.str.strip().str.lower()

#Asegurase que les dades agafades son de 5 anys anteriors
df_habTuristic = df_habTuristic[df_habTuristic['tipus'] == 'Viviendas turísticas']
df_preuLloguer = df_preuLloguer[df_preuLloguer['any'] >= 2020]
df_poblacion = df_poblacion[df_poblacion['periode'] >= 2020]
df_habTuristic = df_habTuristic[df_habTuristic['any'] >= 2020]

#Renombrar columnes per l'unificació
df_habTuristic.rename(columns={'poblacio': 'nomterritori', 'total': 'hab_turistic'}, inplace=True)

#Unificar datasets segons el territori i l'any
print("Columnas preuLloguer:", df_preuLloguer.columns)
print("Columnas poblacion:", df_poblacion.columns)
df = df_preuLloguer.merge(df_poblacion, left_on=['nomterritori', 'any'], right_on=['poblacio', 'periode'], how='left')
df.drop(columns=['poblacio', 'periode'], inplace=True, errors='ignore')
df = df.merge(df_habTuristic[['nomterritori', 'any', 'hab_turistic']], on=['nomterritori', 'any'], how='right')

df['hab_turistic'] = pd.to_numeric(df['hab_turistic'], errors='coerce')

#Si falten dades omplir amb la mitjana
numeric_columns = df.select_dtypes(include=[np.number]).columns
imputer = SimpleImputer(strategy='median')
df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

#Columnes categoriques pel LabelEncoder
categorical_columns = ['nomterritori']
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

#Variables a utilitzar
features = ['nomterritori', 'any', 'habitatges', 'hab_turistic']
if 'poblacion' in df.columns:
    features.append('poblacion')

X = df[features]
y = df['renda'] # Variable a predir

#Per entrenar el model dividir les dades en train i test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_xgb(params):
    model = xgb.XGBRegressor(objective='reg:squarederror', **params)
    model.fit(X_train, y_train) # Entrenar el modelo
    y_pred = model.predict(X_test)
    return mean_absolute_error(y_test, y_pred), model

# Optimización de hiperparámetros con Hill Climbing
def hill_climbing(max_iterations=40):
    current_solution = {'n_estimators': random.randint(50, 200), #Rang de parametres a provar
                        'max_depth': random.randint(3, 10),
                        'learning_rate': random.uniform(0.01, 0.3)}
    current_mae, best_model = evaluate_xgb(current_solution)
    print(f"Iteración 0 - MAE: {current_mae:.4f}") #Anar printant el resultat de cada iteració del MAE

    for i in range(1, max_iterations + 1):
        new_solution = current_solution.copy()
        # Escull param de forma aleatoria per modificar el seu valor
        param_to_modify = random.choice(['n_estimators', 'max_depth', 'learning_rate'])

        # Modificar el valor del parametre escollit
        if param_to_modify == 'n_estimators':
            new_solution['n_estimators'] = max(50, min(200, new_solution['n_estimators'] + random.choice([-50, 50])))
        elif param_to_modify == 'max_depth':
            new_solution['max_depth'] = max(3, min(15, new_solution['max_depth'] + random.choice([-1, 1])))
        elif param_to_modify == 'learning_rate':
            new_solution['learning_rate'] = max(0.01,
                                                min(0.5, new_solution['learning_rate'] + random.uniform(-0.05, 0.05)))

        new_mae, new_model = evaluate_xgb(new_solution) #Evaluar el model amb els nous parametres
        print(f"Iteración {i} -> Config: {new_solution}, MAE: {new_mae:.4f}")

        if new_mae < current_mae: #Si el MAE es millor que el actual, actualitzar
            current_solution = new_solution
            current_mae = new_mae
            best_model = new_model
            print(f"Mejora encontrada -> Config: {current_solution}, MAE: {current_mae:.4f}")

    print(f"Mejor solución encontrada: {current_solution}, MAE: {current_mae:.4f}")
    return best_model #Retornar el millor model

# Entrenar el modelo con la mejor configuración encontrada
best_model = hill_climbing()

df_renda_agrupat = df.groupby('any')['renda'].mean() #Calcular la mitjana de la renta per any
tasa_crecimiento_renda = df_renda_agrupat.pct_change().mean() #Calcular la tasa de crecimiento anual de la renta
print("Tasa de crecimiento anual para renta:", tasa_crecimiento_renda)

# Obtener el último año disponible (base para proyecciones futuras)
last_year = int(df['any'].max())

# Propers 5 anys segons la tasa de creixement de la renta dels últims 5 anys
def predict_future_years(model, last_year, num_years=5):
    future_years = []
    base_habitatges = df[df['any'] == last_year]['habitatges'].mean()
    base_hab_turistic = df[df['any'] == last_year]['hab_turistic'].mean()
    if 'poblacion' in df.columns:
        base_poblacion = df[df['any'] == last_year]['poblacion'].mean()
    else:
        base_poblacion = None

    known_territories = label_encoders['nomterritori'].classes_ 

    for year in range(last_year + 1, last_year + num_years + 1):
        for territori in known_territories:
            new_data = {
                'nomterritori': territori,
                'any': year,
                'habitatges': base_habitatges,
                'hab_turistic': base_hab_turistic
            }
            if base_poblacion is not None:
                new_data['poblacion'] = base_poblacion

            new_data = pd.DataFrame([new_data])
            new_data_numeric = new_data.copy()
            new_data_numeric['nomterritori'] = label_encoders['nomterritori'].transform(new_data_numeric['nomterritori'].astype(str))

            # Se predice la renta y se ajusta aplicando la tasa de crecimiento de la renta
            predicted_renda = model.predict(new_data_numeric)
            ajuste = (1 + tasa_crecimiento_renda) ** (year - last_year)
            predicted_renda_adjusted = predicted_renda * ajuste

            new_data['renda_predicha'] = predicted_renda_adjusted
            future_years.append(new_data)

    return pd.concat(future_years, ignore_index=True)

# Predicción de los próximos 5 años con el modelo entrenado
predictions = predict_future_years(best_model, last_year)
predictions.to_csv('predictions.csv', index=False, sep=';')