import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import random

# Cargar datasets desde archivos CSV
df_altaBaixa = pd.read_csv('data/alta_baixaTarragona.csv', on_bad_lines='skip')
df_habTuristic = pd.read_csv('data/habTuristic.csv', on_bad_lines='skip')
df_poblacio = pd.read_csv('data/poblacio.csv', on_bad_lines='skip')
df_preuLloguer = pd.read_csv('data/preuLloguer.csv', sep=';', on_bad_lines='skip')
df_relacioEconomica = pd.read_csv('data/relacioEconomica.csv', on_bad_lines='skip')
df_tasaAtur = pd.read_csv('data/tasaAtur.csv', on_bad_lines='skip')
df_turTGNBCN = pd.read_csv('data/turTGNBCN.csv', on_bad_lines='skip')


# Función para extraer el mes de la columna 'periode'
def extract_month(periode):
    meses = {
        'gener': 1, 'febrer': 2, 'març': 3, 'abril': 4, 'maig': 5, 'juny': 6,
        'juliol': 7, 'agost': 8, 'setembre': 9, 'octubre': 10, 'novembre': 11, 'desembre': 12
    }

    if isinstance(periode, str):
        parts = periode.split('-')  # Separar el rango
        if len(parts) == 2:
            mes1, mes2 = parts[0].strip(), parts[1].strip()
            if mes1 in meses and mes2 in meses:
                return (meses[mes1] + meses[mes2]) // 2  # Tomar el mes promedio
            elif mes1 in meses:
                return meses[mes1]  # Si solo se reconoce el primero, usarlo
        elif parts[0] in meses:
            return meses[parts[0]]  # Si solo hay un mes, devolverlo

    return np.nan  # Si no se puede convertir, dejar como NaN


# Aplicar la función al DataFrame
df_preuLloguer['mes'] = df_preuLloguer['periode'].apply(extract_month)


def unify_date_format(df, year_col='any', month_col='mes'):
    df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
    df[month_col] = pd.to_numeric(df[month_col], errors='coerce')
    df['date'] = pd.to_datetime(df[year_col].astype(str) + '-' + df[month_col].astype(str), errors='coerce')
    return df


# Aplicar la función para unificar fechas
df_preuLloguer = unify_date_format(df_preuLloguer)
print(df_preuLloguer.head())
df_habTuristic = unify_date_format(df_habTuristic)
print(df_habTuristic.head())
df_tasaAtur = unify_date_format(df_tasaAtur)
print(df_tasaAtur.head())
df_relacioEconomica = unify_date_format(df_relacioEconomica)
print(df_relacioEconomica.head())

df_poblacio['periode'] = pd.to_datetime(df_poblacio['periode'], errors='coerce')

# Convertir variables categóricas en numéricas
label_encoders = {}


def encode_categorical(df, cols):
    for col in cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    return df


# Aplicar transformación
categorical_columns = ['NomTerritori', 'provincia', 'municipi', 'relacio economia']
df_preuLloguer = encode_categorical(df_preuLloguer, ['NomTerritori'])
df_habTuristic = encode_categorical(df_habTuristic, ['provincia', 'municipi'])
df_relacioEconomica = encode_categorical(df_relacioEconomica, ['relacio economia'])

# Manejo de valores faltantes con la mediana
imputer = SimpleImputer(strategy='median')
df_preuLloguer.fillna(df_preuLloguer.median(), inplace=True)
df_habTuristic.fillna(df_habTuristic.median(), inplace=True)
df_tasaAtur.fillna(df_tasaAtur.median(), inplace=True)
df_relacioEconomica.fillna(df_relacioEconomica.median(), inplace=True)

# Unir datasets
merged_df = df_preuLloguer.merge(df_tasaAtur, on=['date', 'provincia'], how='left')
merged_df = merged_df.merge(df_habTuristic, on=['date', 'municipi'], how='left')
merged_df = merged_df.merge(df_relacioEconomica, on=['date', 'provincia'], how='left')
merged_df = merged_df.merge(df_poblacio, on=['periode', 'municipi'], how='left')

# Seleccionar variables predictoras y variable objetivo
X = merged_df[['NomTerritori', 'any', 'periode', 'tasa', 'total', 'quantitat']]
y = merged_df['renda']

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Evaluar modelo XGBoost
def evaluate_xgb(params):
    model = xgb.XGBRegressor(objective='reg:squarederror', **params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_absolute_error(y_test, y_pred)


# Optimización con Hill Climbing
def hill_climbing(max_iterations=20):
    current_solution = {
        'n_estimators': random.randint(50, 200),
        'max_depth': random.randint(3, 10),
        'learning_rate': random.uniform(0.01, 0.3)
    }
    current_mae = evaluate_xgb(current_solution)
    print(f"Inicio -> Config: {current_solution}, MAE: {current_mae:.4f}")

    for i in range(max_iterations):
        new_solution = current_solution.copy()
        param_to_modify = random.choice(['n_estimators', 'max_depth', 'learning_rate'])

        if param_to_modify == 'n_estimators':
            new_solution['n_estimators'] = max(50, min(300, new_solution['n_estimators'] + random.choice([-50, 50])))
        elif param_to_modify == 'max_depth':
            new_solution['max_depth'] = max(3, min(15, new_solution['max_depth'] + random.choice([-1, 1])))
        elif param_to_modify == 'learning_rate':
            new_solution['learning_rate'] = max(0.01,
                                                min(0.5, new_solution['learning_rate'] + random.uniform(-0.05, 0.05)))

        new_mae = evaluate_xgb(new_solution)
        print(f"Iteración {i + 1} -> Config: {new_solution}, MAE: {new_mae:.4f}")

        if new_mae < current_mae:
            current_solution = new_solution
            current_mae = new_mae
            print(f"Mejora encontrada -> Config: {current_solution}, MAE: {current_mae:.4f}")

    print(f"Mejor solución encontrada: {current_solution}, MAE: {current_mae:.4f}")
    return current_solution, current_mae


# Ejecutar optimización de hiperparámetros
best_solution, best_mae = hill_climbing()
