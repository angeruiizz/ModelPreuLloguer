import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import random


# Cargar datasets desde archivos CSV
df_altaBaixa = pd.read_csv('data/alta_baixaTarragona.csv', on_bad_lines='skip')
df_habTuristic = pd.read_csv('data/habTuristic.csv', on_bad_lines='skip')
df_poblacio = pd.read_csv('data/poblacio.csv', on_bad_lines='skip')
df_preuLloguer = pd.read_csv('data/preuLloguer.csv', on_bad_lines='skip')
df_relacioEconomica = pd.read_csv('data/relacioEconomica.csv', on_bad_lines='skip')
df_tasaAtur = pd.read_csv('data/tasaAtur.csv', on_bad_lines='skip')
df_turTGNBCN = pd.read_csv('data/turTGNBCN.csv', on_bad_lines='skip')











# Implementación del algoritmo Hill Climbing para optimización de hiperparámetros
def hill_climbing(max_iterations=20):
    print("Ejecutando Hill Climbing para optimización de hiperparámetros...")
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
            new_solution['learning_rate'] = max(0.01, min(0.5, new_solution['learning_rate'] + random.uniform(-0.05, 0.05)))

        new_mae = evaluate_xgb(new_solution)
        print(f"Iteración {i + 1} -> Config: {new_solution}, MAE: {new_mae:.4f}")

        if new_mae < current_mae:
            current_solution = new_solution
            current_mae = new_mae
            print(f"Mejora encontrada -> Config: {current_solution}, MAE: {current_mae:.4f}")

    print(f"Mejor solución encontrada: {current_solution}, MAE: {current_mae:.4f}")
    return current_solution, current_mae

# Ejecutar Hill Climbing
best_solution, best_mae = hill_climbing()