import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Cargar datasets
df_preuLloguer = pd.read_csv('data/preuLloguer.csv', sep=';', on_bad_lines='skip')
df_poblacion = pd.read_csv('data/poblacio.csv', sep=';', on_bad_lines='skip')
df_habTuristic = pd.read_csv('data/habTuristic.csv', sep=';', on_bad_lines='skip')

# Normalizar nombres de columnas
df_preuLloguer.columns = df_preuLloguer.columns.str.strip().str.lower()
df_poblacion.columns = df_poblacion.columns.str.strip().str.lower()
df_habTuristic.columns = df_habTuristic.columns.str.strip().str.lower()

# (Opcional) Filtrar o preparar cada dataset según tus necesidades
# Ejemplo: filtrar datos a partir del año 2020
df_preuLloguer = df_preuLloguer[df_preuLloguer['any'] >= 2020]
df_poblacion = df_poblacion[df_poblacion['periode'] >= 2020]
df_habTuristic = df_habTuristic[df_habTuristic['any'] >= 2020]

# Renombrar columnas para facilitar la unión (ajusta según corresponda)
df_habTuristic.rename(columns={'poblacio': 'nomterritori', 'total': 'hab_turistic'}, inplace=True)

# Unificar datasets según 'nomterritori' y 'any'
# Ajusta los nombres de las columnas en los merges según tus datasets reales
df = df_preuLloguer.merge(
    df_poblacion,
    left_on=['nomterritori', 'any'],
    right_on=['poblacio', 'periode'],
    how='left'
)
df.drop(columns=['poblacio', 'periode'], inplace=True, errors='ignore')
df = df.merge(
    df_habTuristic[['nomterritori', 'any', 'hab_turistic']],
    on=['nomterritori', 'any'],
    how='left'
)

# Definir las variables (asegúrate de que existan en el dataframe)
features = ['nomterritori', 'any', 'habitatges', 'hab_turistic']
if 'poblacion' in df.columns:
    features.append('poblacion')

# Para el ejemplo, asumiremos que 'renda' es la variable objetivo
# Asegúrate de que 'renda' esté presente en df o ajusta según corresponda

# ===============================
# 1. Gráfico: Evolución temporal de variables
# ===============================
plt.figure(figsize=(10,6))
# Agrupar por año y calcular la media de cada variable
df_grouped = df.groupby('any').mean()
plt.plot(df_grouped.index, df_grouped['habitatges'], marker='o', label='Habitatges')
plt.plot(df_grouped.index, df_grouped['hab_turistic'], marker='o', label='Hab. Turístic')
if 'poblacion' in df.columns:
    plt.plot(df_grouped.index, df_grouped['poblacion'], marker='o', label='Población')
plt.xlabel('Año')
plt.ylabel('Media')
plt.title('Tendencia de variables a lo largo del tiempo')
plt.legend()
plt.grid(True)
plt.show()

# ===============================
# 2. Gráfico: Tasa de crecimiento anual de las variables
# ===============================
# Supongamos que has calculado las tasas de crecimiento previamente:
# Por ejemplo, tasa_creixementHAB, tasa_hab y, si existe, tasa_poblacio
# Para este ejemplo, definimos valores ficticios:
tasa_creixementHAB = 0.05  # 5%
tasa_hab = 0.03           # 3%
tasa_poblacio = 0.02      # 2%

growth_rates = {
    'Habitatges': tasa_creixementHAB,
    'Hab. Turístic': tasa_hab
}
if 'poblacion' in df.columns:
    growth_rates['Población'] = tasa_poblacio

plt.figure(figsize=(6,4))
sns.barplot(x=list(growth_rates.keys()), y=list(growth_rates.values()), palette="viridis")
plt.ylabel('Tasa de crecimiento')
plt.title('Tasa de crecimiento anual de las variables')
plt.ylim(0, max(growth_rates.values()) * 1.2)
plt.show()

# ===============================
# 3. Gráfico: Matriz de correlación entre las características y la variable objetivo
# ===============================
plt.figure(figsize=(8,6))
# Asegúrate de que 'renda' exista en el dataframe; de lo contrario, ajústalo.
corr_matrix = df[features + ['renda']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriz de correlación entre variables y renta')
plt.show()

# ===============================
# 4. Gráfico: Importancia de las características según el modelo XGBoost
# ===============================
# Supongamos que ya has entrenado tu modelo y se llama 'best_model'
# Por ejemplo, para este ejemplo, creamos importancias ficticias:
import numpy as np
importances = np.array([0.25, 0.20, 0.30, 0.15] + ([0.10] if 'poblacion' in df.columns else []))

plt.figure(figsize=(6,4))
sns.barplot(x=importances, y=features, palette="magma")
plt.xlabel('Importancia')
plt.title('Importancia de las características en el modelo XGBoost')
plt.show()
