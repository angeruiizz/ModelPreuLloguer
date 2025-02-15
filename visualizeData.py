import pandas as pd
import matplotlib.pyplot as plt

# Cargar datasets
df_preuLloguer = pd.read_csv('data/preuLloguer.csv', sep=';', on_bad_lines='skip')
df_poblacion = pd.read_csv('data/poblacio.csv', sep=';', on_bad_lines='skip')
df_habTuristic = pd.read_csv('data/habTuristic.csv', sep=';', on_bad_lines='skip')
predictions = pd.read_csv('predictions.csv', sep=';', on_bad_lines='skip')

# Normalizar nombres de columnas
df_preuLloguer.columns = df_preuLloguer.columns.str.strip().str.lower()
df_poblacion.columns = df_poblacion.columns.str.strip().str.lower()
df_habTuristic.columns = df_habTuristic.columns.str.strip().str.lower()
predictions.columns = predictions.columns.str.strip().str.lower()

# Filtrar los datos desde 2020
df_preuLloguer = df_preuLloguer[df_preuLloguer['any'] >= 2020]
df_poblacion = df_poblacion[df_poblacion['periode'] >= 2020]
df_habTuristic = df_habTuristic[(df_habTuristic['any'] >= 2020) & (df_habTuristic['tipus'] == 'Viviendas turísticas')]

# Renombrar columnas para unificación
df_habTuristic.rename(columns={'poblacio': 'nomterritori', 'total': 'hab_turistic'}, inplace=True)

# Unificar datasets según el territorio y el año
df = df_preuLloguer.merge(df_poblacion, left_on=['nomterritori', 'any'], right_on=['poblacio', 'periode'], how='left')
df.drop(columns=['poblacio', 'periode'], inplace=True, errors='ignore')
df = df.merge(df_habTuristic[['nomterritori', 'any', 'hab_turistic']], on=['nomterritori', 'any'], how='right')
df['hab_turistic'] = pd.to_numeric(df['hab_turistic'], errors='coerce')

# Calcular valores medios por año
coste_medio_2020_2024 = df.groupby('any')['renda'].mean()
poblacion_media_2020_2024 = df.groupby('any')['total'].mean()
turismo_medio_2020_2024 = df.groupby('any')['hab_turistic'].mean()

# Filtrar las predicciones para los años 2025-2029
predictions_2025_2029 = predictions[(predictions['any'] >= 2025) & (predictions['any'] <= 2029)]
prediccion_renda = predictions_2025_2029.groupby('any')['renda_predicha'].mean()

# Crear gráficas
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# Gráfico 1: Coste Medio de la Vivienda
axs[0].plot(coste_medio_2020_2024.index, coste_medio_2020_2024.values, color='b', label='Coste medio 2020-2024', linestyle='-', marker='o')
axs[0].plot(prediccion_renda.index, prediccion_renda.values, color='r', label='Predicción 2025-2029', linestyle='-', marker='o')
axs[0].set_title('Tasa de Renda (Coste Medio de la Vivienda)')
axs[0].set_xlabel('Año')
axs[0].set_ylabel('Coste Medio (euros)')
axs[0].legend()
axs[0].grid(True)

# Gráfico 2: Tasa de Población
axs[1].plot(poblacion_media_2020_2024.index, poblacion_media_2020_2024.values, color='g', linestyle='-', marker='o')
axs[1].set_title('Tasa de Población')
axs[1].set_xlabel('Año')
axs[1].set_ylabel('Población Media')
axs[1].grid(True)

# Gráfico 3: Tasa de Turismo (Viviendas Turísticas)
axs[2].plot(turismo_medio_2020_2024.index, turismo_medio_2020_2024.values, color='m', linestyle='-', marker='o')
axs[2].set_title('Tasa de Turismo (Viviendas Turísticas)')
axs[2].set_xlabel('Año')
axs[2].set_ylabel('Número de Viviendas Turísticas')
axs[2].grid(True)

# Mostrar todas las gráficas
plt.tight_layout()
plt.show()
