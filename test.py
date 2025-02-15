import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import random

# Cargar datasets
df_altaBaixa = pd.read_csv('data/alta_baixaTarragona.csv', on_bad_lines='skip')
df_habTuristic = pd.read_csv('data/habTuristic.csv', on_bad_lines='skip')
df_poblacio = pd.read_csv('data/poblacio.csv', on_bad_lines='skip')
df_preuLloguer = pd.read_csv('data/preuLloguer.csv', on_bad_lines='skip')
df_relacioEconomica = pd.read_csv('data/relacioEconomica.csv', on_bad_lines='skip')
df_tasaAtur = pd.read_csv('data/tasaAtur.csv', on_bad_lines='skip')
df_turTGNBCN = pd.read_csv('data/turTGNBCN.csv', on_bad_lines='skip')



print(df_preuLloguer.columns)