import pandas as pd
import numpy as np

DATASET = "dataset_train_completo.csv"
DATASET_PUNTUACIONES = "dataset_puntuaciones.csv"
VENTANA = 24

df = pd.read_csv(DATASET, index_col=0, parse_dates=True)

# Calcular precios a futuro
df['Future_Price'] = df['Close'].shift(-VENTANA).rolling(window=VENTANA).mean()
df['Future_Return'] = (df['Future_Price'] - df['Close']) / df['Close']

# Puntuaciones
df_clean = df.dropna(subset=['Future_Return']).copy()
puntuaciones = [1,2,3,4,5,6,7,8,9,10]
df_clean['Puntuacion'] = pd.qcut(df_clean['Future_Return'], q=10, labels=puntuaciones)

df = df.drop(columns=['Future_Price', 'Future_Return'])
df['Puntuacion'] = df_clean['Puntuacion']
df['Puntuacion'] = df['Puntuacion'].fillna(5).astype(int)

distribucion = df['Puntuacion'].value_counts().sort_index()

df.to_csv(DATASET_PUNTUACIONES)