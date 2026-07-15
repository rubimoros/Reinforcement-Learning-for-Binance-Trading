import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

DATASET = "data/dataset_train_completoATR15minutos2021_24.csv"
DATASET_PUNTUACIONES = "data/dataset_puntuaciones_picos_valles_atr_15minutos2021_24.csv"
VENTANA_ATR = 14
MULTIPLICADOR_ATR = 2.0

def calcular_atr(df, ventana=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=ventana).mean()
    atr = atr.bfill().fillna(0)
    return atr

def main():

    df = pd.read_csv(DATASET, index_col=0, parse_dates=True)
    precios = df['Close'].values

    atr = calcular_atr(df, VENTANA_ATR)
    
    prominencia_dinamica = (atr * MULTIPLICADOR_ATR).values

    picos, _ = find_peaks(precios, prominence=prominencia_dinamica)
    valles, _ = find_peaks(-precios, prominence=prominencia_dinamica)

    puntuaciones_array = np.full(len(precios), np.nan)
    puntuaciones_array[picos] = -1.0   # Venta Perfecta
    puntuaciones_array[valles] = 1.0   # Compra Perfecta

    serie_puntuaciones = pd.Series(puntuaciones_array)
    serie_puntuaciones = serie_puntuaciones.interpolate(method='linear')
    serie_puntuaciones = serie_puntuaciones.fillna(0.0) # 0.0 es el Hold
    df['Puntuacion'] = serie_puntuaciones.values.astype(float)

    df.to_csv(DATASET_PUNTUACIONES)

if __name__ == "__main__":
    main()