import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

DATASET = "data/dataset_train_completo15minutosmercado.csv"
DATASET_PUNTUACIONES = "data/dataset_puntuaciones_picos_valles.csv"
PROMINENCIA = 300

def main():
    df = pd.read_csv(DATASET, index_col=0, parse_dates=True)
    precios = df['Close'].values
    picos, _ = find_peaks(precios, prominence=PROMINENCIA)
    valles, _ = find_peaks(-precios, prominence=PROMINENCIA)
    puntuaciones_array = np.full(len(precios), np.nan)

    puntuaciones_array[picos] = -1.0   # -1.0 = Venta (Pico)
    puntuaciones_array[valles] = 1.0   #  1.0 = Compra (Valle)

    serie_puntuaciones = pd.Series(puntuaciones_array)
    serie_puntuaciones = serie_puntuaciones.interpolate(method='linear')
    serie_puntuaciones = serie_puntuaciones.fillna(0.0)
    df['Puntuacion'] = serie_puntuaciones.values.astype(float)

    df.to_csv(DATASET_PUNTUACIONES)

if __name__ == "__main__":
    main()