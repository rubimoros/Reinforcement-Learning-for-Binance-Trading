import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import os

def main():
    RUTA_DATOS = "data/dataset_puntuaciones_picos_valles_atr_15minutos2021.csv"
    
    try:
        df = pd.read_csv(RUTA_DATOS, index_col=0, parse_dates=True)
    except FileNotFoundError:
        return

    columnas_a_borrar = ["Open", "High", "Low", "Close", "Volume", "Puntuacion"]
    indicadores = df.drop(columns=columnas_a_borrar, errors='ignore').copy()
    scaler = StandardScaler()
    scaler.fit(indicadores)
    os.makedirs("scalers", exist_ok=True)
    ruta_guardado = "scalers/scaler_15m.pkl"
    joblib.dump(scaler, ruta_guardado)

if __name__ == "__main__":
    main()