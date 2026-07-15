import pandas as pd
import os
import sys
from utils.klines_example import klines_example_time_tfg

def generar_dataset():
    symbol = "BTCUSDT"  # Par Bitcoin/Dólar
    datos_crudos = klines_example_time_tfg(symbol)
    
    if not datos_crudos:
        return
    columnas = [
        "Open Time", "Open", "High", "Low", "Close", "Volume",
        "Close Time", "Quote Asset Volume", "Number of Trades",
        "Taker Buy Base", "Taker Buy Quote", "Ignore"
    ]
    
    df = pd.DataFrame(datos_crudos, columns=columnas)
    df["Date"] = pd.to_datetime(df["Open Time"], unit='ms')
    cols_numericas = ["Open", "High", "Low", "Close", "Volume"]
    for col in cols_numericas:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    nombre_archivo = "dataset_btc_inicial.csv"
    df.to_csv(nombre_archivo, index=False)

if __name__ == "__main__":
    generar_dataset()