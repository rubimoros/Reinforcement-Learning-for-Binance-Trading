import pandas as pd
import numpy as np
import os
import sys
import time

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from utils.klines_example import klines_example_time_tfg
from utils.indicators_final import TechnicalIndicators

PERIODOS = {
    "1_Marzo_2020_Covid": ("2020-03-01", "2020-03-31"),
    "2_Febrero_2021_Bull": ("2021-02-01", "2021-02-28"),
    "3_Julio_2021_Lateral": ("2021-07-01", "2021-07-31"),
    "4_Junio_2022_Bear": ("2022-06-01", "2022-06-30"),
    "5_Noviembre_2022_Cambio": ("2022-11-01", "2022-11-30"),
    "6_Agosto_2023_Volatilidad": ("2023-08-01", "2023-08-31"),
    "7_Marzo_2024_BullReciente": ("2024-03-01", "2024-03-31")
}

SYMBOL = "BTCUSDT"
INTERVAL = "15m" 
DATA_DIR = os.path.join(BASE_DIR, "data", "evaluaciones_tfg")

def generar_dataset_mes(nombre, start_str, end_str, output_path):
    raw_data = klines_example_time_tfg(SYMBOL, start_str, end_str, INTERVAL)
    if not raw_data:
        print(f"Error al descargar {nombre}")
        return False

    columnas = ["Open Time", "Open", "High", "Low", "Close", "Volume",
                "Close Time", "Quote Volume", "Number of Trades",
                "Taker Buy Base", "Taker Buy Quote", "Ignore"]
    df = pd.DataFrame(raw_data, columns=columnas)

    numericas = ["Open", "High", "Low", "Close", "Volume"]
    df[numericas] = df[numericas].apply(pd.to_numeric, axis=1)

    df["Time"] = pd.to_datetime(df["Open Time"], unit='ms')
    df = df.set_index("Time").sort_index()
    df_base = df[["Open", "High", "Low", "Close", "Volume"]].copy()

    # ---INDICADORES ---
    sma = TechnicalIndicators.calculate_sma(df_base, period_1=20, period_2=80)
    df_base = df_base.join(sma.rename(columns={"INDICADOR_1": "SMA_20", "INDICADOR_2": "SMA_80"}))
    ema = TechnicalIndicators.calculate_ema(df_base, period_1=20, period_2=80)
    df_base = df_base.join(ema.rename(columns={"INDICADOR_1": "EMA_20", "INDICADOR_2": "EMA_80"}))
    macd = TechnicalIndicators.calculate_macd(df_base)
    df_base = df_base.join(macd.rename(columns={"INDICADOR_1": "MACD", "INDICADOR_2": "MACD_Signal"}))
    ichimoku = TechnicalIndicators.calculate_ichimoku(df_base)
    df_base = df_base.join(ichimoku.rename(columns={"INDICADOR_1": "Ichi_Tenkan", "INDICADOR_2": "Ichi_Kijun", 
                                                   "INDICADOR_3": "Ichi_SenkouA", "INDICADOR_4": "Ichi_SenkouB", "INDICADOR_5": "Ichi_Lagging"}))
    sar = TechnicalIndicators.calculate_sar(df_base)
    df_base = df_base.join(sar.rename(columns={"INDICADOR_1": "SAR"}))
    supertrend = TechnicalIndicators.calculate_supertrend(df_base)
    df_base = df_base.join(supertrend.rename(columns={"INDICADOR_1": "Supertrend_Upper", "INDICADOR_2": "Supertrend_Lower"}))
    adx = TechnicalIndicators.calculate_adx(df_base)
    df_base = df_base.join(adx.rename(columns={"INDICADOR_1": "ADX_Pos", "INDICADOR_2": "ADX_Neg", "INDICADOR_3": "ADX_Main"}))
    rsi = TechnicalIndicators.calculate_rsi(df_base)
    df_base = df_base.join(rsi.rename(columns={"INDICADOR_1": "RSI_14", "INDICADOR_2": "RSI_28"}))
    stoch = TechnicalIndicators.calculate_stochastic_oscillator(df_base)
    df_base = df_base.join(stoch.rename(columns={"INDICADOR_1": "Stoch_K", "INDICADOR_2": "Stoch_D"}))
    cci = TechnicalIndicators.calculate_cci(df_base)
    df_base = df_base.join(cci.rename(columns={"INDICADOR_1": "CCI"}))
    williams = TechnicalIndicators.calculate_williams_r(df_base)
    df_base = df_base.join(williams.rename(columns={"INDICADOR_1": "Williams"}))
    cmo = TechnicalIndicators.calculate_cmo(df_base)
    df_base = df_base.join(cmo.rename(columns={"INDICADOR_1": "CMO"}))
    bollinger = TechnicalIndicators.calculate_bollinger_bands(df_base)
    df_base = df_base.join(bollinger.rename(columns={"INDICADOR_1": "Bollinger_Middle", "INDICADOR_2": "Bollinger_Upper", "INDICADOR_3": "Bollinger_Lower"}))
    atr = TechnicalIndicators.calculate_atr(df_base)
    df_base = df_base.join(atr.rename(columns={"INDICADOR_1": "ATR"}))
    keltner = TechnicalIndicators.calculate_keltner_channels(df_base)
    df_base = df_base.join(keltner.rename(columns={"INDICADOR_1": "Keltner_Upper", "INDICADOR_2": "Keltner_Middle", "INDICADOR_3": "Keltner_Lower"}))
    donchian = TechnicalIndicators.calculate_donchian_channel(df_base)
    df_base = df_base.join(donchian.rename(columns={"INDICADOR_1": "Donchian_Upper", "INDICADOR_2": "Donchian_Lower", "INDICADOR_3": "Donchian_Middle"}))
    obv = TechnicalIndicators.calculate_obv(df_base)
    df_base = df_base.join(obv.rename(columns={"INDICADOR_1": "OBV"}))
    mfi = TechnicalIndicators.calculate_mfi(df_base)
    df_base = df_base.join(mfi.rename(columns={"INDICADOR_1": "MFI"}))
    vwap = TechnicalIndicators.calculate_vwap(df_base)
    df_base = df_base.join(vwap.rename(columns={"INDICADOR_1": "VWAP"}))
    fib_rolling = TechnicalIndicators.calculate_rolling_fibonacci(df_base, period=144)
    df_base = df_base.join(fib_rolling)
    df_base.dropna(inplace=True)
    df_base['Puntuacion'] = 0.0
    df_base.to_csv(output_path)
    return True

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    for nombre, (inicio, fin) in PERIODOS.items():
        output_path = os.path.join(DATA_DIR, f"dataset_{nombre}.csv")
        
        if not os.path.exists(output_path):
            exito = generar_dataset_mes(nombre, inicio, fin, output_path)
            if exito:
                time.sleep(2) # Cortesía para Binance
        else:
            print(f"\nEl archivo {nombre} ya existe en {output_path}. Saltando descarga.")

if __name__ == "__main__":
    main()