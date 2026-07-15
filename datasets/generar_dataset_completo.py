import pandas as pd
import numpy as np
import os
from utils.klines_example import klines_example_time_tfg
from utils.indicators_final import TechnicalIndicators

SYMBOL = "BTCUSDT"
INTERVAL = "15m"
START_STR = "2021-01-01"
END_STR = "2024-12-31"

def generar_dataset_metricas():
    raw_data = klines_example_time_tfg(SYMBOL, START_STR, END_STR, INTERVAL)
    if not raw_data:
        print("Error")
        return
    columnas = [
        "Open Time", "Open", "High", "Low", "Close", "Volume",
        "Close Time", "Quote Volume", "Number of Trades",
        "Taker Buy Base", "Taker Buy Quote", "Ignore"
    ]
    df = pd.DataFrame(raw_data, columns=columnas)

    numericas = ["Open", "High", "Low", "Close", "Volume"]
    df[numericas] = df[numericas].apply(pd.to_numeric, axis=1)
    df["Time"] = pd.to_datetime(df["Open Time"], unit='ms')
    df = df.set_index("Time").sort_index()
    df_base = df[["Open", "High", "Low", "Close", "Volume"]].copy()

    # MÉTRICAS
    # SMA, EMA, MACD, Ichimoku, SAR

    #SMA (20,80)
    sma = TechnicalIndicators.calculate_sma(df_base, period_1=20, period_2=80)
    df_base = df_base.join(sma.rename(columns={"INDICADOR_1": "SMA_20", "INDICADOR_2": "SMA_80"}))

    #EMA (20,80)
    ema = TechnicalIndicators.calculate_ema(df_base, period_1=20, period_2=80)
    df_base = df_base.join(ema.rename(columns={"INDICADOR_1": "EMA_20", "INDICADOR_2": "EMA_80"}))

    #MACD
    macd = TechnicalIndicators.calculate_macd(df_base)
    df_base = df_base.join(macd.rename(columns={"INDICADOR_1": "MACD", "INDICADOR_2": "MACD_Signal"}))

    #Ichimoku
    ichimoku = TechnicalIndicators.calculate_ichimoku(df_base)
    df_base = df_base.join(ichimoku.rename(columns={"INDICADOR_1": "Ichi_Tenkan", "INDICADOR_2": "Ichi_Kijun", 
        "INDICADOR_3": "Ichi_SenkouA", "INDICADOR_4": "Ichi_SenkouB", "INDICADOR_5": "Ichi_Lagging"
    }))

    #SAR
    sar = TechnicalIndicators.calculate_sar(df_base)
    df_base = df_base.join(sar.rename(columns={"INDICADOR_1": "SAR"}))

    #Supertrend
    supertrend = TechnicalIndicators.calculate_supertrend(df_base)
    df_base = df_base.join(supertrend.rename(columns={"INDICADOR_1": "Supertrend_Upper", "INDICADOR_2": "Supertrend_Lower"}))

    #ADX
    adx = TechnicalIndicators.calculate_adx(df_base)
    df_base = df_base.join(adx.rename(columns={"INDICADOR_1": "ADX_Pos", "INDICADOR_2": "ADX_Neg", "INDICADOR_3": "ADX_Main"}))

    #---MOMENTUM ---
    #RSI, Estocástico, CCI, Williams %R, CMO

    # RSI
    rsi = TechnicalIndicators.calculate_rsi(df_base)
    df_base = df_base.join(rsi.rename(columns={"INDICADOR_1": "RSI_14", "INDICADOR_2": "RSI_28"}))

    # Stochastic
    stoch = TechnicalIndicators.calculate_stochastic_oscillator(df_base)
    df_base = df_base.join(stoch.rename(columns={"INDICADOR_1": "Stoch_K", "INDICADOR_2": "Stoch_D"}))
    
    # CCI
    cci = TechnicalIndicators.calculate_cci(df_base)
    df_base = df_base.join(cci.rename(columns={"INDICADOR_1": "CCI"}))

    # Williams %R
    williams = TechnicalIndicators.calculate_williams_r(df_base)
    df_base = df_base.join(williams.rename(columns={"INDICADOR_1": "Williams"}))

    # CMO
    cmo = TechnicalIndicators.calculate_cmo(df_base)
    df_base = df_base.join(cmo.rename(columns={"INDICADOR_1": "CMO"}))

    # ---VOLATILIDAD ---
    # Bollinger Bands, ATR, Keltner Channels,Donchian Channel
    # Bollinger Bands
    bollinger = TechnicalIndicators.calculate_bollinger_bands(df_base)
    df_base = df_base.join(bollinger.rename(columns={"INDICADOR_1": "Bollinger_Middle", "INDICADOR_2": "Bollinger_Upper", "INDICADOR_3": "Bollinger_Lower"}))

    # ATR
    atr = TechnicalIndicators.calculate_atr(df_base)
    df_base = df_base.join(atr.rename(columns={"INDICADOR_1": "ATR"}))

    # Keltner Channels
    keltner = TechnicalIndicators.calculate_keltner_channels(df_base)
    df_base = df_base.join(keltner.rename(columns={"INDICADOR_1": "Keltner_Upper", "INDICADOR_2": "Keltner_Middle", "INDICADOR_3": "Keltner_Lower"}))

    # Donchian Channel
    donchian = TechnicalIndicators.calculate_donchian_channel(df_base)
    df_base = df_base.join(donchian.rename(columns={"INDICADOR_1": "Donchian_Upper", "INDICADOR_2": "Donchian_Lower", "INDICADOR_3": "Donchian_Middle"}))

    # ---VOLUMEN ---
    # OBV, MFI, VWAP
    # OBV
    obv = TechnicalIndicators.calculate_obv(df_base)
    df_base = df_base.join(obv.rename(columns={"INDICADOR_1": "OBV"}))

    # MFI
    mfi = TechnicalIndicators.calculate_mfi(df_base)
    df_base = df_base.join(mfi.rename(columns={"INDICADOR_1": "MFI"}))

    # VWAP
    vwap = TechnicalIndicators.calculate_vwap(df_base)
    df_base = df_base.join(vwap.rename(columns={"INDICADOR_1": "VWAP"}))

    # Fibonacci Rolling
    fib_rolling = TechnicalIndicators.calculate_rolling_fibonacci(df_base, period= 144)
    df_base = df_base.join(fib_rolling)

    df_base.dropna(inplace=True)
    if not os.path.exists("data"):
        os.makedirs("data")
    output_path = "data/dataset_train_completoATR15minutos2021_24.csv"
    df_base.to_csv(output_path)
    print(df_base.columns.tolist())

if __name__ == "__main__":
    generar_dataset_metricas()