import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import joblib
import warnings
from sb3_contrib import RecurrentPPO
from utils.config_params import get_api_secret_test
warnings.filterwarnings('ignore')
from compraVenta import crear_cliente, API_KEY
from utils.indicators_final import TechnicalIndicators

RUTA_MODELO = "../models/checkpoints/LSTM_INDICADORESFINAL_LRDECAY(RecurrentPPO87_SEMILLA42)_425000_steps.zip"
RUTA_SCALER = "../scalers/scaler_15m.pkl"

def descargar_velas_binance(client, symbol="BTCUSDT", interval="15m", limit=250):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    
    df = pd.DataFrame(klines, columns=[
        'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close Time', 'Quote Asset Volume', 'Number of Trades',
        'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
    ])
    numericas = ['Open', 'High', 'Low', 'Close', 'Volume']
    df[numericas] = df[numericas].astype(float)
    df['Time'] = pd.to_datetime(df['Open Time'], unit='ms')
    df.set_index('Time', inplace=True)
    
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]

def preparar_secuencia_lstm(df_crudo):
    sma = TechnicalIndicators.calculate_sma(df_crudo, period_1=20, period_2=80)
    df_crudo = df_crudo.join(sma.rename(columns={"INDICADOR_1": "SMA_20", "INDICADOR_2": "SMA_80"}))
    
    ema = TechnicalIndicators.calculate_ema(df_crudo, period_1=20, period_2=80)
    df_crudo = df_crudo.join(ema.rename(columns={"INDICADOR_1": "EMA_20", "INDICADOR_2": "EMA_80"}))
    
    macd = TechnicalIndicators.calculate_macd(df_crudo)
    df_crudo = df_crudo.join(macd.rename(columns={"INDICADOR_1": "MACD", "INDICADOR_2": "MACD_Signal"}))
    
    ichimoku = TechnicalIndicators.calculate_ichimoku(df_crudo)
    df_crudo = df_crudo.join(ichimoku.rename(columns={"INDICADOR_1": "Ichi_Tenkan", "INDICADOR_2": "Ichi_Kijun",
                                                    "INDICADOR_3": "Ichi_SenkouA", "INDICADOR_4": "Ichi_SenkouB", "INDICADOR_5": "Ichi_Lagging"}))
    
    sar = TechnicalIndicators.calculate_sar(df_crudo)
    df_crudo = df_crudo.join(sar.rename(columns={"INDICADOR_1": "SAR"}))
    
    supertrend = TechnicalIndicators.calculate_supertrend(df_crudo)
    df_crudo = df_crudo.join(supertrend.rename(columns={"INDICADOR_1": "Supertrend_Upper", "INDICADOR_2": "Supertrend_Lower"}))
    
    adx = TechnicalIndicators.calculate_adx(df_crudo)
    df_crudo = df_crudo.join(adx.rename(columns={"INDICADOR_1": "ADX_Pos", "INDICADOR_2": "ADX_Neg", "INDICADOR_3": "ADX_Main"}))
    
    rsi = TechnicalIndicators.calculate_rsi(df_crudo)
    df_crudo = df_crudo.join(rsi.rename(columns={"INDICADOR_1": "RSI_14", "INDICADOR_2": "RSI_28"}))
    
    stoch = TechnicalIndicators.calculate_stochastic_oscillator(df_crudo)
    df_crudo = df_crudo.join(stoch.rename(columns={"INDICADOR_1": "Stoch_K", "INDICADOR_2": "Stoch_D"}))
    
    cci = TechnicalIndicators.calculate_cci(df_crudo)
    df_crudo = df_crudo.join(cci.rename(columns={"INDICADOR_1": "CCI"}))
    
    williams = TechnicalIndicators.calculate_williams_r(df_crudo)
    df_crudo = df_crudo.join(williams.rename(columns={"INDICADOR_1": "Williams"}))
    
    cmo = TechnicalIndicators.calculate_cmo(df_crudo)
    df_crudo = df_crudo.join(cmo.rename(columns={"INDICADOR_1": "CMO"}))
    
    bollinger = TechnicalIndicators.calculate_bollinger_bands(df_crudo)
    df_crudo = df_crudo.join(bollinger.rename(columns={"INDICADOR_1": "Bollinger_Middle", "INDICADOR_2": "Bollinger_Upper", "INDICADOR_3": "Bollinger_Lower"}))
    
    atr = TechnicalIndicators.calculate_atr(df_crudo)
    df_crudo = df_crudo.join(atr.rename(columns={"INDICADOR_1": "ATR"}))
    
    keltner = TechnicalIndicators.calculate_keltner_channels(df_crudo)
    df_crudo = df_crudo.join(keltner.rename(columns={"INDICADOR_1": "Keltner_Upper", "INDICADOR_2": "Keltner_Middle", "INDICADOR_3": "Keltner_Lower"}))
    
    donchian = TechnicalIndicators.calculate_donchian_channel(df_crudo)
    df_crudo = df_crudo.join(donchian.rename(columns={"INDICADOR_1": "Donchian_Upper", "INDICADOR_2": "Donchian_Lower", "INDICADOR_3": "Donchian_Middle"}))
    
    obv = TechnicalIndicators.calculate_obv(df_crudo)
    df_crudo = df_crudo.join(obv.rename(columns={"INDICADOR_1": "OBV"}))
    
    mfi = TechnicalIndicators.calculate_mfi(df_crudo)
    df_crudo = df_crudo.join(mfi.rename(columns={"INDICADOR_1": "MFI"}))
    
    vwap = TechnicalIndicators.calculate_vwap(df_crudo)
    df_crudo = df_crudo.join(vwap.rename(columns={"INDICADOR_1": "VWAP"}))
    
    fib_rolling = TechnicalIndicators.calculate_rolling_fibonacci(df_crudo, period=144)
    df_crudo = df_crudo.join(fib_rolling)
    
    df_crudo.dropna(inplace=True)
    
    columnas_base = ["Open", "High", "Low", "Close", "Volume"]
    if "Puntuacion" in df_crudo.columns:
        columnas_base.append("Puntuacion")
        
    indicadores = df_crudo.drop(columns=columnas_base, errors='ignore')
    
    try:
        scaler = joblib.load(RUTA_SCALER)
    except FileNotFoundError:
        print("Error: No se encuentra el scaler en la ruta especificada.")
        sys.exit(1)
        
    indicadores_scaled = scaler.transform(indicadores)
    portfolio_falso = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    todas_las_observaciones = []
    for step_market in indicadores_scaled:
        obs_completa = np.concatenate([step_market, portfolio_falso])
        todas_las_observaciones.append(obs_completa)
        
    return np.array(todas_las_observaciones, dtype=np.float32)

def consultar_modelo():
    client = crear_cliente(api_key=API_KEY, api_secret=get_api_secret_test(), is_testnet=True)
    df_mercado = descargar_velas_binance(client)
    secuencia_obs = preparar_secuencia_lstm(df_mercado)
    
    try:
        model = RecurrentPPO.load(RUTA_MODELO)
    except Exception as e:
        sys.exit(1)
    
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    accion_final = None
    
    for step_idx, obs in enumerate(secuencia_obs):
        obs_batch = np.array([obs])
        accion_predicha, lstm_states = model.predict(
            obs_batch, 
            state=lstm_states, 
            episode_start=episode_starts, 
            deterministic=True
        )
        episode_starts = np.zeros((1,), dtype=bool) 
        accion_final = accion_predicha[0]
    
    tipo_operacion = accion_final[0]
    cantidad_operacion = np.clip(accion_final[1], 0.01, 1.0) if len(accion_final) > 1 else 1.0
    
    print("DECISION DEL MODELO")
    if tipo_operacion > 0.33:
        print(f"VEREDICTO: COMPRAR AHORA")
        print(f"   Fuerza: {tipo_operacion:.2f} | Cantidad: {cantidad_operacion:.2%}")
    elif tipo_operacion < -0.33:
        print(f"VEREDICTO: VENDER AHORA")
        print(f"   Fuerza: {tipo_operacion:.2f} | Cantidad: {cantidad_operacion:.2%}")
    else:
        print(f"VEREDICTO: ESPERAR")
        print(f"   Señal plana: {tipo_operacion:.2f}")

if __name__ == "__main__":
    consultar_modelo()