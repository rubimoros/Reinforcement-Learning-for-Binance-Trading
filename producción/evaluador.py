import pandas as pd
import numpy as np
import os
import sys
import joblib

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

from envs.trading_env_v2 import TradingEnv
from utils.results_manager import ResultadosManager
from utils.wrappers import HistoryWrapperCNN, HistoryWrapperLSTM

PERIODOS_NOMBRES = [
    "1_Marzo_2020_Covid",
    "2_Febrero_2021_Bull",
    "3_Julio_2021_Lateral",
    "4_Junio_2022_Bear",
    "5_Noviembre_2022_Cambio",
    "6_Agosto_2023_Volatilidad",
    "7_Marzo_2024_BullReciente"
]

# Lista de candidatos a evaluar
CANDIDATOS = [
    # ---------- LSTM (4) ----------
    ("LSTM_42_175k", os.path.join(BASE_DIR, "models", "checkpoints_lstm42",
        "LSTM_INDICADORESFINAL_LRDECAY(RecurrentPPO88_SEMILLA42)_175000_steps.zip")),

    ("LSTM_7_500k", os.path.join(BASE_DIR, "models", "checkpoints_lstm7",
        "LSTM_INDICADORESFINAL_LRDECAY(RecurrentPPO89_SEMILLA7)_500000_steps.zip")),

    ("LSTM_2024_250k", os.path.join(BASE_DIR, "models", "checkpoints_lstm2024",
        "LSTM_INDICADORESFINAL_LRDECAY(RecurrentPPO90_SEMILLA2024)_250000_steps.zip")),

    ("LSTM_123_100k", os.path.join(BASE_DIR, "models", "checkpoints_lstm123",
        "LSTM_INDICADORESFINAL_LRDECAY(RecurrentPPO91_SEMILLA123)_100000_steps.zip")),

    # ---------- CNN (3 -- semilla 123 excluida, sin candidato válido) ----------
    ("PPO_CNN_42_50k", os.path.join(BASE_DIR, "models", "checkpoints_cnn42",
        "CNN_INDICADORESFINAL(PPO_24_SEMILLA42)_50000_steps.zip")),

    ("PPO_CNN_7_150k", os.path.join(BASE_DIR, "models", "checkpoints_cnn7",
        "PPO_CNN_INDICADORESFINAL(PPO_26_SEMILLA7)_150000_steps.zip")),

    ("PPO_CNN_2024_75k", os.path.join(BASE_DIR, "models", "checkpoints_cnn2024",
        "PPO_CNN_INDICADORESFINAL(PPO_27_SEMILLA2024)_75000_steps.zip")),

    # ---------- CNN-LSTM (4) ----------
    ("CNNLSTM_42_375k", os.path.join(BASE_DIR, "models", "checkpoints_cnn_lstm42",
        "RECURRENT_PPO_CNN_LSTM_INDICADORESFINAL(Recurrent_PPO_91_SEMILLA42)_375000_steps.zip")),

    ("CNNLSTM_7_75k", os.path.join(BASE_DIR, "models", "checkpoints_cnn_lstm7",
        "RECURRENT_PPO_CNN_LSTM_INDICADORESFINAL(Recurrent_PPO_92_SEMILLA7)_75000_steps.zip")),

    ("CNNLSTM_2024_475k", os.path.join(BASE_DIR, "models", "checkpoints_cnn_lstm2024",
        "RECURRENT_PPO_CNN_LSTM_INDICADORESFINAL(Recurrent_PPO_95_SEMILLA2024)_475000_steps.zip")),

    ("CNNLSTM_123_450k", os.path.join(BASE_DIR, "models", "checkpoints_cnn_lstm123",
        "RECURRENT_PPO_CNN_LSTM_INDICADORESFINAL(Recurrent_PPO_96_SEMILLA123)_450000_steps.zip")),
]

RUTA_SCALER = os.path.join(BASE_DIR, "scalers", "scaler_15m.pkl")
DATA_DIR = os.path.join(BASE_DIR, "data", "evaluaciones_tfg")
ARCHIVO_JSON = os.path.join(BASE_DIR, "resultados_comparativa_tfg.json")

def evaluar_periodo(etiqueta_candidato, nombre_periodo, ruta_csv, model, scaler):

    if not os.path.exists(ruta_csv):
        print(f"   [!] Archivo no encontrado: {ruta_csv}")
        return None

    df = pd.read_csv(ruta_csv, index_col=0, parse_dates=True)
    
    # Entorno base con scaler fijo de producción
    env = TradingEnv(df, scaler=scaler, objetivo="USD")

    # Wrapper según la arquitectura
    if "CNNLSTM" in etiqueta_candidato:
        env = HistoryWrapperLSTM(env, window_size=50)
        is_recurrent = True
    elif "CNN" in etiqueta_candidato:
        env = HistoryWrapperCNN(env, window_size=50)
        is_recurrent = False
    else:
        # LSTM puro no usa wrapper de historial extra, pero es recurrente
        is_recurrent = True

    obs, _ = env.reset()
    lstm_state = None
    episode_starts = np.ones((1,), dtype=bool)
    episode_reward = 0.0

    terminated = False
    truncated = False

    while not terminated and not truncated:
        if is_recurrent:
            action, lstm_state = model.predict(obs, state=lstm_state, episode_start=episode_starts, deterministic=True)
        else:
            action, _ = model.predict(obs, deterministic=True)
            
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        episode_starts = np.zeros((1,), dtype=bool)

    # Nombre único: candidato + periodo
    ResultadosManager.guardar(
        modelo_nombre=f"TFG_{etiqueta_candidato}_{nombre_periodo}",
        roi_pct=info['profit_pct'],
        reward_acumulado=episode_reward,
        dataset=f"Evaluación TFG: {nombre_periodo} (candidato: {etiqueta_candidato})",
        metricas_adicionales=info,
        archivo_destino=ARCHIVO_JSON
    )
    return info['profit_pct']


def main():
    if not os.path.exists(RUTA_SCALER):
        print(f"No se encuentra el scaler en {RUTA_SCALER}. Debe ser el mismo que usa producción.")
        return

    print(f"Cargando scaler fijo de producción: {RUTA_SCALER}")
    scaler = joblib.load(RUTA_SCALER)

    resumen = {}

    for etiqueta, ruta_modelo in CANDIDATOS:
        if not os.path.exists(ruta_modelo):
            print(f"No se encuentra el candidato '{etiqueta}' en {ruta_modelo}. Verifica las rutas de las carpetas.")
            continue

        print(f"\nCargando candidato: {etiqueta}...")
        
        # Carga dinámica del algoritmo según la arquitectura
        if "CNN" in etiqueta and "LSTM" not in etiqueta:
            model = PPO.load(ruta_modelo)
        else:
            model = RecurrentPPO.load(ruta_modelo)

        rois_periodo = {}
        for nombre_periodo in PERIODOS_NOMBRES:
            ruta_csv = os.path.join(DATA_DIR, f"dataset_{nombre_periodo}.csv")
            roi = evaluar_periodo(etiqueta, nombre_periodo, ruta_csv, model, scaler)
            rois_periodo[nombre_periodo] = roi

        resumen[etiqueta] = rois_periodo

    # Tabla resumen en terminal
    print("\n" + "=" * 105)
    print("RESUMEN: ROI% por candidato y periodo de Test")
    print("=" * 105)
    header = f"{'CANDIDATO':<20}" + "".join(f"{p.split('_')[1]:>12}" for p in PERIODOS_NOMBRES)
    print(header)
    print("-" * 105)
    for etiqueta, rois_periodo in resumen.items():
        fila = f"{etiqueta:<20}"
        for nombre_periodo in PERIODOS_NOMBRES:
            valor = rois_periodo.get(nombre_periodo)
            fila += f"{valor:>12.2f}" if valor is not None else f"{'N/A':>12}"
        print(fila)
    print("=" * 105)
    print(f"\nResultados detallados guardados para análisis en: {ARCHIVO_JSON}")

if __name__ == "__main__":
    main()
