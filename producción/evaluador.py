import pandas as pd
import numpy as np
import os
import sys
import joblib

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from sb3_contrib import RecurrentPPO
from envs.trading_env_v2 import TradingEnv
from utils.results_manager import ResultadosManager

PERIODOS_NOMBRES = [
    "1_Marzo_2020_Covid",
    "2_Febrero_2021_Bull",
    "3_Julio_2021_Lateral",
    "4_Junio_2022_Bear",
    "5_Noviembre_2022_Cambio",
    "6_Agosto_2023_Volatilidad",
    "7_Marzo_2024_BullReciente"
]

CANDIDATOS = [
    ("checkpoint_150000", os.path.join(BASE_DIR, "models", "checkpoints", "LSTM_INDICADORESFINAL_LRDECAY(RecurrentPPO87_SEMILLA42)_150000_steps.zip")),
    ("checkpoint_425000", os.path.join(BASE_DIR, "models", "checkpoints", "LSTM_INDICADORESFINAL_LRDECAY(RecurrentPPO87_SEMILLA42)_425000_steps.zip")),
]

RUTA_SCALER = os.path.join(BASE_DIR, "scalers", "scaler_15m.pkl")
DATA_DIR = os.path.join(BASE_DIR, "data", "evaluaciones_tfg")
ARCHIVO_JSON = os.path.join(BASE_DIR, "resultados_comparativa_tfg.json")

def evaluar_periodo(etiqueta_candidato, nombre_periodo, ruta_csv, model, scaler):

    if not os.path.exists(ruta_csv):
        return

    df = pd.read_csv(ruta_csv, index_col=0, parse_dates=True)
    # Scaler fijo de producción, NO uno reajustado local a este periodo.
    env = TradingEnv(df, scaler=scaler, objetivo="USD")

    obs, _ = env.reset()
    lstm_state = None
    episode_starts = np.ones((1,), dtype=bool)
    episode_reward = 0.0

    terminated = False
    truncated = False

    while not terminated and not truncated:
        action, lstm_state = model.predict(obs, state=lstm_state, episode_start=episode_starts, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        episode_starts = np.zeros((1,), dtype=bool)

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
            print(f"No se encuentra el candidato '{etiqueta}' en {ruta_modelo}. Saltando.")
            continue

        print(f"\nCargando candidato: {etiqueta} ({ruta_modelo})")
        model = RecurrentPPO.load(ruta_modelo)

        rois_periodo = {}
        for nombre_periodo in PERIODOS_NOMBRES:
            ruta_csv = os.path.join(DATA_DIR, f"dataset_{nombre_periodo}.csv")
            roi = evaluar_periodo(etiqueta, nombre_periodo, ruta_csv, model, scaler)
            rois_periodo[nombre_periodo] = roi

        resumen[etiqueta] = rois_periodo

if __name__ == "__main__":
    main()