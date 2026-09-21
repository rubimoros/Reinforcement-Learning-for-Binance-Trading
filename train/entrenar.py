import sys
import os
import pandas as pd
import numpy as np
from sb3_contrib import RecurrentPPO as PPO
from typing import Callable
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from datetime import datetime
import joblib
import torch
import random

from utils.results_manager import ResultadosManager
from utils.metricas_comunes import evaluar_modelo, comparar_con_buy_and_hold
from utils.callbacks import SeleccionPorMetricaCallback
from envs.trading_env_v2 import TradingEnv

RUTA_DATOS = "data/dataset_puntuaciones_picos_valles_atr_15minutos2021_24.csv"
RUTA_SCALER = "scalers/scaler_15m.pkl"  # generado aparte; los agentes SOLO lo cargan, nunca lo reajustan
TIME_STEPS = 500000  # IMPORTANTE: mismo valor en los 3 scripts para que la comparativa sea homogénea
CHECKPOINT_FREQ = 25000        # snapshot periódico -> CHECKPOINTS_DIR (fichero de inspección, para la memoria)
SELECCION_FREQ = 25000         # frecuencia de evaluación contra validación para elegir el mejor
SELECCION_METRICA = "profit_pct"   # variable controlada: misma métrica de selección en las 3 arquitecturas
NOMBRE_EXPERIMENTO = "LSTM_INDICADORESFINAL_LRDECAY(RecurrentPPO91_SEMILLA123)"
NOTAS_EXPERIMENTO = "impacto=0.05, vago=-0.05, sell=-0.1. LR decay + selección de checkpoint por ROI en validación (sin fuga a test)."
ARQUITECTURA = "LSTM"
MODELO_DIR = "models"
CHECKPOINTS_DIR = os.path.join(MODELO_DIR, "checkpoints_lstm123")           # TODOS los steps periódicos (fichero aparte)
BEST_MODEL_DIR = os.path.join(MODELO_DIR, "best_model_lstm_comparativa123")    # el elegido por validación (otro fichero aparte)
LOGS_DIR = "logs"
OBJETIVO = "USD"

SEMILLA = 123
random.seed(SEMILLA)
np.random.seed(SEMILLA)
torch.manual_seed(SEMILLA)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if len(sys.argv) > 1:
    objetivo = sys.argv[1].upper()
    if objetivo not in ["USD", "BTC"]:
        print("Objetivo incorrecto. Debe ser USD o BTC")
        sys.exit()
else:
    objetivo = OBJETIVO


def curva_de_aprendizaje(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def main():
    os.makedirs(MODELO_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    dataset = pd.read_csv(RUTA_DATOS, index_col=0, parse_dates=True)

    n = len(dataset)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    df_train = dataset.iloc[:train_end].copy()
    df_val = dataset.iloc[train_end:val_end].copy()
    df_test = dataset.iloc[val_end:].copy()

    df_val['Puntuacion'] = 0.0
    df_test['Puntuacion'] = 0.0
    if not os.path.exists(RUTA_SCALER):
        raise FileNotFoundError(
            f"No se encontró el scaler en '{RUTA_SCALER}'. Genéralo primero con "
            f"el script dedicado a ello antes de entrenar; este script no debe "
            f"crear un scaler nuevo."
        )
    scaler_maestro = joblib.load(RUTA_SCALER)

    env_train = Monitor(TradingEnv(df_train, scaler=scaler_maestro, objetivo=OBJETIVO),
                         filename=os.path.join(LOGS_DIR, "train_monitor.csv"))
    env_val = TradingEnv(df_val, scaler=scaler_maestro, objetivo=OBJETIVO)
    env_test = TradingEnv(df_test, scaler=scaler_maestro, objetivo=OBJETIVO)

    env_train_vec = DummyVecEnv([lambda: env_train])
    env_train_vec.seed(SEMILLA)
    seleccion_callback = SeleccionPorMetricaCallback(
        env_val=env_val,
        ruta_guardado=BEST_MODEL_DIR,
        eval_freq=SELECCION_FREQ,
        n_episodes=5,
        metrica=SELECCION_METRICA,
        recurrente=True
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=CHECKPOINTS_DIR,
        name_prefix=NOMBRE_EXPERIMENTO
    )

    callbacks = CallbackList([seleccion_callback, checkpoint_callback])

    model = PPO(
        policy="MlpLstmPolicy",
        env=env_train_vec,
        learning_rate=curva_de_aprendizaje(3e-4),
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=os.path.join(LOGS_DIR, "tensorboard")
    )

    start_time = datetime.now()
    model.learn(
        total_timesteps=TIME_STEPS,
        callback=callbacks,
        progress_bar=True
    )
    end_time = datetime.now()

    tiempo_entrenamiento_seg = (end_time - start_time).total_seconds()
    print(f"\nEntrenamiento completado en {tiempo_entrenamiento_seg:.2f} segundos "
          f"({tiempo_entrenamiento_seg / 60:.2f} minutos)")

    model_path = os.path.join(MODELO_DIR, f"{NOMBRE_EXPERIMENTO}_final.zip")
    model.save(model_path)
    print(f"Modelo final guardado en: {model_path}")
    ruta_best = os.path.join(BEST_MODEL_DIR, "best_model.zip")
    ruta_evaluar = ruta_best if os.path.exists(ruta_best) else model_path
    if not os.path.exists(ruta_best):
        print("No se generó best_model.zip (SELECCION_FREQ mayor que TIME_STEPS "
              "o validación vacía). Se evalúa el modelo _final como alternativa.")

    modelo_final = PPO.load(ruta_evaluar)
    metricas_test = evaluar_modelo(modelo_final, env_test, n_episodes=10, recurrente=True)
    bh_metrics = comparar_con_buy_and_hold(env_test)

    print(f"EVALUACIÓN EN TEST DEL CANDIDATO SELECCIONADO ({os.path.basename(ruta_evaluar)})")
    print("=" * 70)
    print(f"   ROI Agente:            {metricas_test['roi_pct']:.2f}%")
    print(f"   ROI Buy & Hold:        {bh_metrics['profit_pct']:.2f}%")
    print(f"   Alpha:                 {metricas_test['roi_pct'] - bh_metrics['profit_pct']:.2f}%")
    print(f"   Sharpe Ratio:          {metricas_test.get('sharpe_ratio', 0.0):.3f}")
    print(f"   Sortino Ratio:         {metricas_test.get('sortino_ratio', 0.0):.3f}")
    print(f"   Max Drawdown:          {metricas_test.get('max_drawdown', 0.0):.2f}%")
    print(f"   Win Rate:              {metricas_test.get('win_rate', 0.0):.2f}%")
    print(f"   Profit Factor:         {metricas_test.get('profit_factor', 0.0):.3f}")
    print(f"   Compras/Ventas/Holds:  {metricas_test.get('total_buys', 0):.0f} / "
          f"{metricas_test.get('total_sells', 0):.0f} / {metricas_test.get('total_holds', 0):.0f}")
    print("=" * 70)

    ResultadosManager.guardar(
        modelo_nombre=NOMBRE_EXPERIMENTO,
        roi_pct=metricas_test['roi_pct'],
        reward_acumulado=0.0,
        balance_final=10000 * (1 + metricas_test['roi_pct'] / 100),
        dataset=f"{RUTA_DATOS} (TradingEnv_v2, LR decay + selección por validación)",
        notas=NOTAS_EXPERIMENTO,
        metricas_adicionales=metricas_test,
        tiempo_entrenamiento_seg=tiempo_entrenamiento_seg,
        semilla=SEMILLA,
        arquitectura=ARQUITECTURA
    )


if __name__ == "__main__":
    main()
