import sys
import os
import pandas as pd
import numpy as np
import torch as th
import torch.nn as nn
import gymnasium as gym
import random
import joblib
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from datetime import datetime

from utils.results_manager import ResultadosManager
from utils.metricas_comunes import evaluar_modelo, comparar_con_buy_and_hold
from utils.callbacks import SeleccionPorMetricaCallback
from utils.wrappers import HistoryWrapperCNN
from envs.trading_env_v2 import TradingEnv

RUTA_DATOS = "data/dataset_puntuaciones_picos_valles_atr_15minutos2021_24.csv"
RUTA_SCALER = "scalers/scaler_15m.pkl"
TIME_STEPS = 500000  # IMPORTANTE: mismo valor en los 3 scripts para que la comparativa sea homogénea
CHECKPOINT_FREQ = 25000        # snapshot periódico -> CHECKPOINTS_DIR (fichero de inspección, para la memoria)
SELECCION_FREQ = 25000          # frecuencia de evaluación contra validación para elegir el mejor
SELECCION_METRICA = "profit_pct"   # variable controlada: misma métrica de selección en las 3 arquitecturas
NOMBRE_EXPERIMENTO = "PPO_CNN_INDICADORESFINAL(PPO_28_SEMILLA123)"
NOTAS_EXPERIMENTO = "impacto=0.05, vago=-0.05, sell=-0.1 selección de checkpoint por ROI en validación (sin fuga a test)."
ARQUITECTURA = "CNN"
MODELO_DIR = "models"
CHECKPOINTS_DIR = os.path.join(MODELO_DIR, "checkpoints_cnn123")               # TODOS los steps periódicos
BEST_MODEL_DIR = os.path.join(MODELO_DIR, "best_model_cnn_comparativa123")       # el elegido por validación
LOGS_DIR = "logs"
OBJETIVO = "USD"
SEMILLA = 123

random.seed(SEMILLA)
np.random.seed(SEMILLA)
th.manual_seed(SEMILLA)
th.backends.cudnn.deterministic = True
th.backends.cudnn.benchmark = False

if len(sys.argv) > 1:
    objetivo_arg = sys.argv[1].upper()
    if objetivo_arg not in ["USD", "BTC"]:
        print("Objetivo incorrecto. Debe ser USD o BTC")
        sys.exit()
    OBJETIVO = objetivo_arg


class TradingCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        window_size, n_features = observation_space.shape
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=n_features, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten()
        )

        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            sample_permuted = sample.permute(0, 2, 1)
            n_flatten = self.cnn(sample_permuted).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = observations.permute(0, 2, 1)
        x = self.cnn(x)
        return self.linear(x)


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

    if os.path.exists(RUTA_SCALER):
        print(f"Reutilizando scaler ya existente: {RUTA_SCALER}")
        scaler_maestro = joblib.load(RUTA_SCALER)
    else:
        cols_a_ignorar = ["Open", "High", "Low", "Close", "Volume", "Puntuacion"]
        indicadores_train = df_train.drop(columns=cols_a_ignorar, errors='ignore')
        scaler_maestro = StandardScaler()
        scaler_maestro.fit(indicadores_train)
        os.makedirs("scalers", exist_ok=True)
        joblib.dump(scaler_maestro, RUTA_SCALER)

    env_train_base = TradingEnv(df_train, scaler=scaler_maestro, objetivo=OBJETIVO)
    env_val_base = TradingEnv(df_val, scaler=scaler_maestro, objetivo=OBJETIVO)
    env_test_base = TradingEnv(df_test, scaler=scaler_maestro, objetivo=OBJETIVO, verbose=True)

    env_train = HistoryWrapperCNN(env_train_base, window_size=50)
    env_val = HistoryWrapperCNN(env_val_base, window_size=50)
    env_test = HistoryWrapperCNN(env_test_base, window_size=50)

    env_train_vec = DummyVecEnv([lambda: env_train])
    env_train_vec.seed(SEMILLA)
    seleccion_callback = SeleccionPorMetricaCallback(
        env_val=env_val,
        ruta_guardado=BEST_MODEL_DIR,
        eval_freq=SELECCION_FREQ,
        n_episodes=5,
        metrica=SELECCION_METRICA,
        recurrente=False  # PPO estándar (CnnPolicy), sin estado LSTM
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=CHECKPOINTS_DIR,
        name_prefix=NOMBRE_EXPERIMENTO
    )

    callbacks = CallbackList([seleccion_callback, checkpoint_callback])

    policy_kwargs = dict(
        features_extractor_class=TradingCNN,
        features_extractor_kwargs=dict(features_dim=256),
    )

    model = PPO(
        policy="CnnPolicy",
        env=env_train_vec,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
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
        tensorboard_log=os.path.join(LOGS_DIR, "tensorboard_cnn")
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
    ruta_best = os.path.join(BEST_MODEL_DIR, "best_model.zip")
    ruta_evaluar = ruta_best if os.path.exists(ruta_best) else model_path
    if not os.path.exists(ruta_best):
        print("No se generó best_model.zip. Se evalúa el modelo _final como alternativa.")

    modelo_final = PPO.load(ruta_evaluar)
    metricas_test = evaluar_modelo(modelo_final, env_test, n_episodes=10, recurrente=False)
    bh_metrics = comparar_con_buy_and_hold(env_test)

    print("\n" + "=" * 60)
    print(f"EVALUACIÓN EN TEST DEL CANDIDATO SELECCIONADO ({os.path.basename(ruta_evaluar)})")
    print("=" * 60)
    print(f"   ROI Agente:            {metricas_test['roi_pct']:.2f}%")
    print(f"   ROI Buy & Hold:        {bh_metrics['profit_pct']:.2f}%")
    print(f"   Alpha:                 {metricas_test['roi_pct'] - bh_metrics['profit_pct']:.2f}%")
    print(f"   Sharpe Ratio:          {metricas_test.get('sharpe_ratio', 0.0):.3f}")
    print(f"   Sortino Ratio:         {metricas_test.get('sortino_ratio', 0.0):.3f}")
    print(f"   Max Drawdown:          {metricas_test.get('max_drawdown', 0.0):.2f}%")
    print(f"   Win Rate:              {metricas_test.get('win_rate', 0.0):.2f}%")
    print(f"   Profit Factor:         {metricas_test.get('profit_factor', 0.0):.3f}")
    print("=" * 60)

    dataset_nombre = "dataset_pca" if "pca" in RUTA_DATOS.lower() else RUTA_DATOS
    ResultadosManager.guardar(
        modelo_nombre=NOMBRE_EXPERIMENTO,
        roi_pct=metricas_test['roi_pct'],
        reward_acumulado=0.0,
        balance_final=10000 * (1 + metricas_test['roi_pct'] / 100),
        dataset=f"{dataset_nombre} (TradingEnv_v2, selección por validación)",
        notas=NOTAS_EXPERIMENTO,
        metricas_adicionales=metricas_test,
        tiempo_entrenamiento_seg=tiempo_entrenamiento_seg,
        semilla=SEMILLA,
        arquitectura=ARQUITECTURA
    )


if __name__ == "__main__":
    main()
