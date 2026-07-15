import sys
import os
import pandas as pd
import numpy as np
import gymnasium as gym
from collections import deque
import torch as th
import torch.nn as nn
import random
import joblib
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from datetime import datetime
from utils.results_manager import ResultadosManager
from envs.trading_env_v2 import TradingEnv

RUTA_DATOS = "data/dataset_puntuaciones_picos_valles_atr_15minutos2021_24.csv"
RUTA_SCALER = "scalers/scaler_15m.pkl"
TIME_STEPS = 50000
NOMBRE_EXPERIMENTO = "PPO_CNN_INDICADORESFINAL(PPO_22_SEMILLA42)"
NOTAS_EXPERIMENTO = "Fase 3 CNN: INDICADORES PERSISTENTES PARA EL FUTURO, ENTRENAMIENTO PARA PUESTA EN PRODUCCIÓN SCALER Y 2021-2024. impacto=0.05, vago=-0.05, sell=-0.1."
MODELO_DIR = "models"
LOGS_DIR = "logs"
OBJETIVO = "USD"
SEMILLA = 42

random.seed(SEMILLA)
np.random.seed(SEMILLA)
th.manual_seed(SEMILLA)
th.backends.cudnn.deterministic = True
th.backends.cudnn.benchmark = False

# Capturador de terminal
if len(sys.argv) > 1:
    objetivo_arg = sys.argv[1].upper()
    if objetivo_arg not in ["USD", "BTC"]:
        print("Objetivo incorrecto. Debe ser USD o BTC")
        sys.exit()
    OBJETIVO = objetivo_arg
    
class HistoryWrapper(gym.Wrapper):
    """
    Acumula las últimas 'window_size' observaciones.
    Convierte un paso aislado en una "foto" bidimensional.
    """
    def __init__(self, env, window_size=50):
        super().__init__(env)
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        
        # Asumimos que el entorno base devuelve un Box 1D
        base_shape = env.observation_space.shape
        self.n_features = base_shape[0]
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.window_size, self.n_features), 
            dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Llenamos la historia inicial con la primera observación repetida
        for _ in range(self.window_size):
            self.history.append(obs)
        return np.array(self.history, dtype=np.float32), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.history.append(obs)
        return np.array(self.history, dtype=np.float32), reward, terminated, truncated, info

class TradingCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # dimension de entrada: (window_size, n_features)
        window_size, n_features = observation_space.shape
        
        # PyTorch Conv1d espera: (batch_size, channels, length)
        # Convertiremos los features en channels y el window_size en length
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=n_features, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten()
        )
        
        # Calculamos la forma de salida de la CNN automáticamente
        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            # Permutamos de (batch, window, features) a (batch, features, window)
            sample_permuted = sample.permute(0, 2, 1)
            n_flatten = self.cnn(sample_permuted).shape[1]
            
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # observations shape: (batch_size, window_size, n_features)
        # Conv1d espera: (batch_size, n_features, window_size)
        x = observations.permute(0, 2, 1)
        x = self.cnn(x)
        return self.linear(x)

def evaluar_modelo(model, env, n_episodes=10):
    """Evalúa el modelo durante n episodios y retorna métricas COMPLETAS"""
    all_rewards = []
    all_net_worths = []
    all_profits = []
    all_bonus = []
    all_net_worths_btc = []
    all_profits_btc = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            all_bonus.append(info['bonus'])

        final_net_worth = info['net_worth']
        profit_pct = info['profit_pct']
        
        all_rewards.append(episode_reward)
        all_net_worths.append(final_net_worth)
        all_profits.append(profit_pct)
        all_net_worths_btc.append(info.get('net_worth_btc', 0.0))
        all_profits_btc.append(info.get('profit_btc_pct', 0.0))
        
    total_recompensa = np.sum(all_rewards)
    total_bonus = np.sum(all_bonus)
    total_recompensa_mercado = total_recompensa - total_bonus
    
    return {
        'mean_reward': np.mean(all_rewards),
        'mean_net_worth': np.mean(all_net_worths),
        'mean_profit_pct': np.mean(all_profits),
        
        'mean_net_worth_btc': np.mean(all_net_worths_btc),
        'mean_profit_btc_pct': np.mean(all_profits_btc),
        
        'recompensa_neta_mercado': float(total_recompensa_mercado),
        'recompensa_neta_bonus': float(total_bonus),
        
        'total_buys': info['total_buys'],
        'total_sells': info['total_sells'],
        'total_holds': info['total_holds'],
        
        'ventas_ganadoras': info.get('ventas_ganadoras', 0),
        'ventas_perdedoras': info.get('ventas_perdedoras', 0),
        'ratio_ganadoras': info.get('ratio_ganadoras', 0.0),
        'ratio_perdedoras': info.get('ratio_perdedoras', 0.0),
        
        'win_rate': info.get('ratio_ganadoras', 0.0),
        'max_drawdown': info['max_drawdown'],
        'sharpe_ratio': info.get('sharpe_ratio', 0.0),
        
        'profit_factor': info.get('profit_factor', 0.0),
        'holding_time': info.get('holding_time', 0.0),
        
        'mejor_trade_pct': info['mejor_trade_pct'] * 100,
        'peor_trade_pct': info['peor_trade_pct'] * 100,
        'fecha_mejor_trade': info.get('fecha_mejor_trade', None),
        'fecha_peor_trade': info.get('fecha_peor_trade', None),
    }

def main():
    os.makedirs(MODELO_DIR, exist_ok=True)
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

    # Scaler único: ajustado solo con train, reutilizado en val/test. Si ya existe
    # el guardado por entrenar.py (misma corrida de dataset), lo reutiliza tal cual
    # para que las 3 arquitecturas compitan con exactamente el mismo scaler.
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
    
    env_train = HistoryWrapper(env_train_base, window_size=50)
    env_val = HistoryWrapper(env_val_base, window_size=50)
    env_test = HistoryWrapper(env_test_base, window_size=50)
    
    env_train_vec = DummyVecEnv([lambda: env_train])
    env_train_vec.seed(SEMILLA)
    
    eval_callback = EvalCallback(
        env_val, 
        best_model_save_path=os.path.join(MODELO_DIR, "campeon_cnn_pura_15m_USD"),
        log_path=os.path.join(LOGS_DIR, "eval_cnn_pura"),
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
    
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
        callback=eval_callback,
        progress_bar=True
    )
    end_time = datetime.now()
    
    training_time = (end_time - start_time).total_seconds()
    print(f"\n   ✓ Entrenamiento completado en {training_time:.2f} segundos ({training_time/60:.2f} minutos)")
    
    model_path = os.path.join(MODELO_DIR, f"{NOMBRE_EXPERIMENTO}_final.zip")
    model.save(model_path)
    
    print("\n" + "=" * 60)
    print("📊 EVALUACIÓN FINAL CNN PURA")
    print("=" * 60)
    
    train_metrics = evaluar_modelo(model, env_train, n_episodes=1)
    
    print("\nTEST")
    test_metrics = evaluar_modelo(model, env_test, n_episodes=10)
    
    test_metrics['train_recompensa_experto'] = train_metrics['recompensa_neta_bonus']
    test_metrics['train_recompensa_mercado'] = train_metrics['recompensa_neta_mercado']
    
    obs, info = env_test.reset()
    episode_reward = 0
    step_count = 0
    terminated = False
    truncated = False

    while not terminated and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_test.step(action)
        episode_reward += reward
        step_count += 1
        
        if step_count % 50 == 0:
            env_test.render()
    
    print("-" * 60)
    print(f"\n EPISODIO DETALLADO:")
    print(f"   Balance Inicial:  $10,000.00")
    print(f"   Balance Final:    ${info['net_worth']:,.2f}")
    print(f"   ROI:              {info['profit_pct']:+.2f}%")
    print(f"   Reward Acum.:     {episode_reward:.4f}")
    
    dataset_nombre = "dataset_pca" if "pca" in RUTA_DATOS.lower() else RUTA_DATOS
    ResultadosManager.guardar(
        modelo_nombre=NOMBRE_EXPERIMENTO,
        roi_pct=test_metrics['mean_profit_pct'],
        reward_acumulado=test_metrics['mean_reward'],
        balance_final=test_metrics['mean_net_worth'],
        dataset=f"{dataset_nombre} (TradingEnv_v2)",
        notas=NOTAS_EXPERIMENTO,
        metricas_adicionales=test_metrics
    )

if __name__ == "__main__":
    main()