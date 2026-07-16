import sys
import os
from typing import Callable
import pandas as pd
import numpy as np
import gymnasium as gym
import torch as th
import torch.nn as nn
import joblib
from sklearn.preprocessing import StandardScaler
from sb3_contrib import RecurrentPPO as PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from datetime import datetime
from utils.results_manager import ResultadosManager
from envs.trading_env_v2 import TradingEnv
import random

RUTA_DATOS = "data/dataset_puntuaciones_picos_valles_atr_15minutos2021_24.csv"
RUTA_SCALER = "scalers/scaler_15m.pkl"
TIME_STEPS = 50000
NOMBRE_EXPERIMENTO = "RECURRENT_PPO_CNN_LSTM_INDICADORESFINAL(Recurrent_PPO_85_SEMILLA42)"
NOTAS_EXPERIMENTO = "Fase 3 CNN+LSTM: INDICADORES PERSISTENTES PARA EL FUTURO, ENTRENAMIENTO PARA PUESTA EN PRODUCCIÓN SCALER Y 2021-2024. impacto=0.05, vago=-0.05, sell=-0.1."
MODELO_DIR = "models"
LOGS_DIR = "logs"
OBJETIVO = "USD"
SEMILLA = 42

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


def curva_de_aprendizaje(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

class HistoryWrapper(gym.ObservationWrapper):
    def __init__(self, env, window_size=50):
        super().__init__(env)
        self.window_size = window_size
        self.num_features = env.observation_space.shape[0]
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.num_features, window_size),
            dtype=np.float32
        )
        self.obs_buffer = np.zeros((self.num_features, window_size), dtype=np.float32)

    def observation(self, obs):
        self.obs_buffer = np.roll(self.obs_buffer, shift=-1, axis=1)
        self.obs_buffer[:, -1] = obs
        return self.obs_buffer.copy()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.obs_buffer = np.zeros_like(self.obs_buffer)
        return self.observation(obs), info

class TradingCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            nn.Conv1d(n_input_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.Tanh(),
            nn.Flatten(),
        )

        with th.no_grad():
            sample_obs = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_obs).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.Tanh())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

def evaluar_modelo(model, env, n_episodes=10):
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

        lstm_state = None  
        episode_starts = np.ones((1,), dtype=bool)  
        
        while not terminated and not truncated:
            action, lstm_state = model.predict(
                obs, 
                state=lstm_state, 
                episode_start=episode_starts, 
                deterministic=True
            )
            obs, reward, terminated, truncated, info = env.step(action)
            episode_starts = np.zeros((1,), dtype=bool)
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
    
def comparar_con_buy_and_hold(env):
    env_u = env.unwrapped if hasattr(env, 'unwrapped') else env
    precio_inicial = env_u.precios[0]
    precio_final = env_u.precios[env_u.max_steps] 
    balance_inicial = env_u.balance_inicial
    comision = env_u.comision
    spread = env_u.spread
 
    precio_compra = precio_inicial * (1 + spread)
    cripto_comprada = (balance_inicial / (1 + comision)) / precio_compra
    precio_venta = precio_final * (1 - spread)
    balance_final = cripto_comprada * precio_venta * (1 - comision)
    
    return {
        'net_worth': balance_final,
        'profit_pct': ((balance_final - balance_inicial) / balance_inicial) * 100
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

    if os.path.exists(RUTA_SCALER):
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
    
    env_train_hist = HistoryWrapper(env_train_base, window_size=50)
    env_val_hist = HistoryWrapper(env_val_base, window_size=50)
    env_test = HistoryWrapper(env_test_base, window_size=50)
    
    env_train = Monitor(env_train_hist, filename=os.path.join(LOGS_DIR, "train_monitor.csv"))
    env_val = Monitor(env_val_hist, filename=os.path.join(LOGS_DIR, "val_monitor.csv"))
    
    env_train_vec = DummyVecEnv([lambda: env_train])
    env_train_vec.seed(SEMILLA)
    
    eval_callback = EvalCallback(
        env_val,
        best_model_save_path=os.path.join(MODELO_DIR, "best_model_cnn_lstm"),
        log_path=os.path.join(LOGS_DIR, "eval_cnn_lstm"),
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
    
    policy_kwargs = dict(
        features_extractor_class=TradingCNN,
        features_extractor_kwargs=dict(features_dim=256),
        lstm_hidden_size=256,
        n_lstm_layers=1
    )
    
    model = PPO(
        policy="CnnLstmPolicy",
        env=env_train_vec,
        learning_rate=curva_de_aprendizaje(3e-4),
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01, # Lo cambiamos para que el modelo pueda explorar un poco más y no se quede solo holdeando
        vf_coef=0.5,
        policy_kwargs=policy_kwargs, 
        verbose=1,
        tensorboard_log=os.path.join(LOGS_DIR, "tensorboard")
    )
    
    start_time = datetime.now()
    model.learn(total_timesteps=TIME_STEPS, callback=eval_callback, progress_bar=True)
    end_time = datetime.now()
    
    training_time = (end_time - start_time).total_seconds()
    print(f"\n   ✓ Entrenamiento completado en {training_time:.2f} segundos")
    
    model.save(os.path.join(MODELO_DIR, f"{NOMBRE_EXPERIMENTO}_final.zip"))
    
    print("\n" + "=" * 60)
    print("EVALUACIÓN FINAL CNN + LSTM")
    print("=" * 60)
    
    train_metrics = evaluar_modelo(model, env_train_vec.envs[0].env, n_episodes=1)
    
    print("\nTEST")
    test_metrics = evaluar_modelo(model, env_test, n_episodes=10)
    test_metrics['train_recompensa_experto'] = train_metrics['recompensa_neta_bonus']
    test_metrics['train_recompensa_mercado'] = train_metrics['recompensa_neta_mercado']

    print(f"   Influencia en Train -> Mercado: {train_metrics['recompensa_neta_mercado']:.4f} | Experto (Bonus): {train_metrics['recompensa_neta_bonus']:.4f}")
    print(f"   Mean Net Worth:   ${test_metrics['mean_net_worth']:,.2f}")
    print(f"   Mean Profit:      {test_metrics['mean_profit_pct']:.2f}%")
    
    ganancia_promedio = test_metrics['mean_net_worth'] - 10000
    bh_metrics = comparar_con_buy_and_hold(env_test)
    bh_ganancia = bh_metrics['net_worth'] - 10000
    
    print(f"\n   Buy & Hold ROI:   {bh_metrics['profit_pct']:.2f}%")
    print(f"   RL Agent ROI:     {test_metrics['mean_profit_pct']:.2f}%")
    
    obs, info = env_test.reset()
    episode_reward = 0
    step_count = 0
    terminated = False
    truncated = False
    
    lstm_state = None  
    episode_starts = np.ones((1,), dtype=bool)  

    while not terminated and not truncated:
        action, lstm_state = model.predict(
            obs,
            state=lstm_state,
            episode_start=episode_starts,
            deterministic=True
        )
        obs, reward, terminated, truncated, info = env_test.step(action)
        episode_reward += reward
        step_count += 1
        episode_starts = np.zeros((1,), dtype=bool)
        
        if step_count % 500 == 0:
            env_test.render()
    
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
