import sys
import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from typing import Callable
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from datetime import datetime
from utils.results_manager import ResultadosManager
from envs.trading_env_v2 import TradingEnv

RUTA_DATOS = "data/dataset_train_completo.csv"
TIME_STEPS = 500000
MODELO_DIR = "models"
LOGS_DIR = "logs"

NOMBRE_EXPERIMENTO = "PPO_29_500k_curva_aprendizaje"
NOTAS_EXPERIMENTO = "Entrenamiento con dataset completo, TradingEnv_v2, 500k steps"

def curva_de_aprendizaje(initial_value: float) -> Callable[[float], float]:
    """
    Desciende el learning rate linealmente desde initial_value hasta 0.
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def evaluar_modelo(model, env, n_episodes=10):
    """Evalúa el modelo durante n episodios y retorna métricas COMPLETAS"""
    all_rewards = []
    all_net_worths = []
    all_profits = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        terminated = False
        truncated = False
        
        while not terminated and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
        
        final_net_worth = info['net_worth']
        profit_pct = info['profit_pct']
        
        all_rewards.append(episode_reward)
        all_net_worths.append(final_net_worth)
        all_profits.append(profit_pct)
    
    return {
        'mean_reward': np.mean(all_rewards),
        'std_reward': np.std(all_rewards),
        'mean_net_worth': np.mean(all_net_worths),
        'std_net_worth': np.std(all_net_worths),
        'mean_profit_pct': np.mean(all_profits),
        'std_profit_pct': np.std(all_profits),
        'max_profit_pct': np.max(all_profits),
        'min_profit_pct': np.min(all_profits),
        'max_net_worth': np.max(all_net_worths),
        'min_net_worth': np.min(all_net_worths)
    }

def comparar_con_buy_and_hold(env):
    """Estrategia baseline: comprar al inicio y vender al final"""
    obs, info = env.reset()

    action_buy = np.array([1.0, 1.0], dtype=np.float32)
    action_sell = np.array([-1.0, 1.0], dtype=np.float32)
    action_hold = np.array([0.0, 0.0], dtype=np.float32)
    
    obs, reward, terminated, truncated, info = env.step(action_buy)
    
    while not terminated and not truncated:
        obs, reward, terminated, truncated, info = env.step(action_hold)
    
    if info['crypto_held'] > 0:
        obs, _ = env.reset()
        env.step(action_buy)
        for _ in range(env.max_steps - 1):
            env.step(action_hold)
        obs, reward, terminated, truncated, info = env.step(action_sell)
    
    return {
        'net_worth': info['net_worth'],
        'profit_pct': info['profit_pct']
    }

# ============================================================================
# MAIN
# ============================================================================

def main():
    # Crear directorios
    os.makedirs(MODELO_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # Cargar dataset
    dataset = pd.read_csv(RUTA_DATOS, index_col=0, parse_dates=True)
    
    # Split temporal
    n = len(dataset)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    df_train = dataset.iloc[:train_end]
    df_val = dataset.iloc[train_end:val_end]
    df_test = dataset.iloc[val_end:]
    
    # Crear entornos
    env_train = Monitor(TradingEnv(df_train), filename=os.path.join(LOGS_DIR, "train_monitor.csv"))
    env_val = Monitor(TradingEnv(df_val), filename=os.path.join(LOGS_DIR, "val_monitor.csv"))
    env_test = TradingEnv(df_test)
    
    env_train_vec = DummyVecEnv([lambda: env_train])
    
    eval_callback = EvalCallback(
        env_val,
        best_model_save_path=os.path.join(MODELO_DIR, "best_model"),
        log_path=os.path.join(LOGS_DIR, "eval"),
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
    
    # Instanciar modelo PPO
    print("\n🤖 Creando modelo PPO...")
    model = PPO(
        policy="MlpPolicy",
        env=env_train_vec,
        learning_rate= curva_de_aprendizaje(3e-4),
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
    
    # Entrenar modelo
    start_time = datetime.now()
    model.learn(
        total_timesteps=TIME_STEPS,
        callback=eval_callback,
        progress_bar=True
    )
    end_time = datetime.now()
    
    training_time = (end_time - start_time).total_seconds()
    print(f"\n   ✓ Entrenamiento completado en {training_time:.2f} segundos ({training_time/60:.2f} minutos)")
    
    # Guardar modelo final
    model_path = os.path.join(MODELO_DIR, f"{NOMBRE_EXPERIMENTO}_final.zip")
    model.save(model_path)
    
    # Evaluar en conjunto de test
    print("\nTEST")
    test_metrics = evaluar_modelo(model, env_test, n_episodes=10)
    
    print(f"   Mean Reward:      {test_metrics['mean_reward']:.4f} ± {test_metrics['std_reward']:.4f}")
    print(f"   Mean Net Worth:   ${test_metrics['mean_net_worth']:,.2f} ± ${test_metrics['std_net_worth']:,.2f}")
    print(f"   Mean Profit:      {test_metrics['mean_profit_pct']:.2f}% ± {test_metrics['std_profit_pct']:.2f}%")
    print(f"   Max Profit:       {test_metrics['max_profit_pct']:.2f}% (${test_metrics['max_net_worth']:,.2f})")
    print(f"   Min Profit:       {test_metrics['min_profit_pct']:.2f}% (${test_metrics['min_net_worth']:,.2f})")
    
    # Calcular ganancia promedio
    ganancia_promedio = test_metrics['mean_net_worth'] - 10000
    print(f"\n Ganancia Promedio: ${ganancia_promedio:+,.2f}")
    
    # Comparar con Buy & Hold
    print("\n  Comparación con Buy & Hold...")
    bh_metrics = comparar_con_buy_and_hold(env_test)
    bh_ganancia = bh_metrics['net_worth'] - 10000
    
    print(f"   Buy & Hold:")
    print(f"      ROI:      {bh_metrics['profit_pct']:.2f}%")
    print(f"      Balance:  ${bh_metrics['net_worth']:,.2f}")
    print(f"      Ganancia: ${bh_ganancia:+,.2f}")
    
    print(f"\n   RL Agent:")
    print(f"      ROI:      {test_metrics['mean_profit_pct']:.2f}%")
    print(f"      Balance:  ${test_metrics['mean_net_worth']:,.2f}")
    print(f"      Ganancia: ${ganancia_promedio:+,.2f}")
    
    if test_metrics['mean_profit_pct'] > bh_metrics['profit_pct']:
        diferencia_roi = test_metrics['mean_profit_pct'] - bh_metrics['profit_pct']
        diferencia_usd = ganancia_promedio - bh_ganancia
        print(f"\n    El agente RL supera a Buy & Hold:")
        print(f"      Diferencia ROI: +{diferencia_roi:.2f}%")
        print(f"      Diferencia USD: ${diferencia_usd:+,.2f}")
    else:
        diferencia_roi = bh_metrics['profit_pct'] - test_metrics['mean_profit_pct']
        diferencia_usd = bh_ganancia - ganancia_promedio
        print(f"\n    Buy & Hold supera al agente RL:")
        print(f"      Diferencia ROI: +{diferencia_roi:.2f}%")
        print(f"      Diferencia USD: ${diferencia_usd:+,.2f}")
    
    # Ejecutar 1 episodio detallado
    print("\n🎬 Ejecutando episodio detallado en TEST...")
    print("-" * 60)
    
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
        
        if step_count % 500 == 0:
            env_test.render()
    
    print("-" * 60)
    print(f"\n EPISODIO DETALLADO:")
    print(f"   Balance Inicial:  $10,000.00")
    print(f"   Balance Final:    ${info['net_worth']:,.2f}")
    print(f"   Ganancia:         ${info['net_worth'] - 10000:+,.2f}")
    print(f"   ROI:              {info['profit_pct']:+.2f}%")
    print(f"   Total Steps:      {step_count}")
    print(f"   Reward Acum.:     {episode_reward:.4f}")
    
    dataset_nombre = "dataset_pca" if "pca" in RUTA_DATOS.lower() else "dataset_train_completo"
    ResultadosManager.guardar(
        modelo_nombre=NOMBRE_EXPERIMENTO,
        roi_pct=test_metrics['mean_profit_pct'],
        reward_acumulado=test_metrics['mean_reward'],
        balance_final=test_metrics['mean_net_worth'],  # ← NUEVO
        dataset=f"{dataset_nombre} (TradingEnv_v2)",
        notas=NOTAS_EXPERIMENTO
    )

if __name__ == "__main__":
    main()