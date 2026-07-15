import sys
import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO, SAC 
from typing import Callable
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from datetime import datetime
from utils.results_manager import ResultadosManager
from envs.trading_env_v2 import TradingEnv
import torch
import random

RUTA_DATOS = "data/dataset_puntuaciones.csv"
TIME_STEPS = 500000
NOMBRE_EXPERIMENTO = "SAC_2_Mlp_500k_SEMILLA42"
NOTAS_EXPERIMENTO = "Algoritmo SAC implementa una entropia automatica, tiene vision del pasado y ajusta la exploracion de forma automatica. Tiene que batir ese ROI del +54.98%  PPO-MLPLSTM. AHORA CON : 1- mejor cerebro 512, 512, 256, curva de aprendizaje, estudia y opera con mas velas mas a largo plazo (gamma 0.995)."
MODELO_DIR = "models"
LOGS_DIR = "logs"

SEMILLA = 42
random.seed(SEMILLA)
np.random.seed(SEMILLA)
torch.manual_seed(SEMILLA)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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
    all_bonus = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        terminated = False
        truncated = False
        
        while not terminated and not truncated:
            action, _ = model.predict(
                obs, 
                deterministic=True
            )
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            all_bonus.append(info['bonus'])
            # Si el bonus representa el 90% de la recompensa: La IA es adicta.
            # Si el bonus representa el 5%: La IA te ignora.
            # Si el bonus representa entre un 20% y un 40%: La IA te hace caso pero también aprende de la experiencia.
        
        final_net_worth = info['net_worth']
        profit_pct = info['profit_pct']
        
        all_rewards.append(episode_reward)
        all_net_worths.append(final_net_worth)
        all_profits.append(profit_pct)
        
    total_recompensa = np.sum(all_rewards)
    total_bonus = np.sum(all_bonus)
    total_recompensa_mercado = total_recompensa - total_bonus
    
    if total_recompensa != 0:
        porcentaje_bonus = (abs(total_bonus) / abs(total_recompensa)) * 100
    else:
        porcentaje_bonus = 0.0
    
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
        'min_net_worth': np.min(all_net_worths),
        'total_buys': info['total_buys'],
        'total_sells': info['total_sells'],
        'total_holds': info['total_holds'],
        'win_rate': info['ratio_ganadoras'],
        'max_drawdown': info['max_drawdown'],
        'mejor_trade_pct': info['mejor_trade_pct'] * 100,
        'peor_trade_pct': info['peor_trade_pct'] * 100,
        'fecha_mejor_trade': info['fecha_mejor_trade'],
        'fecha_peor_trade': info['fecha_peor_trade'],
        'recompensa_neta_mercado': float(total_recompensa_mercado),
        'recompensa_neta_bonus': float(total_bonus)
    }

def comparar_con_buy_and_hold(env):
    """Estrategia baseline: matemática pura (All-In real sin restricciones del entorno)"""

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
    profit_pct = ((balance_final - balance_inicial) / balance_inicial) * 100
    
    return {
        'net_worth': balance_final,
        'profit_pct': profit_pct
    }

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
    
    df_train = dataset.iloc[:train_end].copy()
    df_val = dataset.iloc[train_end:val_end].copy()
    df_test = dataset.iloc[val_end:].copy()

    df_val['Puntuacion'] = 5
    df_test['Puntuacion'] = 5
    
    # Crear entornos
    env_train = Monitor(TradingEnv(df_train), filename=os.path.join(LOGS_DIR, "train_monitor.csv"))
    env_val = Monitor(TradingEnv(df_val), filename=os.path.join(LOGS_DIR, "val_monitor.csv"))
    env_test = TradingEnv(df_test)
    
    env_train_vec = DummyVecEnv([lambda: env_train])
    env_train_vec.seed(SEMILLA)
    
    eval_callback = EvalCallback(
        env_val,
        best_model_save_path=os.path.join(MODELO_DIR, "best_model"),
        log_path=os.path.join(LOGS_DIR, "eval"),
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
    
    policy_kwargs = dict(net_arch=[512, 512, 256])
    
    # Instanciar modelo SAC
    model = SAC(
        policy="MlpPolicy",
        env=env_train_vec,
        learning_rate=curva_de_aprendizaje(3e-4), # LR Dinámico
        buffer_size=100000,
        batch_size=256,
        ent_coef='auto',
        gamma=0.995,              # Visión más a largo plazo
        tau=0.005,
        train_freq=(100, "step"), # Opera 100 pasos seguidos...
        gradient_steps=50,        # ...y estudia 50 (menos pánico)
        policy_kwargs=policy_kwargs, # Red neuronal masiva
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
    print(f"   ✓ Modelo guardado en: {model_path}")
    
    # ========================================================================
    # EVALUACIÓN FINAL
    # ========================================================================
    print("\n" + "=" * 60)
    print("📊 EVALUACIÓN FINAL")
    print("=" * 60)
    
    # Evaluar en conjunto de test
    train_metrics = evaluar_modelo(model, env_train_vec.envs[0].env, n_episodes=1)
    
    print("\nTEST")
    test_metrics = evaluar_modelo(model, env_test, n_episodes=10)
    
    test_metrics['train_recompensa_experto'] = train_metrics['recompensa_neta_bonus']
    test_metrics['train_recompensa_mercado'] = train_metrics['recompensa_neta_mercado']
    print(f"   Influencia en Train -> Mercado: {train_metrics['recompensa_neta_mercado']:.4f} | Experto (Bonus): {train_metrics['recompensa_neta_bonus']:.4f}")

    print(f"   Mean Reward:      {test_metrics['mean_reward']:.4f} ± {test_metrics['std_reward']:.4f}")
    print(f"   Mean Net Worth:   ${test_metrics['mean_net_worth']:,.2f} ± ${test_metrics['std_net_worth']:,.2f}")
    print(f"   Mean Profit:      {test_metrics['mean_profit_pct']:.2f}% ± {test_metrics['std_profit_pct']:.2f}%")
    print(f"   Max Profit:       {test_metrics['max_profit_pct']:.2f}% (${test_metrics['max_net_worth']:,.2f})")
    print(f"   Min Profit:       {test_metrics['min_profit_pct']:.2f}% (${test_metrics['min_net_worth']:,.2f})")
    
    # Calcular ganancia promedio
    ganancia_promedio = test_metrics['mean_net_worth'] - 10000
    print(f"\nGanancia Promedio: ${ganancia_promedio:+,.2f}")
    
    # Comparar con Buy & Hold
    print("\nBuy & Hold...")
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
    obs, info = env_test.reset()
    episode_reward = 0
    step_count = 0
    terminated = False
    truncated = False

    while not terminated and not truncated:
        action, _ = model.predict(
            obs,
            deterministic=True
        )
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
        balance_final=test_metrics['mean_net_worth'],
        dataset=f"{dataset_nombre} (TradingEnv_v2)",
        notas=NOTAS_EXPERIMENTO,
        metricas_adicionales=test_metrics
    )

if __name__ == "__main__":
    main()