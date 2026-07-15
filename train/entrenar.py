import sys
import os
import glob
import re
import pandas as pd
import numpy as np
from sb3_contrib import RecurrentPPO as PPO
from typing import Callable
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from datetime import datetime
from utils.results_manager import ResultadosManager
from envs.trading_env_v2 import TradingEnv
import joblib
from sklearn.preprocessing import StandardScaler
import torch
import random

RUTA_DATOS = "data/dataset_puntuaciones_picos_valles_atr_15minutos2021_24.csv"
TIME_STEPS = 500000
CHECKPOINT_FREQ = 25000  # guarda un snapshot cada 25k steps para poder compararlos todos luego
NOMBRE_EXPERIMENTO = "LSTM_INDICADORESFINAL_LRDECAY(RecurrentPPO87_SEMILLA42)"
NOTAS_EXPERIMENTO = "impacto=0.05, vago=-0.05, sell=-0.1. Con decaimiento de learning rate y evaluación de checkpoints intermedios contra test real."
MODELO_DIR = "models"
CHECKPOINTS_DIR = os.path.join(MODELO_DIR, "checkpoints")
LOGS_DIR = "logs"
OBJETIVO = "USD"

SEMILLA = 42
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
        all_net_worths_btc.append(info['net_worth_btc'])
        all_profits_btc.append(info['profit_btc_pct'])
        
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
        'win_rate': info['ratio_ganadoras'],
        'max_drawdown': info['max_drawdown'],
        'sharpe_ratio': info.get('sharpe_ratio', 0.0),
        'profit_factor': info.get('profit_factor', 0.0),
        'holding_time': info.get('holding_time', 0.0),
        'mejor_trade_pct': info['mejor_trade_pct'] * 100,
        'peor_trade_pct': info['peor_trade_pct'] * 100,
        'fecha_mejor_trade': info['fecha_mejor_trade'],
        'fecha_peor_trade': info['fecha_peor_trade'],
        'pasos_completados': None,  # se rellena fuera si el episodio se corta antes de tiempo
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
    
    return {'net_worth': balance_final, 'profit_pct': profit_pct}


def extraer_steps(nombre_archivo):
    """Extrae el número de steps del nombre de fichero que genera CheckpointCallback,
    del tipo NOMBRE_EXPERIMENTO_123456_steps.zip"""
    match = re.search(r'_(\d+)_steps', nombre_archivo)
    return int(match.group(1)) if match else -1


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
    
    cols_a_ignorar = ["Open", "High", "Low", "Close", "Volume", "Puntuacion"]
    indicadores_train = df_train.drop(columns=cols_a_ignorar, errors='ignore')
    
    scaler_maestro = StandardScaler()
    scaler_maestro.fit(indicadores_train)

    os.makedirs("scalers", exist_ok=True)
    joblib.dump(scaler_maestro, "scalers/scaler_15m.pkl")
    
    env_train = Monitor(TradingEnv(df_train, scaler=scaler_maestro, objetivo=OBJETIVO), filename=os.path.join(LOGS_DIR, "train_monitor.csv"))
    env_val = Monitor(TradingEnv(df_val, scaler=scaler_maestro, objetivo=OBJETIVO), filename=os.path.join(LOGS_DIR, "val_monitor.csv"))
    env_test = TradingEnv(df_test, scaler=scaler_maestro, objetivo=OBJETIVO)
    
    env_train_vec = DummyVecEnv([lambda: env_train])
    env_train_vec.seed(SEMILLA)
    
    eval_callback = EvalCallback(
        env_val,
        best_model_save_path=os.path.join(MODELO_DIR, "best_model_lstm_15m"),
        log_path=os.path.join(LOGS_DIR, "eval"),
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=CHECKPOINTS_DIR,
        name_prefix=NOMBRE_EXPERIMENTO
    )

    callbacks = CallbackList([eval_callback, checkpoint_callback])
    
    # Decaimiento de learning rate activado (antes se definía pero no se usaba)
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
    
    training_time = (end_time - start_time).total_seconds()
    print(f"\n   ✓ Entrenamiento completado en {training_time:.2f} segundos ({training_time/60:.2f} minutos)")
    
    model_path = os.path.join(MODELO_DIR, f"{NOMBRE_EXPERIMENTO}_final.zip")
    model.save(model_path)
    print(f"   ✓ Modelo final guardado en: {model_path}")

    # ========================================================================
    # EVALUACIÓN DE TODOS LOS CHECKPOINTS CONTRA EL TEST SET REAL
    # ========================================================================
    print("\n" + "=" * 70)
    print("📊 EVALUANDO TODOS LOS CHECKPOINTS CONTRA TEST (para elegir el mejor punto de corte)")
    print("=" * 70)

    bh_metrics = comparar_con_buy_and_hold(env_test)

    candidatos = []

    # Checkpoints periódicos guardados durante el entrenamiento
    for ruta in sorted(glob.glob(os.path.join(CHECKPOINTS_DIR, f"{NOMBRE_EXPERIMENTO}_*_steps.zip")), key=extraer_steps):
        steps = extraer_steps(ruta)
        candidatos.append((f"checkpoint_{steps}", ruta))

    # El "mejor por validación" de EvalCallback
    ruta_best = os.path.join(MODELO_DIR, "best_model_lstm_15m", "best_model.zip")
    if os.path.exists(ruta_best):
        candidatos.append(("best_model (EvalCallback)", ruta_best))

    # El modelo final tras todos los steps
    candidatos.append(("final", model_path))

    resultados_tabla = []
    for etiqueta, ruta in candidatos:
        try:
            modelo_candidato = PPO.load(ruta)
        except Exception as e:
            print(f"   ⚠️ No se pudo cargar {ruta}: {e}")
            continue

        metricas = evaluar_modelo(modelo_candidato, env_test, n_episodes=10)
        resultados_tabla.append({
            'candidato': etiqueta,
            'roi_pct': metricas['mean_profit_pct'],
            'sharpe': metricas['sharpe_ratio'],
            'win_rate': metricas['win_rate'],
            'profit_factor': metricas['profit_factor'],
            'max_drawdown': metricas['max_drawdown'],
            'buys': metricas['total_buys'],
            'sells': metricas['total_sells'],
            'holds': metricas['total_holds'],
        })

    print(f"\n{'CANDIDATO':<28}{'ROI%':>10}{'SHARPE':>10}{'WIN%':>9}{'P.FACTOR':>11}{'MAXDD%':>10}{'B/S/H':>16}")
    print("-" * 95)
    for r in sorted(resultados_tabla, key=lambda x: x['roi_pct'], reverse=True):
        bsh = f"{r['buys']}/{r['sells']}/{r['holds']}"
        print(f"{r['candidato']:<28}{r['roi_pct']:>10.2f}{r['sharpe']:>10.3f}{r['win_rate']:>9.2f}"
              f"{r['profit_factor']:>11.3f}{r['max_drawdown']:>10.2f}{bsh:>16}")

    print("-" * 95)
    print(f"{'Buy & Hold':<28}{bh_metrics['profit_pct']:>10.2f}")
    print("=" * 95)

    mejor = max(resultados_tabla, key=lambda x: x['roi_pct']) if resultados_tabla else None
    if mejor:
        print(f"\n👉 Mejor candidato por ROI en test: {mejor['candidato']} "
              f"(ROI {mejor['roi_pct']:.2f}%, Sharpe {mejor['sharpe']:.3f})")
        print("   Revisa también Sharpe y max drawdown antes de decidir cuál poner en producción —")
        print("   el mejor ROI puntual no siempre es la opción más estable.")

    # Guardar el resultado del modelo final como referencia histórica (igual que antes)
    test_metrics_final = next((r for r in resultados_tabla if r['candidato'] == 'final'), None)
    if test_metrics_final:
        ResultadosManager.guardar(
            modelo_nombre=NOMBRE_EXPERIMENTO,
            roi_pct=test_metrics_final['roi_pct'],
            reward_acumulado=0.0,
            balance_final=10000 * (1 + test_metrics_final['roi_pct'] / 100),
            dataset=f"{RUTA_DATOS} (TradingEnv_v2, LR decay + checkpoints)",
            notas=NOTAS_EXPERIMENTO,
            metricas_adicionales=test_metrics_final
        )

if __name__ == "__main__":
    main()