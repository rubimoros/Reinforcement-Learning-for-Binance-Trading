import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os
from envs.trading_env import TradingEnv

RUTA_DATOS = "data/dataset_train_completo.csv"
TIME_STEPS = 50000

def main():
    if not os.path.exists(RUTA_DATOS):
        print("Error generando el dataset: No existe el archivo de datos")
        return

    dataset = pd.read_csv(RUTA_DATOS, index_col=0, parse_dates=True)
    env_train = DummyVecEnv([lambda: TradingEnv(dataset)])
    model = PPO("MlpPolicy", env_train, verbose=1, learning_rate=0.0003, ent_coef=0.01)

    model.learn(total_timesteps=TIME_STEPS)
    env_test = TradingEnv(dataset)
    obs, info = env_test.reset()
    terminated = False
    
    while not terminated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_test.step(action)

    beneficio = info['net_worth'] - 10000
    print("-" * 30)
    print(f"    RESULTADO:")
    print(f"   Saldo Inicial: 10,000 $")
    print(f"   Saldo Final:   {info['net_worth']:.2f} $")
    print(f"   Ganancia:      {beneficio:.2f} $")
    print("-" * 30)

if __name__ == "__main__":
    main()
