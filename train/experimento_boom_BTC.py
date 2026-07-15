import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os
from envs.trading_env import TradingEnv

# Usamos el dataset "fácil" que acabas de generar
RUTA_DATOS = "data/dataset_train_completo.csv"
TIME_STEPS = 50000  # Dale suficientes pasos para que memorice

def main():
    if not os.path.exists(RUTA_DATOS):
        print("❌ Error: Genera primero el dataset con las fechas nuevas.")
        return

    # 1. Cargar TODO el dataset
    dataset = pd.read_csv(RUTA_DATOS, index_col=0, parse_dates=True)

    # 2. Crear entorno con TODOS los datos (Trampa para ver si aprende)
    # Usamos DummyVecEnv para el entrenamiento
    env_train = DummyVecEnv([lambda: TradingEnv(dataset)])

    # 3. Modelo PPO
    # Learning rate un poco más bajo para que afine mejor
    model = PPO("MlpPolicy", env_train, verbose=1, learning_rate=0.0003, ent_coef=0.01)

    print("💪 Entrenando para MEMORIZAR la subida...")
    model.learn(total_timesteps=TIME_STEPS)
    
    print("\n🧐 PROBANDO RESULTADOS (Debería ganar dinero sí o sí)...")
    
    # 4. Prueba manual
    # Usamos el entorno normal (no Dummy) para ver los prints que pusimos
    env_test = TradingEnv(dataset)
    obs, info = env_test.reset()
    terminated = False
    
    while not terminated:
        # deterministic=True OBLIGA a la IA a usar lo mejor que ha aprendido
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_test.step(action)

    beneficio = info['net_worth'] - 10000
    print("-" * 30)
    print(f"💰 RESULTADO FINAL DEL EXPERIMENTO:")
    print(f"   Saldo Inicial: 10,000 $")
    print(f"   Saldo Final:   {info['net_worth']:.2f} $")
    print(f"   Ganancia:      {beneficio:.2f} $")
    print("-" * 30)

if __name__ == "__main__":
    main()