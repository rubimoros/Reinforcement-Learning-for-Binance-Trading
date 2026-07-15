import gymnasium as gym
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    def __init__(self, dataset):
        super(TradingEnv, self).__init__()
        self.dataset = dataset.select_dtypes(include=[np.number]).copy()

        self.observation_matriz = self.dataset.values.astype(np.float32)
        self.precios = self.dataset["Close"].values.astype(np.float32)
        self.aperturas = self.dataset["Open"].values.astype(np.float32) 

        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(dataset.shape[1],), dtype=np.float32)
        
        self.current_step = 0
        self.balance_inicial = 10000  # Capital inicial
        self.balance = self.balance_inicial  # Capital actual
        self.crypto_held = 0  # Cantidad de criptomoneda que tenemos
        self.net_worth = self.balance  # Valor total (balance + valor de la criptomoneda)
        self.max_steps = len(self.dataset) - 1  # Máximo número de pasos en el episodio
        self.precio_promedio = 0.0

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.balance = self.balance_inicial
        self.crypto_held = 0
        self.precio_promedio = 0.0
        self.net_worth = self.balance_inicial
        observation = self.next_observation()
        info = {}
        
        return observation, info
    
    def next_observation(self):
        return self.observation_matriz[self.current_step]

    def step(self, action):
        precio_actual = self.precios[self.current_step]
        comision = 0.001 # 0.1% de comisión por operación
        
        if action == 1:  # Comprar
            if self.balance > 0:
                coste_compra = self.balance * (1 - comision)  # Comisión al comprar
                cantidad_a_comprar = coste_compra / precio_actual
                total_usd_value = (self.crypto_held * self.precio_promedio) + coste_compra
                self.crypto_held += cantidad_a_comprar
                self.precio_promedio = total_usd_value / self.crypto_held if self.crypto_held > 0 else 0
                self.balance = 0  # Usamos todo el balance para comprar
                
                print(f"[COMPRA] Paso {self.current_step} | Precio: {self.precio_promedio:.2f} | Invertido: {coste_compra:.2f} $")
        elif action == 2:  # Vender
            if self.crypto_held > 0:
                cantidad_a_vender = self.crypto_held * precio_actual * (1 - comision)
                beneficio_operacion = (precio_actual - self.precio_promedio) * self.crypto_held
                beneficio_operacion -= (cantidad_a_vender * comision)
                self.balance += cantidad_a_vender
                self.crypto_held = 0 # Vendemos toda la criptomoneda que tenemos
                self.precio_promedio = 0.0
                print(f"[VENTA ] Paso {self.current_step} | Precio: {precio_actual:.2f} | Recibido: {cantidad_a_vender:.2f} $ | Beneficio: {beneficio_operacion:.2f} $")
        else:  # Esperar
            pass

        self.current_step += 1
        #TERMINADO
        terminated = self.current_step >= self.max_steps
        truncated = False
        #REWARD
        precio_siguiente = self.dataset.iloc[self.current_step]["Close"]
        nuevo_net_worth = self.balance + (self.crypto_held * precio_siguiente)
        reward = nuevo_net_worth - self.net_worth
        self.net_worth = nuevo_net_worth

        observation = self.next_observation()
        info = {
            'net_worth': self.net_worth, 
            'balance': self.balance, 
            'crypto_held': self.crypto_held,
            'step': self.current_step
        }
        return observation, reward, terminated, truncated, info
    
    def render(self, mode="human"):
        beneficio = self.net_worth - self.balance_inicial
        print(f"Paso: {self.current_step} | Saldo: {self.net_worth:.2f} | Ganancia: {beneficio:.2f}")