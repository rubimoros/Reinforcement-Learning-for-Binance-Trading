"""
Wrappers de ventana temporal compartidos entre entrenamiento y análisis.

Antes cada script (entrenarCNN.py, entrenarCNN-LSTM.py) definía su propio
HistoryWrapper de forma local. Eso funcionaba para entrenar, pero en cuanto
se necesita evaluar esos mismos checkpoints fuera del script de entrenamiento
(p. ej. en analizar_checkpoints.py) hay que reconstruir la observación
EXACTAMENTE igual, o el modelo revienta con un ValueError de forma
incompatible. Por eso viven aquí, en un único sitio.
"""

import numpy as np
import gymnasium as gym
from collections import deque


class HistoryWrapperCNN(gym.Wrapper):
    """
    Usado por la arquitectura CNN pura (entrenarCNN.py).
    Devuelve observaciones con forma (window_size, n_features).
    """

    def __init__(self, env, window_size=50):
        super().__init__(env)
        self.window_size = window_size
        self.history = deque(maxlen=window_size)

        base_shape = env.observation_space.shape
        self.n_features = base_shape[0]

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size, self.n_features),
            dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.history.clear()
        for _ in range(self.window_size):
            self.history.append(obs)
        return np.array(self.history, dtype=np.float32), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.history.append(obs)
        return np.array(self.history, dtype=np.float32), reward, terminated, truncated, info


class HistoryWrapperLSTM(gym.ObservationWrapper):
    """
    Usado por la arquitectura CNN-LSTM (entrenarCNN-LSTM.py).
    Devuelve observaciones con forma (n_features, window_size) -- ojo,
    ejes invertidos respecto a HistoryWrapperCNN, porque Conv1d espera
    (canales, longitud_secuencia).
    """

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