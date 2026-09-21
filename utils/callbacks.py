import os
from stable_baselines3.common.callbacks import BaseCallback
from utils.metricas_comunes import evaluar_modelo


class SeleccionPorMetricaCallback(BaseCallback):

    def __init__(self, env_val, ruta_guardado, eval_freq=25000, n_episodes=5,
                 metrica="profit_pct", recurrente=True, verbose=1):
        super().__init__(verbose)
        self.env_val = env_val
        self.ruta_guardado = ruta_guardado
        self.eval_freq = eval_freq
        self.n_episodes = n_episodes
        self.metrica = metrica
        self.recurrente = recurrente
        self.mejor_valor = -float("inf")
        self.mejor_step = None

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            metricas = evaluar_modelo(
                self.model, self.env_val,
                n_episodes=self.n_episodes,
                recurrente=self.recurrente
            )
            valor_actual = metricas[self.metrica]

            if self.verbose:
                print(f"[val step {self.num_timesteps}] {self.metrica} = "
                      f"{valor_actual:.3f} (mejor hasta ahora: {self.mejor_valor:.3f})")

            if valor_actual > self.mejor_valor:
                self.mejor_valor = valor_actual
                self.mejor_step = self.num_timesteps
                os.makedirs(self.ruta_guardado, exist_ok=True)
                self.model.save(os.path.join(self.ruta_guardado, "best_model.zip"))
                if self.verbose:
                    print(f" Nuevo mejor modelo guardado (step {self.num_timesteps})")

        return True
