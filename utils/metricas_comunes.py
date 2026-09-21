"""
Evaluación común para todos los experimentos del TFG (LSTM, CNN, CNN-LSTM).

Objetivo: que los tres scripts de entrenamiento (entrenar.py, entrenarCNN.py,
entrenarCNN-LSTM.py) usen exactamente la misma lógica de evaluación, para que
las comparativas de arquitecturas sean homogéneas (variable controlada =
método de evaluación).

En vez de listar a mano cada clave del diccionario `info` (como se hacía
antes, con el riesgo de que cada script recogiera un subconjunto distinto de
métricas), se agregan automáticamente TODAS las claves numéricas que
TradingEnv_v2._generar_metricas() devuelve. Así, si en el futuro se añade una
métrica nueva al entorno (p. ej. Sortino, Calmar, Alpha...), los tres scripts
la recogen sin tener que tocar este fichero.
"""

import numpy as np


def evaluar_modelo(model, env, n_episodes=10, recurrente=True):
    """
    Ejecuta n_episodes episodios completos en modo determinista y agrega
    las métricas finales de cada episodio (media, desviación, mejor y peor).

    Parameters
    ----------
    model : PPO | RecurrentPPO
    env : gym.Env (o wrapper) que expone step()/reset() y devuelve en el
        último step de cada episodio el diccionario `info` con las métricas
        de TradingEnv_v2.
    n_episodes : int
    recurrente : bool
        True para RecurrentPPO (LSTM / CNN-LSTM), False para PPO estándar (CNN).

    Returns
    -------
    dict con, para cada métrica numérica devuelta por el entorno:
        <metrica>          -> media entre episodios
        <metrica>_std      -> desviación estándar entre episodios
        <metrica>_min      -> peor episodio
        <metrica>_max      -> mejor episodio
    """
    infos_finales = []

    for _ in range(n_episodes):
        obs, info = env.reset()
        terminated, truncated = False, False

        lstm_state = None
        episode_starts = np.ones((1,), dtype=bool)

        while not terminated and not truncated:
            if recurrente:
                action, lstm_state = model.predict(
                    obs,
                    state=lstm_state,
                    episode_start=episode_starts,
                    deterministic=True,
                )
                episode_starts = np.zeros((1,), dtype=bool)
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)

        infos_finales.append(info)

    return _agregar_episodios(infos_finales)


def _agregar_episodios(infos):
    """Agrega automáticamente todas las claves numéricas presentes en `info`."""
    claves_numericas = [
        k for k, v in infos[0].items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]

    resultado = {}
    for k in claves_numericas:
        valores = np.array([info[k] for info in infos if k in info], dtype=float)
        resultado[k] = float(np.mean(valores))
        resultado[f"{k}_std"] = float(np.std(valores))
        resultado[f"{k}_min"] = float(np.min(valores))
        resultado[f"{k}_max"] = float(np.max(valores))

    # Alias legibles para las métricas clave que pide el tutor (facilita
    # generar tablas en la memoria sin recordar el nombre exacto del campo)
    resultado["roi_pct"] = resultado.get("profit_pct", 0.0)
    resultado["buy_and_hold_roi_pct"] = resultado.get("buy_and_hold_roi", 0.0)
    resultado["alpha_pct"] = resultado.get("alpha_vs_bnh", 0.0)

    return resultado


def comparar_con_buy_and_hold(env):
    """Estrategia baseline: All-In matemático, sin restricciones del entorno."""
    env_u = env.unwrapped if hasattr(env, "unwrapped") else env
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

    return {"net_worth": balance_final, "profit_pct": profit_pct}