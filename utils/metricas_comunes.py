import numpy as np


def evaluar_modelo(model, env, n_episodes=10, recurrente=True):
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
    resultado["roi_pct"] = resultado.get("profit_pct", 0.0)
    resultado["buy_and_hold_roi_pct"] = resultado.get("buy_and_hold_roi", 0.0)
    resultado["alpha_pct"] = resultado.get("alpha_vs_bnh", 0.0)

    return resultado


def comparar_con_buy_and_hold(env):
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
