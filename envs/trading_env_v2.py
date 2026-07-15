import gymnasium as gym
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class TradingEnv_v2(gym.Env):

    def __init__(self, dataset,scaler, verbose=False, objetivo= "USD"):
        super(TradingEnv_v2, self).__init__()
        self.dataset = dataset.select_dtypes(include=[np.number]).copy()
        self.scaler = scaler
        self.verbose = verbose
        self.objetivo = objetivo.upper()

        self.precios = self.dataset["Close"].values.astype(np.float32)
        self.aperturas = self.dataset["Open"].values.astype(np.float32)
        self.puntuaciones = self.dataset["Puntuacion"].values.astype(np.float32)

        indicadores = dataset.drop(
            columns=["Open", "High", "Low", "Close", "Volume", "Puntuacion"], 
            errors='ignore'
        ).copy()

        indicadores_scaled = self.scaler.transform(indicadores)
        self.observation_matriz = indicadores_scaled.astype(np.float32)

        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0]), 
            high=np.array([1.0, 1.0]), 
            dtype=np.float32
        )

        obs_dim = indicadores_scaled.shape[1] + 4 
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        self.balance_inicial = 10000.0
        self.comision = 0.001   
        self.spread = 0.0005    
        
        self.reset_variables()

    def reset_variables(self):
        self.current_step = 0
        self.balance = self.balance_inicial
        self.btc_inicial = self.balance_inicial / self.precios[0]
        self.crypto_held = 0.0
        self.precio_promedio = 0.0
        self.net_worth = self.balance_inicial
        self.max_steps = len(self.dataset) - 1
        
        # Tracking básico
        self.total_buys = 0
        self.total_sells = 0
        self.total_holds = 0
        self.ventas_ganadoras = 0
        self.ventas_perdedoras = 0
        self.gross_profit = 0.0
        self.gross_loss = 0.0
        self.mejor_trade_pct = 0.0
        self.peor_trade_pct = 0.0
        self.fecha_mejor_trade = None
        self.fecha_peor_trade = None
        self.peak_net_worth = self.balance_inicial
        self.max_drawdown = 0.0
        self.historial_net_worth = [self.net_worth]
        self.historial_trades_pct = [] 
        self.current_streak = 0
        self.max_win_streak = 0
        self.max_lose_streak = 0 
        self.step_ultima_compra = 0
        self.tiempos_operacion = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_variables()
        return self._get_observation(), {}
    
    def _get_observation(self):
        market_obs = self.observation_matriz[self.current_step]
        precio_actual = self.precios[self.current_step]
        
        balance_change = (self.balance - self.balance_inicial) / (self.balance_inicial + 1e-8)
        crypto_value = (self.crypto_held * precio_actual)
        crypto_change = (crypto_value - 0) / (self.balance_inicial + 1e-8)
        net_worth_change = (self.net_worth - self.balance_inicial) / (self.balance_inicial + 1e-8)
        
        if self.precio_promedio > 0 and self.crypto_held > 0:
            pnl = (precio_actual - self.precio_promedio) / self.precio_promedio
        else:
            pnl = 0.0
        
        portfolio_obs = np.array([
            np.clip(balance_change, -10, 10),     
            np.clip(crypto_change, -10, 10),      
            np.clip(net_worth_change, -10, 10),   
            np.clip(pnl, -1, 1)                   
        ], dtype=np.float32)
        
        return np.concatenate([market_obs, portfolio_obs])

    def step(self, action):
        tipo_operacion = action[0]
        porcentaje_operacion = np.clip(action[1], 0.01, 1.0)
        
        precio_actual = self.precios[self.current_step]
        precio_compra = precio_actual * (1 + self.spread)
        precio_venta = precio_actual * (1 - self.spread)
        
        nw_before = self.balance + (self.crypto_held * precio_venta)
        self.balance_anterior = self.balance
        self.crypto_held_anterior = self.crypto_held
        
        profit_realizado = 0.0
        operacion = "HOLD"
        fecha_actual = self.dataset.index[self.current_step] if True else self.current_step
        
        # ---COMPRA ---
        if tipo_operacion > 0.33 and self.balance > 10:  
            NUM_SACOS = 5  
            tamano_saco_maximo = self.balance_inicial / NUM_SACOS
            gasto_deseado = tamano_saco_maximo * porcentaje_operacion
            
            cantidad_gastar = min(gasto_deseado, self.balance)
            cantidad_comprada = (cantidad_gastar / (1 + self.comision)) / precio_compra
            
            if self.crypto_held > 0:
                total_usd = (self.crypto_held * self.precio_promedio) + cantidad_gastar * (1 - self.comision)
                self.crypto_held += cantidad_comprada
                self.precio_promedio = total_usd / self.crypto_held
            else:
                self.crypto_held = cantidad_comprada
                self.precio_promedio = precio_compra
            
            self.balance -= cantidad_gastar
            operacion = "BUY"
            self.total_buys += 1

        # ---VENTA ---
        elif tipo_operacion < -0.33 and self.crypto_held > 1e-5:  
            cantidad_vender = self.crypto_held * porcentaje_operacion
            ingresos_brutos = cantidad_vender * precio_venta
            ingresos_netos = ingresos_brutos * (1 - self.comision)
            
            if self.precio_promedio > 0:
                profit_realizado = (precio_venta - self.precio_promedio) / self.precio_promedio
            
            self.balance += ingresos_netos
            self.crypto_held -= cantidad_vender
            self.total_sells += 1
            
            if self.crypto_held < 1e-5:
                self.crypto_held = 0.0
                self.precio_promedio = 0.0
            self.historial_trades_pct.append(profit_realizado)
            
            if profit_realizado > 0:
                self.ventas_ganadoras += 1
                self.current_streak = self.current_streak + 1 if self.current_streak > 0 else 1
                self.max_win_streak = max(self.max_win_streak, self.current_streak)
            else:
                self.ventas_perdedoras += 1
                self.current_streak = self.current_streak - 1 if self.current_streak < 0 else -1
                self.max_lose_streak = max(self.max_lose_streak, abs(self.current_streak))

            if profit_realizado > self.mejor_trade_pct:
                self.mejor_trade_pct = profit_realizado
                self.fecha_mejor_trade = fecha_actual
            
            if profit_realizado < self.peor_trade_pct:
                self.peor_trade_pct = profit_realizado
                self.fecha_peor_trade = fecha_actual

            operacion = "SELL"
        else:
            operacion = "HOLD"
            self.total_holds += 1
        
        # ---RECOMPENSAS ---
        nw_after = self.balance + (self.crypto_held * precio_venta)
        
        if self.objetivo == "USD":
            objetivo_ganancia_antes = self.balance_anterior + (self.crypto_held_anterior * precio_actual)
            objetivo_ganancia_despues = self.balance + (self.crypto_held * precio_actual)
            reward_base = (objetivo_ganancia_despues - objetivo_ganancia_antes) / 100.0  
        elif self.objetivo == "BTC":
            objetivo_ganancia_antes = self.crypto_held_anterior + (self.balance_anterior / precio_actual)
            objetivo_ganancia_despues = self.crypto_held + (self.balance / precio_actual)
            reward_base = (objetivo_ganancia_despues - objetivo_ganancia_antes) * 1000.0  

        if nw_after > self.peak_net_worth:
            self.peak_net_worth = nw_after
        caida_actual = (self.peak_net_worth - nw_after) / self.peak_net_worth
        if caida_actual > self.max_drawdown:
            self.max_drawdown = caida_actual
            
        self.historial_net_worth.append(nw_after)
        
        bonus_sell = 0.0
        penalty_loss = 0.0
        if operacion == "SELL":
	    # Bonificación por venta rentable
            if profit_realizado > 0:
                if profit_realizado < 0.005:  # Si la ganancia es menor al 0.5%
                	bonus_sell = -0.1       # Castigo bestial por cobarde se va bajando a medida que hacemos que sea menos adicto
            	else:
                	# bonus_sell = min(profit_realizado * 2, 0.5)  # vamos a multiplicar por 2 el bonus previo por compras con buen beneficio
                	bonus_sell = min(profit_realizado / 10, 0.05)  # Max 0.05
            else:
		# Penalización por ventas con pérdida
                penalty_loss = max(profit_realizado / 10, -0.05)

	# Recompensa con puntuación
        puntuacion_actual = self.puntuaciones[self.current_step]
        bonus_puntuacion = 0.0
        impacto_puntuacion = 0.05 #Cambiar a 0,02 0,01 si solo hace caso a mis indicaciones, si pasa de mi ponerlo en 0,1
        # ep_mean_reard sube mucho pero si salta el EvalCallback y miramos el mean_reward es negativo o cercano a cera cambia el factor 0.02 o 0.01
        # Si el WinRate no ha mejorado con respecto a modelos anteriores subir el impacto a 0.01 o 0.015

        if operacion == "BUY":
            bonus_puntuacion = puntuacion_actual * impacto_puntuacion
            if self.step_ultima_compra == 0: 
                self.step_ultima_compra = self.current_step
        elif operacion == "SELL":
            bonus_puntuacion = -puntuacion_actual * impacto_puntuacion
            net_worth_antes_vender = self.net_worth
            self.net_worth = self.balance + (self.crypto_held * precio_venta)
            beneficio_monetario = self.net_worth - net_worth_antes_vender
            if beneficio_monetario > 0:
                self.gross_profit += beneficio_monetario
            else:
                self.gross_loss += abs(beneficio_monetario)

            if self.step_ultima_compra > 0:
                duracion_velas = self.current_step - self.step_ultima_compra
                self.tiempos_operacion.append(duracion_velas)
                self.step_ultima_compra = 0

        # Penalizacion por vago        
        penalizacion_vago = -0.05 if operacion == "HOLD" and self.crypto_held == 0.0 else 0.0
	# Reward final
        reward = reward_base + bonus_sell + penalty_loss + bonus_puntuacion + penalizacion_vago
        
        self.net_worth = nw_after
	# Avanzar
        self.current_step += 1
        # Terminación
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        info = self._generar_metricas(precio_actual, operacion, profit_realizado, fecha_actual, bonus_puntuacion)

        if not terminated:
            observation = self._get_observation()
        else:
            self.current_step -= 1
            observation = self._get_observation()
            self.current_step += 1
        
        # Bancarrota
        CIRCUIT_BREAKER_THRESHOLD = -25.0
        if info['profit_pct'] <= CIRCUIT_BREAKER_THRESHOLD:
            terminated = True
            reward -= 10.0  
            info['operacion'] = "BANKRUPT"
        
        return observation, reward, terminated, truncated, info

    def _generar_metricas(self, precio_actual, operacion, profit_realizado, fecha_actual, bonus_puntuacion):
        nw_btc = self.crypto_held + (self.balance / precio_actual)
        profit_btc_pct = ((nw_btc - self.btc_inicial) / self.btc_inicial) * 100
        profit_pct = ((self.net_worth - self.balance_inicial) / self.balance_inicial) * 100
        ratio_ganadoras = (self.ventas_ganadoras / self.total_sells) * 100 if self.total_sells > 0 else 0.0
        ratio_perdedoras = (self.ventas_perdedoras / self.total_sells) * 100 if self.total_sells > 0 else 0.0
        trades = np.array(self.historial_trades_pct) * 100
        avg_win = avg_loss = risk_reward = expectancy = 0.0
        
        if len(trades) > 0:
            wins = trades[trades > 0]
            losses = trades[trades <= 0]
            avg_win = np.mean(wins) if len(wins) > 0 else 0.0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0.0
            risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else (avg_win if avg_win != 0 else 0.0)
            
            win_r = self.ventas_ganadoras / self.total_sells
            expectancy = (win_r * avg_win) + ((1 - win_r) * avg_loss)
        sharpe_ratio = sortino_ratio = calmar_ratio = 0.0
        historial_np = np.array(self.historial_net_worth)
        
        if len(historial_np) > 1:
            denominador = historial_np[:-1].copy()
            denominador[denominador == 0] = 1e-9 
            returns = np.diff(historial_np) / denominador
            
            std_returns = np.std(returns)
            negative_returns = returns[returns < 0]
            std_negative = np.std(negative_returns) if len(negative_returns) > 0 else 0.0
            
            if std_returns > 0:
                sharpe_ratio = (np.mean(returns) / std_returns) * np.sqrt(35040) # Velas 15m, si se entrena con otro intervalo CAMBIAR ESTE VALOR
            if std_negative > 0:
                sortino_ratio = (np.mean(returns) / std_negative) * np.sqrt(35040)
        
        calmar_ratio = (profit_pct / (self.max_drawdown * 100)) if self.max_drawdown > 0 else 0.0
        
        # Benchmark (Buy & Hold y Alpha)
        bnh_roi_pct = ((precio_actual - self.precios[0]) / self.precios[0]) * 100
        alpha = profit_pct - bnh_roi_pct

        return {
            'net_worth': float(self.net_worth),
            'net_profit': float(self.net_worth - self.balance_inicial),
            'balance': float(self.balance),
            'crypto_held': float(self.crypto_held),
            'step': int(self.current_step),
            'precio_actual': float(precio_actual),
            'operacion': operacion,
            
            'profit_pct': float(profit_pct),
            'profit_realizado': float(profit_realizado),
            'net_worth_btc': float(nw_btc),
            'profit_btc_pct': float(profit_btc_pct),
            
            'total_buys': self.total_buys,
            'total_sells': self.total_sells,
            'total_holds': self.total_holds,
            'total_trades': self.total_sells, 
            'trades_per_month': float((self.total_sells / self.current_step) * 2880) if self.current_step > 0 else 0.0,
            
            'ventas_ganadoras': self.ventas_ganadoras,
            'ventas_perdedoras': self.ventas_perdedoras,
            'ratio_ganadoras': float(ratio_ganadoras),
            'ratio_perdedoras': float(ratio_perdedoras),
            'win_rate': float(ratio_ganadoras),
            
            'average_win_pct': float(avg_win),
            'average_loss_pct': float(avg_loss),
            'risk_reward_ratio': float(risk_reward),
            'expectancy_pct': float(expectancy),
            
            'peak_net_worth': float(self.peak_net_worth),
            'max_drawdown': float(self.max_drawdown * 100),
            'current_drawdown': float(((self.peak_net_worth - self.net_worth) / self.peak_net_worth) * 100),
            
            'max_winning_streak': self.max_win_streak,
            'max_losing_streak': self.max_lose_streak,
            
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'calmar_ratio': float(calmar_ratio),
            
            'buy_and_hold_roi': float(bnh_roi_pct),
            'alpha_vs_bnh': float(alpha),
            
            'mejor_trade_pct': float(self.mejor_trade_pct * 100),
            'peor_trade_pct': float(self.peor_trade_pct * 100),
            'fecha_mejor_trade': str(self.fecha_mejor_trade) if self.fecha_mejor_trade else None,
            'fecha_peor_trade': str(self.fecha_peor_trade) if self.fecha_peor_trade else None,
            
            'bonus': float(bonus_puntuacion),
            'gross_profit': float(self.gross_profit),
            'gross_loss': float(self.gross_loss),
            'profit_factor': float(self.gross_profit / self.gross_loss) if self.gross_loss > 0 else 999.0,
            'holding_time': float(sum(self.tiempos_operacion) / len(self.tiempos_operacion)) if self.tiempos_operacion else 0.0
        }
    
    def render(self, mode="human"):
        roi = ((self.net_worth - self.balance_inicial) / self.balance_inicial) * 100
        print(f"[{self.current_step}/{self.max_steps}] "
              f"NetWorth: ${self.net_worth:.2f} | "
              f"Balance: ${self.balance:.2f} | "
              f"BTC: {self.crypto_held:.8f} | "
              f"ROI: {roi:.2f}%")

class TradingEnv(TradingEnv_v2):
    pass