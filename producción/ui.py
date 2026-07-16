import os
import sys
import time
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sb3_contrib import RecurrentPPO
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, FloatPrompt, IntPrompt
from rich.text import Text

from lstm_controller import consultar_modelo, descargar_velas_binance, preparar_secuencia_lstm, RUTA_MODELO
from envs.trading_env_v2 import TradingEnv
from compraVenta import crear_cliente, monitorProd, API_KEY, auditorProd, ejecutar_orden_circuit_breaker
from utils.config_params import get_api_secret_test
from evaluador import main as ejecutar_evaluacion

console = Console()

def mostrar_cabecera():
    os.system('cls' if os.name == 'nt' else 'clear')
    titulo = Text("SISTEMA DE TRADING ALGORÍTMICO", justify="center", style="bold cyan")
    console.print(Panel(titulo, border_style="cyan"))

def menu_principal():
    mostrar_cabecera()
    console.print("\n[1][bold yellow]Evaluación con 7 periodos definidos[/bold yellow]")
    console.print("[2][bold green]Paper Trading en Vivo[/bold green]")
    console.print("[3][bold magenta]Pedir veredicto inmediato al agente[/bold magenta]")
    console.print("[4][bold red]Salir[/bold red]\n")
    
    opcion = Prompt.ask("Selecciona una opción", choices=["1", "2", "3", "4"])
    
    if opcion == "1": ejecutar_backtesting()
    elif opcion == "2": ejecutar_paper_trading()
    elif opcion == "3": ejecutar_senal_instantanea()
    elif opcion == "4": exit()

def ejecutar_backtesting():
    mostrar_cabecera()
    console.print("[bold yellow]Iniciando Evaluación de 7 periodos...[/bold yellow]")
    try:
        ejecutar_evaluacion()
    except Exception as e:
        console.print(f"[bold red]Error durante la evaluación: {e}[/bold red]")
    Prompt.ask("\nPresiona Enter para volver")
    menu_principal()

def ejecutar_paper_trading():
    mostrar_cabecera()
    console.print("[bold cyan]Conectando con Binance Testnet y verificando saldo...[/bold cyan]")
    client = crear_cliente(API_KEY, get_api_secret_test(), is_testnet=True)
    account = client.get_account()
    
    usdt_real = 0.0
    btc_real = 0.0
    
    # Para probar el paper trading sin el BTC que nos da Binace, si se quiere usar el saldo real, cambiar virtual por real
    usdt_virtual = 10000.0
    btc_virtual = 0.0
    
    # Si no se quiere usar la pseudo cartera porque se quiere probar con la cartera real, borrar este bucle y cambiar las variables usdt_virtual y btc_virtual por usdt_real y btc_real
    if os.path.exists("operaciones_por_agente.csv"):
        import csv
        with open("operaciones_por_agente.csv", "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) > 0 and isinstance(row[0], str) and "sin 1 BTC" in row[0]:
                    usdt_virtual = 10000.0
                    btc_virtual = 0.0
                    continue
                
                if len(row) >= 4:
                    accion = row[1]
                    try:
                        precio = float(row[2])
                        cantidad = float(row[3])
                        if accion == "BUY":
                            btc_virtual += cantidad
                            usdt_virtual -= (cantidad * precio * 1.001) # Restamos USDT + comisión
                        elif accion == "SELL":
                            btc_virtual -= cantidad
                            usdt_virtual += (cantidad * precio * 0.999) # Sumamos USDT - comisión
                    except ValueError:
                        continue
    
    for b in account['balances']:
        if b['asset'] == 'USDT':
            usdt_real = float(b['free'])
        elif b['asset'] == 'BTC':
            btc_real = float(b['free'])
    
    console.print(f"[bold green]Saldo de la Pseudocartera Aislada:[/bold green] {usdt_virtual:.2f} USDT | {btc_virtual:.6f} BTC")        
    console.print(f"[bold green]Saldo disponible en Binance:[/bold green] {usdt_real:.2f} USDT | {btc_real:.6f} BTC")
    
    # 2. El usuario elige cuánto capital asignar al bot, cambiar virtual por real con un mínimo de 1000 tope_defecto = min(1000.0, usdt_real)
    tope_defecto = min(usdt_virtual, usdt_real)
    dinero_asignado = FloatPrompt.ask("¿Cuánto USDT de la pseudocartera quieres que el bot invierta?", default=tope_defecto)
    # dinero_asignado = FloatPrompt.ask("¿Cuánto USDT quieres que el bot invierta?", default=tope_defecto)
    
    if dinero_asignado > usdt_virtual:
        console.print(f"[bold yellow]No tienes suficiente USDT virtual. Se asignará tu máximo: {usdt_virtual:.2f}[/bold yellow]")
        dinero_asignado = usdt_virtual
        
    balance_usd = dinero_asignado
    crypto_held = btc_virtual # Cambiar a btc_real si se quiere usar el saldo real
    
    df_inicial = descargar_velas_binance(client, limit=1)
    precio_arranque = df_inicial.iloc[-1]['Close']
    net_worth_inicial = balance_usd + (crypto_held * precio_arranque)
    
    monitorProd.set_balance_inicial(net_worth_inicial)
    
    console.print(f"\n[bold green]Iniciando Paper Trading con un capital asignado de {balance_usd:.2f} USDT...[/bold green]")
    console.print("[yellow]Presiona Ctrl+C en cualquier momento para detener el bot y volver al menú.[/yellow]\n")
    
    dinero = net_worth_inicial # Lo guardamos para el cálculo del ROI
    comision = 0.001
    spread = 0.0005
    model = RecurrentPPO.load(RUTA_MODELO)
    lstm_state = None
    episode_starts = np.ones((1,), dtype=bool)

    try:
        while True:
            hora_actual = time.strftime("%Y-%m-%d %H:%M:%S")
            console.print(f"\n[cyan]--- Evaluación del Mercado: {hora_actual} ---[/cyan]")
            df = descargar_velas_binance(client, limit=250)
            precio_actual = df.iloc[-1]['Close']
            
            secuencia_obs = preparar_secuencia_lstm(df)
            
            lstm_state = None
            episode_starts = np.ones((1,), dtype=bool)
            
            for obs in secuencia_obs[:-1]:
                obs_completa = obs.copy()
                _, lstm_state = model.predict(obs_completa.reshape(1, -1), state=lstm_state, episode_start=episode_starts)
                episode_starts = np.zeros((1,), dtype=bool)
            
            net_worth = balance_usd + (crypto_held * precio_actual)
            pnl = 0.0
            balance_change = (balance_usd - dinero) / (dinero + 1e-8)
            crypto_value = crypto_held * precio_actual
            crypto_change = crypto_value / (dinero + 1e-8)
            net_worth_change = (net_worth - dinero) / (dinero + 1e-8)
            
            obs_actual = secuencia_obs[-1].copy()
            obs_actual[-4:] = np.array([
                np.clip(balance_change, -10, 10),
                np.clip(crypto_change, -10, 10),
                np.clip(net_worth_change, -10, 10),
                np.clip(pnl, -1, 1)
            ], dtype=np.float32)
            
            accion_predicha, lstm_state = model.predict(
                obs_actual.reshape(1, -1),
                state=lstm_state, 
                episode_start=episode_starts, 
                deterministic=True
            )
            episode_starts = np.zeros((1,), dtype=bool)
            accion_final = accion_predicha[0]

            tipo_operacion = accion_final[0]
            porcentaje_operacion = np.clip(accion_final[1], 0.01, 1.0) if len(accion_final) > 1 else 1.0
            console.print(f"[dim]Porcentaje solicitado por el modelo: {porcentaje_operacion:.4f}[/dim]")
            
            precio_compra = precio_actual * (1 + spread)
            precio_venta = precio_actual * (1 - spread)
            
            if not monitorProd.es_seguro_operar(balance_usd, net_worth, 0): # Pasamos 0 porque es una comprobación general
                console.print("[bold red]Circuit Breaker Activado. Deteniendo operativa virtual para proteger capital.[/bold red]")
                break

            if tipo_operacion > 0.33 and balance_usd > 10:
                NUM_SACOS = 5
                REFERENCIA_ENTRENAMIENTO = 10000.0
                tamano_saco = REFERENCIA_ENTRENAMIENTO / NUM_SACOS   # 2.000 USD, igual que entrenamiento
                gasto_deseado = tamano_saco * porcentaje_operacion
                
                # REGLA NOTIONAL DE BINANCE: Forzamos un mínimo de 10 USDT
                if gasto_deseado < 10.0:
                    gasto_deseado = 10.0
                
                cantidad_gastar = min(gasto_deseado, balance_usd)
                cantidad_comprada = (cantidad_gastar / (1 + comision)) / precio_compra
                
                console.print("[bold green]Enviando orden COMPRA a Binance Testnet...[/bold green]")
                order_id = ejecutar_orden_circuit_breaker(client, "BUY", "BTC", "USDT", cantidad_comprada, precio_compra, balance_usd, net_worth)
                
                # Solo actualizamos el saldo si Binance devuelve un order_id válido
                if order_id is not None:
                    balance_usd -= cantidad_gastar
                    crypto_held += cantidad_comprada
                    console.print(f"[bold green]COMPRA EJECUTADA EN BINANCE:[/bold green] {cantidad_comprada:.5f} BTC a {precio_compra:.2f} $ (Fuerza interna: {tipo_operacion:.4f})")
                else:
                    console.print("[bold red]Orden rechazada por Binance. Saldo virtual intacto.[/bold red]")
                
            elif tipo_operacion < -0.33 and crypto_held > 1e-5:
                cantidad_vender = crypto_held * porcentaje_operacion
                ingresos_brutos_estimados = cantidad_vender * precio_venta
                
                if ingresos_brutos_estimados < 10.0:
                    if (crypto_held * precio_venta) > 10.0:
                        cantidad_vender = 10.0 / precio_venta
                    else:
                        cantidad_vender = crypto_held # Vende las migajas que queden
                
                console.print("[bold red]Enviando orden VENTA a Binance Testnet...[/bold red]")
                order_id = ejecutar_orden_circuit_breaker(client, "SELL", "BTC", "USDT", cantidad_vender, precio_venta, balance_usd, net_worth)
                
                if order_id is not None:
                    ingresos_brutos = cantidad_vender * precio_venta
                    ingresos_netos = ingresos_brutos * (1 - comision)
                    balance_usd += ingresos_netos
                    crypto_held -= cantidad_vender
                    console.print(f"[bold red]VENTA EJECUTADA EN BINANCE:[/bold red] {cantidad_vender:.5f} BTC a {precio_venta:.2f} $ (Fuerza interna: {tipo_operacion:.4f})")
                else:
                    console.print("[bold red]Orden rechazada por Binance. Saldo virtual intacto.[/bold red]")
                
            elif tipo_operacion < -0.33 and crypto_held > 1e-5:
                cantidad_vender = crypto_held * porcentaje_operacion
                
                console.print("[bold red]Enviando orden VENTA a Binance Testnet...[/bold red]")
                ejecutar_orden_circuit_breaker(client, "SELL", "BTC", "USDT", cantidad_vender, precio_venta, balance_usd, net_worth)
                
                ingresos_brutos = cantidad_vender * precio_venta
                ingresos_netos = ingresos_brutos * (1 - comision)
                
                balance_usd += ingresos_netos
                crypto_held -= cantidad_vender
                
                console.print(f"[bold red]VENTA EJECUTADA:[/bold red] {cantidad_vender:.6f} BTC a {precio_venta:.2f} $ (Fuerza interna: {tipo_operacion:.4f})")
                
            else:
                console.print(f"[bold white]ESPERAR:[/bold white] El agente decide mantener posiciones. (Fuerza interna: {tipo_operacion:.4f})")
                try:
                    auditorProd.registrar(
                        accion="HOLD",
                        precio=precio_actual,
                        cantidad=0.0,
                        estado=tipo_operacion,
                        razon=f"LSTM"
                    )
                except Exception as e:
                    console.print(f"[dim red]No se pudo registrar el HOLD en el CSV: {e}[/dim red]")
            console.print(f"[bold]Portfolio:[/bold] {balance_usd:.2f} USD | {crypto_held:.6f} BTC | [bold yellow]Net Worth: {net_worth:.2f} USD[/bold yellow]")
            console.print("[dim]Durmiendo hasta el próximo cierre de vela (15 min)...[/dim]")
            time.sleep(900)

    except KeyboardInterrupt:
        console.print("\n[bold yellow]Paper Trading detenido manualmente por el usuario.[/bold yellow]")
        
    Prompt.ask("Presiona Enter para volver al menú principal")
    menu_principal()

def ejecutar_senal_instantanea():
    mostrar_cabecera()
    console.print("[bold magenta]Consultando el mercado ahora mismo...[/bold magenta]")
    consultar_modelo()
    Prompt.ask("\nPresiona Enter para volver")
    menu_principal()

def mostrar_resultados(capital, info):
    tabla = Table(title="RESULTADOS")
    tabla.add_column("Métrica")
    tabla.add_column("Valor")
    tabla.add_row("Capital Inicial", f"{capital:.2f}")
    tabla.add_row("Capital Final", f"{info['net_worth']:.2f}")
    tabla.add_row("ROI", f"{info['profit_pct']:.2f}%")
    console.print(tabla)
    Prompt.ask("Presiona Enter para volver")
    menu_principal()

if __name__ == "__main__":
    menu_principal()
