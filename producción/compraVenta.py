import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from binance.client import Client
import requests
from binance.exceptions import BinanceAPIException
import time
from binance.enums import *
from datetime import datetime
import threading
import json
from utils.config_params import get_api_key_test, get_api_url_base_test, get_api_url_base_prod
from circuit_breaker import CircuitBreaker
from operaciones import OperacionesLogger

client = None
API_KEY = get_api_key_test()
BASE_URL = get_api_url_base_test()
lock_ordenes = threading.Lock()
INTERVALO_CONSULTA_ORDENES = 3
FEE_TAKER = 0.001
FEE_MAKER = 0.001
ordenes_maker_activas = []
monitorProd = CircuitBreaker()
auditorProd = OperacionesLogger()

def consultar_precios(client: Client, symbol: str) -> float | None:
    try:
        ticker = client.get_symbol_ticker(symbol=symbol)
        price = "{:.8f}".format(float(ticker['price'])).rstrip('0').rstrip('.')
        print(f"El precio actual de {symbol} es {price}")
        return price
    except BinanceAPIException as e:
        print(f"Error al obtener el precio de {symbol}: {e.message}")
        return None


def ejecutar_orden_circuit_breaker(client, accion, base_moneda, moneda_cambio, cantidad, precio, balance, net_worth):
    if monitorProd.es_seguro_operar(balance, net_worth, cantidad):
        if accion == "BUY":
            estado_orden = comprar_moneda(client, base_moneda, moneda_cambio, cantidad, modo="maker", precio_deseado=precio)
        else:
            estado_orden = vender_moneda(client, base_moneda, moneda_cambio, cantidad, modo="maker", precio_deseado=precio)
        auditorProd.registrar(accion, precio, cantidad, str(estado_orden), "LSTM")
        return estado_orden
    else:
        auditorProd.registrar(accion, precio, cantidad, "BLOQUEADO", "Riesgo excesivo: Drawdown o Capital superado")
        return None


def consultar_transacciones(client: Client):
    transactions = client.get_my_trades(symbol="IOTAETH")
    print("\n--- TRANSACCIONES RECIENTES ---")
    for trans in transactions:
        timestamp = trans['time'] / 1000
        date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        print(f"Fecha: {date}, Precio: {trans['price']}, Cantidad: {trans['qty']}, Comisiones: {trans['commission']}")

def consultar_saldos(client: Client):
    account = client.get_account()
    balances = {b['asset']: b['free'] for b in account['balances'] if float(b['free']) > 0}
    print("\n--- SALDOS DISPONIBLES ---")
    for asset, amount in balances.items():
        print(f"{asset}: {amount}")

def consultar_comisiones(client: Client, symbol: str):
    print("\n--- COMISIONES DE TRADING ---")
    fees = client.get_trade_fee(symbol=symbol)
    for fee in fees:
        print(f"Símbolo: {fee['symbol']} - Maker: {fee['makerCommission']}, Taker: {fee['takerCommission']}")

def crear_cliente(api_key: str, api_secret: str, is_testnet: bool = True) -> Client:
    global client
    client = Client(api_key, api_secret, testnet=is_testnet)
    
    server_time = client.get_server_time()['serverTime']
    print(f"Hora del servidor Binance: {server_time}")
    
    local_time = int(time.time() * 1000)
    global TIME_OFFSET
    TIME_OFFSET = server_time - local_time
    
    return client


def manejar_evento_compra(estado, symbol, cantidad, precio):
    print(f"✅ ORDEN COMPRA completada: {cantidad} {symbol} a {precio}")

def manejar_evento_venta(estado, symbol, cantidad, precio):
    print(f"✅ ORDEN VENTA completada: {cantidad} {symbol} a {precio}")

def manejar_evento_orden(client: Client, order_id: int, symbol: str, side: str, max_espera=10000):
    tiempo_esperado = 0
    contador = 0
    while tiempo_esperado < max_espera:
        contador += 1
        try:
            orden = client.get_order(symbol=symbol, orderId=order_id)
            estado = orden['status']

            precio = float(orden.get('price', 0))
            cantidad = float(orden.get('origQty', 0))

            if estado == 'FILLED':
                if side == 'BUY':
                    manejar_evento_compra(estado, symbol, cantidad, precio)
                elif side == 'SELL':
                    manejar_evento_venta(estado, symbol, cantidad, precio)
                return

            elif estado == 'CANCELED':
                print(f"❌ Orden {side} cancelada.")
                return

            else:
                print(f"{contador} ⏳ Orden {side} aún no completada (estado: {estado})... esperando {INTERVALO_CONSULTA_ORDENES}s")
                time.sleep(INTERVALO_CONSULTA_ORDENES)
                tiempo_esperado += INTERVALO_CONSULTA_ORDENES

        except BinanceAPIException as e:
            print(f"❌ Error al verificar estado de la orden: {e.message}")
            return

    print(f"⚠️ Tiempo máximo de espera alcanzado ({max_espera}s). La orden sigue en estado pendiente.")




def comprar_moneda(client: Client, base_moneda: str, moneda_cambio: str, cantidad: float,
                   modo: str = "market", precio_deseado: float = None):
    global ordenes_maker_activas

    try:
        symbol = base_moneda.upper() + moneda_cambio.upper()

        if modo.lower() == "maker":
            if precio_deseado is None:
                print("❌ Debes proporcionar un precio_deseado para una orden LIMIT (maker).")
                return None
            
            cantidad_str = "{:.5f}".format(cantidad)
            precio_str = "{:.2f}".format(precio_deseado)

            print(f"\n[MAKER] Enviando orden LIMIT para comprar {cantidad} {base_moneda.upper()} a {precio_deseado} {moneda_cambio.upper()}")
            orden = client.create_order(
                symbol=symbol,
                side=SIDE_BUY,
                type=ORDER_TYPE_LIMIT,
                timeInForce=TIME_IN_FORCE_GTC,
                quantity=cantidad_str,
                price=precio_str
            )

            order_id = orden['orderId']
            ordenes_maker_activas.append({
                'order_id': order_id,
                'symbol': symbol,
                'side': 'BUY',
                'cantidad': cantidad,
                'precio': precio_deseado
            })

            manejar_evento_orden(client, order_id, symbol, 'BUY')
            return order_id

        elif modo.lower() == "taker":
            print(f"\n[TAKER] Enviando orden MARKET para comprar {cantidad} {base_moneda.upper()}")
            orden = client.create_order(
                symbol=symbol,
                side=SIDE_BUY,
                type=ORDER_TYPE_MARKET,
                quantity=cantidad
            )

            order_id = orden['orderId']
            manejar_evento_orden(client, order_id, symbol, 'BUY')
            return order_id

        else:
            print("⚠️ Modo inválido. Usa 'maker' o 'taker'.")
            return None

    except BinanceAPIException as e:
        print(f"❌ Error al comprar {base_moneda.upper()}: {e.message}")
        return None



def vender_moneda(client: Client, base_moneda: str, moneda_cambio: str, cantidad: float,
                  modo: str = "market", precio_deseado: float = None):

    try:
        symbol = base_moneda.upper() + moneda_cambio.upper()

        if modo.lower() == "maker":
            if precio_deseado is None:
                print("❌ Debes proporcionar un precio_deseado para una orden LIMIT (maker).")
                return None

            cantidad_str = "{:.5f}".format(cantidad)
            precio_str = "{:.2f}".format(precio_deseado)

            print(f"\n[MAKER] Enviando orden LIMIT para vender {cantidad} {base_moneda.upper()} a {precio_deseado} {moneda_cambio.upper()}")
            orden = client.create_order(
                symbol=symbol,
                side=SIDE_SELL,
                type=ORDER_TYPE_LIMIT,
                timeInForce=TIME_IN_FORCE_GTC,
                quantity=cantidad_str,
                price=precio_str
            )

            order_id = orden['orderId']
            ordenes_maker_activas.append({
                'order_id': order_id,
                'symbol': symbol,
                'side': 'SELL',
                'cantidad': cantidad,
                'precio': precio_deseado
            })

            manejar_evento_orden(client, order_id, symbol, 'SELL')
            return order_id

        elif modo.lower() == "taker":
            print(f"\n[TAKER] Enviando orden MARKET para vender {cantidad} {base_moneda.upper()}")
            orden = client.create_order(
                symbol=symbol,
                side=SIDE_SELL,
                type=ORDER_TYPE_MARKET,
                quantity=cantidad
            )

            order_id = orden['orderId']
            manejar_evento_orden(client, order_id, symbol, 'SELL')
            return order_id

        else:
            print("⚠️ Modo inválido. Usa 'maker' o 'taker'.")
            return None

    except BinanceAPIException as e:
        print(f"❌ Error al vender {base_moneda.upper()}: {e.message}")
        return None



def cancelar_ordenes_maker(client: Client):
    global ordenes_maker_activas

    if not ordenes_maker_activas:
        print("No hay órdenes maker activas para cancelar.")
        return
    elif not client:
        print("El client ha fallado o no existe")
        return

    for orden in ordenes_maker_activas:
        try:
            client.cancel_order(symbol=orden['symbol'], orderId=orden['order_id'])
            print(f"❌ Orden LIMIT cancelada: ID {orden['order_id']} - {orden['symbol']}")
        except BinanceAPIException as e:
            print(f"Error al cancelar orden {orden['order_id']}: {e.message}")

    ordenes_maker_activas.clear()




