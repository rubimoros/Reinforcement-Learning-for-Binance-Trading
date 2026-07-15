import logging
from datetime import datetime
from binance.client import Client
from utils.config_params import get_api_key_test, get_api_secret_test, get_api_url_base_test

base_url = get_api_url_base_test()
api_key = get_api_key_test()
private_key = get_api_secret_test()

# Inicializar cliente python-binance
a_client = Client(api_key, private_key)
if base_url:
    # 1. Limpieza inteligente de la URL
    clean_url = base_url.rstrip('/')
    if not clean_url.endswith('/api'):
        clean_url += '/api'
    
logging.basicConfig(filename='binance.log',
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

def klines_example(symbol):
    client = a_client

    # Tiempo del servidor
    try:
        server_time = client.get_server_time()
        logging.info(parse_server_time(server_time))
    except Exception as e:
        logging.error(f"Error obteniendo server time: {e}")

    # Datos de cuenta
    try:
        data = client.get_account()
    except Exception as e:
        logging.error(f"Error obteniendo cuenta: {e}")
        data = {}

    if data:
        logging.info(get_comissions(client, symbol))
        try:
            logging.info(get_balance(data, "BTC"))
            logging.info(get_balance(data, "USDC"))
        except StopIteration:
            logging.warning("Activo BTC o USDC no encontrado en balances")

    # Profundidad de mercado
    try:
        logging.info(client.get_order_book(symbol=symbol, limit=1))
    except Exception as e:
        logging.error(f"Error depth: {e}")

    # Trades propios
    try:
        logging.info(client.get_my_trades(symbol=symbol))
    except Exception as e:
        logging.error(f"Error my_trades: {e}")

    # Rate limits (extraído de exchange_info)
    try:
        ex_info = client.get_exchange_info()
        rate_limits = ex_info.get('rateLimits', [])
        logging.info({"rateLimits": rate_limits})
    except Exception as e:
        logging.error(f"Error obteniendo exchange info: {e}")

    # Comisiones por par de monedas (trade fee)
    try:
        fee = client.get_trade_fee(symbol=symbol)
        logging.info({"tradeFee": fee})
    except Exception as e:
        logging.error(f"Error trade fee: {e}")

    interval = "1m"
    limit = 1

    candles = []
    try:
        candles = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    except Exception as e:
        logging.error(f"Error obteniendo klines: {e}")

    return candles

def draw_candles(candles, symbol):
    for c in candles:
        open_time = c[0]
        open_price = float(c[1])
        high_price = float(c[2])
        low_price = float(c[3])
        close_price = float(c[4])
        volume = float(c[5])
        open_dt = datetime.fromtimestamp(open_time / 1000.0)
        print(f"{symbol} {open_dt} - Open: {open_price}, Close: {close_price}, High: {high_price}, Low: {low_price}, Volume: {volume}, c:{c}")


def date_to_millis(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return int(dt.timestamp() * 1000)


def klines_example_time(symbol):
    client = a_client
    interval = "15m"

    start_str = "2023-01-01"
    end_str = "2024-01-01"

    start_time = date_to_millis(start_str)
    end_time = date_to_millis(end_str)

    data = []
    limit = 1000

    while True:
        try:
            klines = client.get_klines(
                symbol=symbol,
                interval=interval,
                startTime=start_time,
                endTime=end_time,
                limit=limit
            )
        except Exception as e:
            logging.error(f"Error descargando klines: {e}")
            break

        if not klines:
            break

        data.extend(klines)
        last_close_time = klines[-1][6]
        start_time = last_close_time + 1
        if last_close_time >= end_time:
            break

    print(f"Total velas descargadas: {len(data)}")
    return data

def klines_example_time_tfg(symbol, start_str="2023-01-01", end_str="2024-01-01", interval="1h"):

    # Para entrenar IA necesitamos histórico real, la Testnet no tiene esos datos.
    # Usamos un cliente anónimo (sin claves) apuntando a Producción. Es gratis y público.
    # 1. Cliente anónimo (None, None) apuntando a la URL oficial
    client = Client(None, None)
    client.API_URL = 'https://api.binance.com/api' 
    start_time = date_to_millis(start_str)
    end_time = date_to_millis(end_str)

    data = []
    limit = 1000 

    while True:
        try:
            # get_klines es público, no requiere firma
            klines = client.get_klines(
                symbol=symbol,
                interval=interval,
                startTime=start_time,
                endTime=end_time,
                limit=limit
            )
        except Exception as e:
            print(f"Error descargando klines: {e}")
            break

        if not klines:
            print("Binance devolvió una lista vacía. Fin de la descarga.")
            break

        data.extend(klines)
        print(f"   ... Descargadas {len(klines)} velas nuevas (Total: {len(data)})")
        
        # Avanzamos el tiempo para la siguiente petición
        last_close_time = klines[-1][6]
        start_time = last_close_time + 1
        
        if last_close_time >= end_time:
            break
        
        # Pequeña pausa para ser educados con el servidor
        import time
        time.sleep(0.1)

    print(f"FINALIZADO. Total velas descargadas: {len(data)}")
    return data


def parse_server_time(server_time):
    timestamp_ms = server_time['serverTime']
    timestamp_s = timestamp_ms / 1000
    dt = datetime.utcfromtimestamp(timestamp_s)
    formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S,') + f"{int(timestamp_ms % 1000):03d}"
    return formatted_time


def get_comissions(client: Client, symbol: str):
    try:
        fee = client.get_trade_fee(symbol=symbol)
        return fee
    except Exception as e:
        return {"trade_fee_error": str(e)}


def get_balance(data, coin):
    return next(item for item in data.get('balances', []) if item['asset'] == coin)
