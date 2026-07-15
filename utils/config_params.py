import configparser
import os

# Obtener la ruta del directorio donde está el script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Crear la ruta completa al archivo config.ini en el mismo directorio
config_path = os.path.join(script_dir, 'config.ini')

# Crear un objeto de configuración
config = configparser.ConfigParser()

# Leer el archivo .ini
config.read(config_path)

# Acceder a las secciones y valores
def get_api_key_prod():
    return config['keys']['api_key_prod']

def get_api_secret_prod():
    return config['keys']['api_secret_prod']

def get_api_url_base_prod():
    return config['keys']['url_base_api_prod']

def get_api_url_ws_stream_klines_prod(symbol, interval):
    return config['keys']['url_ws_stream_klines_prod'].replace("symbol",symbol.lower()).replace("interval",interval)

def get_api_url_ws_stream_trade_prod(symbol):
    return config['keys']['url_ws_stream_trade_prod'].replace("symbol",symbol.lower())

##########################################################3

def get_api_key_test():
    return config['keys']['api_key_test']

def get_api_secret_test():
    return config['keys']['api_secret_test']

def get_api_url_base_test():
    return config['keys']['url_base_api_test']

def get_api_url_ws_stream_klines_test(symbol, interval):
    return config['keys']['url_ws_stream_klines_test'].replace("symbol",symbol.lower()).replace("interval",interval)

def get_api_url_ws_stream_trade_test(symbol):
    return config['keys']['url_ws_stream_trade_test'].replace("symbol",symbol.lower())

##########################################################3

"""
===============================================================
 BINANCE WEBSOCKET ENDPOINTS
===============================================================
- Sustituye:
    <symbol>   -> par de trading en minúsculas, ej. 'btcusdt'
    <interval> -> intervalo de velas: 1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w,1M
===============================================================
"""

# ==================== SPOT (MAINNET) ====================
BINANCE_SPOT_WSS = {
    # --- Trades ---
    # Trade individual cada vez que se ejecuta (tick-by-tick)
    "TRADE":           "wss://stream.binance.com:9443/ws/<symbol>@trade",

    # Trades agregados (menos mensajes que @trade)
    "AGG_TRADE":       "wss://stream.binance.com:9443/ws/<symbol>@aggTrade",

    # --- Velas (OHLCV) ---
    # Velas de tiempo real: reemplaza <interval>
    "KLINE":           "wss://stream.binance.com:9443/ws/<symbol>@kline_<interval>",

    # --- Tickers ---
    # Estadísticas 24h de un par
    "TICKER_24H":      "wss://stream.binance.com:9443/ws/<symbol>@ticker",

    # Mini estadísticas 24h de un par
    "MINI_TICKER":     "wss://stream.binance.com:9443/ws/<symbol>@miniTicker",

    # Tickers de TODOS los pares en un único stream
    "ALL_TICKERS":     "wss://stream.binance.com:9443/ws/!ticker@arr",
    "ALL_MINI_TICKERS":"wss://stream.binance.com:9443/ws/!miniTicker@arr",

    # --- Order Book / Profundidad ---
    # Actualizaciones incrementales de profundidad (100ms por defecto)
    "DEPTH":           "wss://stream.binance.com:9443/ws/<symbol>@depth",

    # Top niveles del order book
    "DEPTH5":          "wss://stream.binance.com:9443/ws/<symbol>@depth5",
    "DEPTH10":         "wss://stream.binance.com:9443/ws/<symbol>@depth10",
    "DEPTH20":         "wss://stream.binance.com:9443/ws/<symbol>@depth20",

    # Mejor oferta y demanda (best bid/ask)
    "BOOK_TICKER":     "wss://stream.binance.com:9443/ws/<symbol>@bookTicker",

    # --- User Data ---
    # Requiere listenKey obtenido por REST. Envía eventos de cuenta y órdenes
    "USER_DATA":       "wss://stream.binance.com:9443/ws/<listenKey>",

    # --- Multiplex ---
    # Permite combinar varios streams en una conexión
    # Ej: wss://stream.binance.com:9443/stream?streams=btcusdt@trade/ethusdt@kline_1m
    "MULTI_STREAM":    "wss://stream.binance.com:9443/stream?streams=<stream1>/<stream2>"
}

# ==================== SPOT (TESTNET) ====================
BINANCE_SPOT_TESTNET_WSS = {
    key: value.replace("stream.binance.com:9443", "testnet.binance.vision")
    for key, value in BINANCE_SPOT_WSS.items()
}

# ==================== FUTURES USDT-M (MAINNET) ====================
BINANCE_FUTURES_WSS = {
    # --- Trades ---
    "TRADE":           "wss://fstream.binance.com/ws/<symbol>@trade",
    "AGG_TRADE":       "wss://fstream.binance.com/ws/<symbol>@aggTrade",

    # --- Velas ---
    "KLINE":           "wss://fstream.binance.com/ws/<symbol>@kline_<interval>",

    # --- Mark Price & Funding ---
    # Precio de marca y próximo funding
    "MARK_PRICE":      "wss://fstream.binance.com/ws/<symbol>@markPrice",

    # Liquidaciones forzadas
    "FORCE_ORDER":     "wss://fstream.binance.com/ws/<symbol>@forceOrder",
    # Todas las liquidaciones
    "ALL_FORCE_ORDER": "wss://fstream.binance.com/ws/!forceOrder@arr",

    # --- Order Book ---
    "DEPTH":           "wss://fstream.binance.com/ws/<symbol>@depth",
    "DEPTH5":          "wss://fstream.binance.com/ws/<symbol>@depth5",
    "DEPTH10":         "wss://fstream.binance.com/ws/<symbol>@depth10",
    "DEPTH20":         "wss://fstream.binance.com/ws/<symbol>@depth20",

    # Mejor bid/ask
    "BOOK_TICKER":     "wss://fstream.binance.com/ws/<symbol>@bookTicker",

    # --- User Data ---
    "USER_DATA":       "wss://fstream.binance.com/ws/<listenKey>",

    # --- Multiplex ---
    "MULTI_STREAM":    "wss://fstream.binance.com/stream?streams=<stream1>/<stream2>"
}

# ==================== FUTURES USDT-M (TESTNET) ====================
BINANCE_FUTURES_TESTNET_WSS = {
    key: value.replace("fstream.binance.com", "stream.binancefuture.com")
    for key, value in BINANCE_FUTURES_WSS.items()
}

# Ejemplo de uso:
# url = BINANCE_SPOT_WSS["KLINE"].replace("<symbol>", "btcusdt").replace("<interval>", "1m")
# print(url)  # wss://stream.binance.com:9443/ws/btcusdt@kline_1m


