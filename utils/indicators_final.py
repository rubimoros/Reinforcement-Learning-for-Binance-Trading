import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class TechnicalIndicators:
    @staticmethod
    def _replace_inf_nan(df: pd.DataFrame) -> pd.DataFrame:
        return df.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _clip_outliers(df: pd.DataFrame, lower_q: float = 0.001, upper_q: float = 0.999) -> pd.DataFrame:
        for col in df.columns:
            serie = df[col]
            if not np.issubdtype(serie.dtype, np.number):
                continue
            valid = serie.dropna()
            if len(valid) < 10:
                continue
            low = valid.quantile(lower_q)
            high = valid.quantile(upper_q)
            if low == high:
                continue
            df[col] = serie.where((serie >= low) & (serie <= high), np.nan)
        return df

    @staticmethod
    def _finalize(df: pd.DataFrame) -> pd.DataFrame:
        df = TechnicalIndicators._replace_inf_nan(df)
        df = TechnicalIndicators._clip_outliers(df)
        return df

    @staticmethod
    def calculate_sma(df: pd.DataFrame, period_1: int = 20, period_2: int = 80) -> pd.DataFrame:
        df_new = df.reset_index()
        df_new["Close"] = pd.to_numeric(df_new["Close"], errors="coerce")

        def calc_sma1():
            return df_new["Close"].rolling(window=period_1, min_periods=1).mean()

        def calc_sma2():
            return df_new["Close"].rolling(window=period_2, min_periods=1).mean()

        with ThreadPoolExecutor() as executor:
            future_1 = executor.submit(calc_sma1)
            future_2 = executor.submit(calc_sma2)
            sma1 = future_1.result()
            sma2 = future_2.result()

        df_new["INDICADOR_1"] = (sma1 - df_new["Close"]) / df_new["Close"]
        df_new["INDICADOR_2"] = (sma2 - df_new["Close"]) / df_new["Close"]

        out = df_new[["Time", "INDICADOR_1", "INDICADOR_2"]].set_index("Time")
        return TechnicalIndicators._finalize(out)

    @staticmethod
    def calculate_ema(df: pd.DataFrame, period_1: int = 20, period_2: int = 80) -> pd.DataFrame:
        df_new = df.reset_index()
        df_new["Close"] = pd.to_numeric(df_new["Close"], errors="coerce")

        def calc_ema1():
            return df_new["Close"].ewm(span=period_1, adjust=False).mean()

        def calc_ema2():
            return df_new["Close"].ewm(span=period_2, adjust=False).mean()

        with ThreadPoolExecutor() as executor:
            future_1 = executor.submit(calc_ema1)
            future_2 = executor.submit(calc_ema2)
            ema1 = future_1.result()
            ema2 = future_2.result()

        df_new["INDICADOR_1"] = (ema1 - df_new["Close"]) / df_new["Close"]
        df_new["INDICADOR_2"] = (ema2 - df_new["Close"]) / df_new["Close"]

        out = df_new[["Time", "INDICADOR_1", "INDICADOR_2"]].set_index("Time")
        return TechnicalIndicators._finalize(out)

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period_1: int = 14, period_2: int = 28) -> pd.DataFrame:
        df_new = df.reset_index()
        df_new["Close"] = pd.to_numeric(df_new["Close"], errors="coerce")

        df_new['Delta'] = df_new['Close'].diff()
        df_new['Gain'] = df_new['Delta'].where(df_new['Delta'] > 0, 0)
        df_new['Loss'] = -df_new['Delta'].where(df_new['Delta'] < 0, 0)

        def calc_rsi(period):
            avg_gain = df_new['Gain'].rolling(window=period, min_periods=period).mean()
            avg_loss = df_new['Loss'].rolling(window=period, min_periods=period).mean()
            rs = np.where(avg_loss == 0, np.nan, avg_gain / avg_loss)
            rsi = 100 - (100 / (1 + rs))
            rsi = np.where((avg_loss == 0) & (avg_gain > 0), 100, rsi)
            rsi = np.where((avg_gain == 0) & (avg_loss > 0), 0, rsi)
            return pd.Series(rsi, index=df_new.index)

        with ThreadPoolExecutor() as executor:
            f1 = executor.submit(calc_rsi, period_1)
            f2 = executor.submit(calc_rsi, period_2)
            df_new['INDICADOR_1'] = f1.result()
            df_new['INDICADOR_2'] = f2.result()

        out = df_new[['Time', 'INDICADOR_1', 'INDICADOR_2']].set_index("Time")
        return TechnicalIndicators._finalize(out)

    @staticmethod
    def calculate_macd(df: pd.DataFrame, short_period: int = 12, long_period: int = 26,
                       signal_period: int = 9) -> pd.DataFrame:
        df_loc = df.copy()
        df_loc["Close"] = pd.to_numeric(df_loc["Close"], errors="coerce")

        with ThreadPoolExecutor() as executor:
            f_ema_short = executor.submit(lambda: df_loc['Close'].ewm(span=short_period, adjust=False).mean())
            f_ema_long = executor.submit(lambda: df_loc['Close'].ewm(span=long_period, adjust=False).mean())

            ema_short = f_ema_short.result(); ema_long = f_ema_long.result()

        macd_line = ema_short - ema_long
        macd_signal = macd_line.ewm(span=signal_period, adjust=False).mean()

        df_loc['INDICADOR_1'] = macd_line / df_loc['Close']
        df_loc['INDICADOR_2'] = macd_signal / df_loc['Close']

        out = df_loc[['INDICADOR_1', 'INDICADOR_2']]
        return TechnicalIndicators._finalize(out)

    @staticmethod
    def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        df_new = df.reset_index()
        for col in ['High','Low','Close']:
            df_new[col] = pd.to_numeric(df_new[col], errors='coerce')
        df_new['UpMove'] = df_new['High'].diff()
        df_new['DownMove'] = df_new['Low'].diff()
        df_new['+DM'] = df_new.apply(lambda row: row['UpMove'] if row['UpMove'] > row['DownMove'] and row['UpMove'] > 0 else 0, axis=1)
        df_new['-DM'] = df_new.apply(lambda row: abs(row['DownMove']) if row['DownMove'] > row['UpMove'] and row['DownMove'] > 0 else 0, axis=1)
        df_new['prev_close'] = df_new['Close'].shift(1)
        df_new['TR'] = np.maximum(df_new['High'] - df_new['Low'],
                                   np.maximum(abs(df_new['High'] - df_new['prev_close']),
                                              abs(df_new['Low'] - df_new['prev_close'])))
        tr_roll = df_new['TR'].rolling(window=period, min_periods=period).sum()
        pdm_roll = df_new['+DM'].rolling(window=period, min_periods=period).sum()
        ndm_roll = df_new['-DM'].rolling(window=period, min_periods=period).sum()
        df_new['+DI'] = 100 * np.where(tr_roll==0, np.nan, pdm_roll / tr_roll)
        df_new['-DI'] = 100 * np.where(tr_roll==0, np.nan, ndm_roll / tr_roll)
        di_sum = df_new['+DI'] + df_new['-DI']
        df_new['DX'] = np.where(di_sum==0, np.nan, (np.abs(df_new['+DI'] - df_new['-DI']) / di_sum) * 100)
        df_new['INDICADOR_3'] = df_new['DX'].rolling(window=period, min_periods=period).mean()
        out = df_new[['Time','+DI','-DI','INDICADOR_3']].rename(columns={'+DI':'INDICADOR_1','-DI':'INDICADOR_2'}).set_index('Time')
        return TechnicalIndicators._finalize(out)

    @staticmethod
    def calculate_sar(df, step=0.02, max_step=0.2):
        df_new = df.copy()
        df_new['INDICADOR_1'] = np.nan
        df_new['EP'] = np.nan
        df_new['AF'] = step
        df_new['trend'] = 1
        if len(df_new) < 2:
            return TechnicalIndicators._finalize(df_new[['INDICADOR_1']])
        if df_new['Close'].iloc[1] > df_new['Close'].iloc[0]:
            df_new.loc[df_new.index[1], 'trend'] = 1
            df_new.loc[df_new.index[1], 'INDICADOR_1'] = df_new['Low'].iloc[0]
            df_new.loc[df_new.index[1], 'EP'] = df_new['High'].iloc[1]
        else:
            df_new.loc[df_new.index[1], 'trend'] = -1
            df_new.loc[df_new.index[1], 'INDICADOR_1'] = df_new['High'].iloc[0]
            df_new.loc[df_new.index[1], 'EP'] = df_new['Low'].iloc[1]
        for i in range(2, len(df_new)):
            prev = df_new.iloc[i - 1]
            curr = df_new.iloc[i]
            sar = prev['INDICADOR_1'] + prev['AF'] * (prev['EP'] - prev['INDICADOR_1']) if not np.isnan(prev['INDICADOR_1']) else prev['EP']
            if prev['trend'] == 1:
                sar = min(sar, df_new['Low'].iloc[i - 1], df_new['Low'].iloc[i - 2])
                if curr['Low'] < sar:
                    df_new.loc[df_new.index[i], 'trend'] = -1
                    df_new.loc[df_new.index[i], 'INDICADOR_1'] = prev['EP']
                    df_new.loc[df_new.index[i], 'EP'] = curr['Low']
                    df_new.loc[df_new.index[i], 'AF'] = step
                else:
                    df_new.loc[df_new.index[i], 'trend'] = 1
                    df_new.loc[df_new.index[i], 'INDICADOR_1'] = sar
                    if curr['High'] > prev['EP']:
                        df_new.loc[df_new.index[i], 'EP'] = curr['High']
                        df_new.loc[df_new.index[i], 'AF'] = min(prev['AF'] + step, max_step)
                    else:
                        df_new.loc[df_new.index[i], 'EP'] = prev['EP']
                        df_new.loc[df_new.index[i], 'AF'] = prev['AF']
            else:
                sar = max(sar, df_new['High'].iloc[i - 1], df_new['High'].iloc[i - 2])
                if curr['High'] > sar:
                    df_new.loc[df_new.index[i], 'trend'] = 1
                    df_new.loc[df_new.index[i], 'INDICADOR_1'] = prev['EP']
                    df_new.loc[df_new.index[i], 'EP'] = curr['High']
                    df_new.loc[df_new.index[i], 'AF'] = step
                else:
                    df_new.loc[df_new.index[i], 'trend'] = -1
                    df_new.loc[df_new.index[i], 'INDICADOR_1'] = sar
                    if curr['Low'] < prev['EP']:
                        df_new.loc[df_new.index[i], 'EP'] = curr['Low']
                        df_new.loc[df_new.index[i], 'AF'] = min(prev['AF'] + step, max_step)
                    else:
                        df_new.loc[df_new.index[i], 'EP'] = prev['EP']
                        df_new.loc[df_new.index[i], 'AF'] = prev['AF']
        df_new['INDICADOR_1'] = (df_new['INDICADOR_1'] - df_new['Close']) / df_new['Close']
        out = df_new[['INDICADOR_1']]
        return TechnicalIndicators._finalize(out)

    @staticmethod
    def calculate_ichimoku(df):
        df_new = df.reset_index()

        def calc_tenkan():
            high9 = df_new['High'].rolling(window=9).max()
            low9 = df_new['Low'].rolling(window=9).min()
            return (high9 + low9) / 2

        def calc_kijun():
            high26 = df_new['High'].rolling(window=26).max()
            low26 = df_new['Low'].rolling(window=26).min()
            return (high26 + low26) / 2

        def calc_senkou_b():
            high52 = df_new['High'].rolling(window=52).max()
            low52 = df_new['Low'].rolling(window=52).min()
            return ((high52 + low52) / 2).shift(26)

        with ThreadPoolExecutor() as executor:
            f1 = executor.submit(calc_tenkan)
            f2 = executor.submit(calc_kijun)
            f3 = executor.submit(calc_senkou_b)

            tenkan = f1.result()
            kijun = f2.result()
            senkou_a = ((tenkan + kijun) / 2).shift(26)
            senkou_b = f3.result()

        close = df_new['Close']
        # Relativo al precio actual para que sea estacionario
        df_new['INDICADOR_1'] = (tenkan - close) / close
        df_new['INDICADOR_2'] = (kijun - close) / close
        df_new['INDICADOR_3'] = (senkou_a - close) / close
        df_new['INDICADOR_4'] = (senkou_b - close) / close
        df_new['INDICADOR_5'] = (close - close.shift(26)) / close.shift(26)

        df_new = df_new[['Time', 'INDICADOR_1', 'INDICADOR_2', 'INDICADOR_3', 'INDICADOR_4', 'INDICADOR_5']]
        return df_new.set_index('Time')

    @staticmethod
    def calculate_bollinger_bands(df, window=20, num_std_dev=2):
        df_new = df.reset_index()

        def calc_sma():
            return df_new['Close'].rolling(window=window).mean()

        def calc_std():
            return df_new['Close'].rolling(window=window).std()

        with ThreadPoolExecutor() as executor:
            f1 = executor.submit(calc_sma)
            f2 = executor.submit(calc_std)
            sma = f1.result()
            std = f2.result()

        close = df_new['Close']
        upper = sma + (std * num_std_dev)
        lower = sma - (std * num_std_dev)
        
        df_new['INDICADOR_1'] = (sma - close) / close
        df_new['INDICADOR_2'] = (upper - close) / close
        df_new['INDICADOR_3'] = (lower - close) / close

        df_new = df_new[['Time', 'INDICADOR_1', 'INDICADOR_2', 'INDICADOR_3']]
        return df_new.set_index('Time')

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        df_new = df.reset_index()

        df_new['High'] = pd.to_numeric(df_new['High'], errors='coerce')
        df_new['Low'] = pd.to_numeric(df_new['Low'], errors='coerce')
        df_new['Close'] = pd.to_numeric(df_new['Close'], errors='coerce')

        def calc_hl():
            return df_new['High'] - df_new['Low']

        def calc_hpc():
            return abs(df_new['High'] - df_new['Close'].shift(1))

        def calc_lpc():
            return abs(df_new['Low'] - df_new['Close'].shift(1))

        with ThreadPoolExecutor() as executor:
            f1 = executor.submit(calc_hl)
            f2 = executor.submit(calc_hpc)
            f3 = executor.submit(calc_lpc)
            hl = f1.result()
            hpc = f2.result()
            lpc = f3.result()

        df_new['TR'] = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
        atr = df_new['TR'].ewm(span=period, adjust=False).mean()

        df_new['INDICADOR_1'] = atr / df_new['Close']

        return df_new[['Time', 'INDICADOR_1']].set_index('Time')

    @staticmethod
    def calculate_stochastic_oscillator(df, period=14, smooth_period=3):
        df_new = df.reset_index()

        if len(df_new) < period:
            print(f"Advertencia: Se requieren al menos {period} filas.")
            return df_new.set_index('Time')

        def calc_min():
            return df_new['Low'].rolling(window=period).min()

        def calc_max():
            return df_new['High'].rolling(window=period).max()

        with ThreadPoolExecutor() as executor:
            f1 = executor.submit(calc_min)
            f2 = executor.submit(calc_max)
            min_low = f1.result()
            max_high = f2.result()

        df_new['INDICADOR_1'] = (df_new['Close'] - min_low) / (max_high - min_low) * 100
        df_new['INDICADOR_2'] = df_new['INDICADOR_1'].rolling(window=smooth_period).mean()
        return df_new[['Time', 'INDICADOR_1', 'INDICADOR_2']].set_index('Time')

    @staticmethod
    def calculate_momentum(df, period=14):
        df_new = df.reset_index()

        def calc_momentum():
            return df_new['Close'] - df_new['Close'].shift(period)

        with ThreadPoolExecutor() as executor:
            f1 = executor.submit(calc_momentum)
            df_new['INDICADOR_1'] = f1.result()

        df_new = df_new[['Time', 'INDICADOR_1']]
        return df_new.set_index('Time')

    @staticmethod
    def calculate_cci(df, period=14):
        df_new = df.reset_index()

        df_new['TP'] = (df_new['High'] + df_new['Low'] + df_new['Close']) / 3

        sma = df_new['TP'].rolling(window=period, min_periods=period).mean()
        mad = (df_new['TP'] - sma).abs().rolling(window=period, min_periods=period).mean()
        denom = 0.015 * mad
        cci = (df_new['TP'] - sma) / denom.replace({0: np.nan})

        df_new['INDICADOR_1'] = cci
        out = df_new[['Time','INDICADOR_1']].set_index('Time')
        return TechnicalIndicators._finalize(out)

    @staticmethod
    def calculate_mfi(df, period=14):
        df_new = df.reset_index()

        df_new['Typical_Price'] = (df_new['High'] + df_new['Low'] + df_new['Close']) / 3
        df_new['Money_Flow'] = df_new['Typical_Price'] * df_new['Volume']

        pos_flow = np.where(df_new['Typical_Price'] > df_new['Typical_Price'].shift(1), df_new['Money_Flow'], 0)
        neg_flow = np.where(df_new['Typical_Price'] < df_new['Typical_Price'].shift(1), df_new['Money_Flow'], 0)

        pos_sum = pd.Series(pos_flow).rolling(window=period, min_periods=period).sum()
        neg_sum = pd.Series(neg_flow).rolling(window=period, min_periods=period).sum()
        ratio = np.where(neg_sum == 0, np.nan, pos_sum / neg_sum)
        mfi = 100 - (100 / (1 + ratio))
        mfi = np.where((neg_sum == 0) & (pos_sum > 0), 100, mfi)
        mfi = np.where((pos_sum == 0) & (neg_sum > 0), 0, mfi)

        df_new['INDICADOR_1'] = mfi
        out = df_new[['Time','INDICADOR_1']].set_index('Time')
        return TechnicalIndicators._finalize(out)

    @staticmethod
    def calculate_volume(df):
        df_new = df.reset_index()
        df_new['INDICADOR_1'] = df_new['Volume']
        df_new = df_new[['Time', 'INDICADOR_1']]
        return df_new.set_index('Time')

    @staticmethod
    def calculate_obv(df, period=20):
        df_new = df.reset_index()

        def calc_obv():
            return np.where(df_new['Close'] > df_new['Close'].shift(1),
                            df_new['Volume'],
                            np.where(df_new['Close'] < df_new['Close'].shift(1),
                                     -df_new['Volume'],
                                     0)).cumsum()

        with ThreadPoolExecutor() as executor:
            f1 = executor.submit(calc_obv)
            obv = f1.result()

        obv_series = pd.Series(obv, index=df_new.index)
        obv_change = obv_series.diff(period)
        vol_rolling = df_new['Volume'].rolling(window=period, min_periods=period).sum()
        df_new['INDICADOR_1'] = obv_change / vol_rolling.replace({0: np.nan})

        df_new = df_new[['Time', 'INDICADOR_1']]
        return TechnicalIndicators._finalize(df_new.set_index('Time'))

    @staticmethod
    def calculate_adl(df):
        df_new = df.reset_index()

        def calc_mfm():
            mfm = ((df_new['Close'] - df_new['Low']) - (df_new['High'] - df_new['Close'])) / (
                    df_new['High'] - df_new['Low'])
            return mfm.fillna(0)

        with ThreadPoolExecutor() as executor:
            f1 = executor.submit(calc_mfm)
            df_new['MFM'] = f1.result()

        df_new['MFV'] = df_new['MFM'] * df_new['Volume']
        df_new['INDICADOR_1'] = df_new['MFV'].cumsum()

        df_new = df_new[['Time', 'INDICADOR_1']]
        return df_new.set_index('Time')

    @staticmethod
    def calculate_keltner_channels(df, ema_period=20, atr_period=10, multiplier=2):
        df_new = df.reset_index()

        df_new['H-L'] = df_new['High'] - df_new['Low']
        df_new['H-PC'] = abs(df_new['High'] - df_new['Close'].shift(1))
        df_new['L-PC'] = abs(df_new['Low'] - df_new['Close'].shift(1))
        df_new['TR'] = df_new[['H-L', 'H-PC', 'L-PC']].max(axis=1)

        def calc_ema():
            return df_new['Close'].ewm(span=ema_period, adjust=False).mean()

        def calc_atr():
            return df_new['TR'].rolling(window=atr_period).mean()

        with ThreadPoolExecutor() as executor:
            f1 = executor.submit(calc_ema)
            f2 = executor.submit(calc_atr)
            ema = f1.result()
            atr = f2.result()

        close = df_new['Close']
        upper = ema + multiplier * atr
        lower = ema - multiplier * atr

        df_new['INDICADOR_2'] = (ema - close) / close
        df_new['INDICADOR_1'] = (upper - close) / close
        df_new['INDICADOR_3'] = (lower - close) / close

        df_new = df_new[['Time', 'INDICADOR_1', 'INDICADOR_2', 'INDICADOR_3']]
        return df_new.set_index('Time')

    @staticmethod
    def calculate_fibonacci_retracement(df):
        df_new = df.copy()

        max_price = df_new['High'].max()
        min_price = df_new['Low'].min()

        diff = max_price - min_price
        levels = {
            '0.0%': max_price,
            '23.6%': max_price - 0.236 * diff,
            '38.2%': max_price - 0.382 * diff,
            '50.0%': max_price - 0.500 * diff,
            '61.8%': max_price - 0.618 * diff,
            '100.0%': min_price
        }

        fib_df = pd.DataFrame(index=df_new.index)
        for level_name, level_value in levels.items():
            fib_df[level_name] = level_value

        return fib_df

    @staticmethod
    def calculate_williams_r(df, period=14):
        df_new = df.reset_index()

        def calc_high():
            return df_new['High'].rolling(window=period).max()

        def calc_low():
            return df_new['Low'].rolling(window=period).min()

        with ThreadPoolExecutor() as executor:
            f1 = executor.submit(calc_high)
            f2 = executor.submit(calc_low)
            high_max = f1.result()
            low_min = f2.result()

        df_new['INDICADOR_1'] = -100 * (high_max - df_new['Close']) / (high_max - low_min)
        df_new.replace([float('inf'), -float('inf')], float('nan'), inplace=True)

        df_new = df_new[['Time', 'INDICADOR_1']]
        return df_new.set_index('Time')

    @staticmethod
    def calculate_cmo(df, period=14):
        df_new = df.reset_index()

        df_new['delta'] = df_new['Close'].diff()
        df_new['gain'] = df_new['delta'].clip(lower=0)
        df_new['loss'] = -df_new['delta'].clip(upper=0)

        def sum_gain():
            return df_new['gain'].rolling(window=period).sum()

        def sum_loss():
            return df_new['loss'].rolling(window=period).sum()

        with ThreadPoolExecutor() as executor:
            f1 = executor.submit(sum_gain)
            f2 = executor.submit(sum_loss)
            sg = f1.result()
            sl = f2.result()

        df_new['INDICADOR_1'] = 100 * (sg - sl) / (sg + sl)
        df_new = df_new[['Time', 'INDICADOR_1']]
        return df_new.set_index('Time')
    
    @staticmethod
    def calculate_vwap(df):
        df_new = df.reset_index()
        v = df_new['Volume']
        tp = (df_new['High'] + df_new['Low'] + df_new['Close']) / 3
        vwap = (tp * v).cumsum() / v.cumsum()

        df_new['INDICADOR_1'] = (vwap - df_new['Close']) / df_new['Close']
        out = df_new[['Time','INDICADOR_1']].set_index('Time')
        return TechnicalIndicators._finalize(out)
    
    @staticmethod
    def calculate_donchian_channel(df, period=20):
        df_new = df.reset_index()
        close = df_new['Close']
        upper = df_new['High'].rolling(window=period).max()
        lower = df_new['Low'].rolling(window=period).min()
        middle = (upper + lower) / 2

        df_new['INDICADOR_1'] = (upper - close) / close
        df_new['INDICADOR_2'] = (lower - close) / close
        df_new['INDICADOR_3'] = (middle - close) / close
        out = df_new[['Time','INDICADOR_1','INDICADOR_2','INDICADOR_3']].set_index('Time')
        return TechnicalIndicators._finalize(out)
    
    @staticmethod
    def calculate_supertrend(df, period=7, multiplier=3):
        df_new = df.reset_index()
        hl = df_new['High'] - df_new['Low']
        atr = hl.ewm(span=period).mean()
        
        hl2 = (df_new['High'] + df_new['Low']) / 2
        basic_upper = hl2 + (multiplier * atr)
        basic_lower = hl2 - (multiplier * atr)

        close = df_new['Close']
        df_new['INDICADOR_1'] = (basic_upper - close) / close
        df_new['INDICADOR_2'] = (basic_lower - close) / close
        
        out = df_new[['Time', 'INDICADOR_1', 'INDICADOR_2']].set_index('Time')
        return TechnicalIndicators._finalize(out)
    
    @staticmethod
    def calculate_rolling_fibonacci(df, period=144):
        df_new = df.reset_index()

        roll_max = df_new['High'].rolling(window=period).max()
        roll_min = df_new['Low'].rolling(window=period).min()
        diff = roll_max - roll_min
        
        fib_236 = roll_max - (0.236 * diff)
        fib_382 = roll_max - (0.382 * diff)
        fib_500 = roll_max - (0.500 * diff)
        fib_618 = roll_max - (0.618 * diff)
        
        close = df_new['Close']
        df_new['FIB_236'] = (fib_236 - close) / close
        df_new['FIB_382'] = (fib_382 - close) / close
        df_new['FIB_500'] = (fib_500 - close) / close
        df_new['FIB_618'] = (fib_618 - close) / close
        
        out = df_new[['Time', 'FIB_236', 'FIB_382', 'FIB_500', 'FIB_618']].set_index('Time')
        return TechnicalIndicators._finalize(out)