import pandas as pd
import numpy as np
import ta.momentum
import ta.volatility
import ta.volume
import ta.trend

def applytech(df:pd.DataFrame):
    df.sort_index(inplace=True)
    df['bol_upper']=ta.volatility.bollinger_hband(df['trade_price'])
    df['bol_lower']=ta.volatility.bollinger_lband(df['trade_price'])
    df['rsi']=ta.momentum.rsi(df['trade_price'])
    df['stochastic_oscillators']=ta.momentum.StochasticOscillator(df['high_price'],df['low_price'],df['trade_price']).stoch_signal()
    df['12ema']=ta.trend.ema_indicator(df['trade_price'],window=12)
    df['26ema']=ta.trend.ema_indicator(df['trade_price'],window=26)
    df['macd']=ta.trend.macd(df['trade_price'])
    df['macd_signal']=ta.trend.macd_signal(df['trade_price'])
    df['macd_oscillators']=ta.trend.macd_diff(df['trade_price'])
    df['Daily return']=df['trade_price'].pct_change()
    df.rename(columns = {'opening_price':'open','trade_price':'close','high_price':'high','low_price':'low','candle_acc_trade_volume':'volume'},inplace=True)
    return df