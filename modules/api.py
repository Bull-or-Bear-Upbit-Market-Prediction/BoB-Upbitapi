import jwt    # PyJWT 
import uuid
import hashlib
import requests
import configparser
import json
import pandas as pd
from ratelimit import limits, sleep_and_retry

class UpbitAPI:
    def __init__(self, config_file='config.ini'):
        config = configparser.ConfigParser()
        config.read(config_file)
        keys = config["DEFAULT"]

        self.ak = keys["ACCESS_KEY"]
        self.sk = keys["SECRET_KEY"]

    def generate_auth_token(self, query):
        '''
        인증토큰 생성하는 내부메소드
        '''
        m = hashlib.sha512()
        m.update(query.encode())
        query_hash = m.hexdigest()

        payload = {
            'access_key': self.ak,
            'nonce': str(uuid.uuid4()),
            'query_hash': query_hash,
            'query_hash_alg': 'SHA512',
        }
            
        jwt_token = jwt.encode(payload, self.sk)
        authorization_token = 'Bearer {}'.format(jwt_token)

        return authorization_token


    @sleep_and_retry
    @limits(calls=5, period=1)
    def get_df(self, url, query, case="candle"):
        '''
        요청헤더 생성하고 실제 api 요청하여 결과를 pandas DataFrame으로 변환하는 내부메소드
        '''
        headers = {
            'accept': 'application/json',
            'Authorization': self.generate_auth_token(query)
        }
        res = requests.get(url+query, headers=headers)
        assert res.status_code == 200, "requests fail"

        data = json.loads(res.text)

        if len(data):
            data = pd.DataFrame({key: [row[key] for row in data] for key in data[0].keys()})
            data = self.add_datetime64_cols(data, case)
        else:
            data = None

        return data


    def get_ohlcv_candle(self, df):
        '''
        캔들에 대한 ohlcv 만 뽑아오는 함수
        티커의 경우 get_ticker에서 내부 구현
        '''
        ohlcv = ["opening_price", "high_price", "low_price", "trade_price", "candle_acc_trade_price", "candle_acc_trade_volume"]

        return df.set_index("datetime64_kst")[ohlcv]


    def add_datetime64_cols(self, df, case):
        '''
        데이터에 datetime64 타입의 칼럼 추가
        '''
        if case == "candle":
            df["datetime64_kst"] = pd.to_datetime(df["candle_date_time_kst"].apply(lambda x: x.replace('T', ' ')))
            df["datetime64_utc"] = pd.to_datetime(df["candle_date_time_utc"].apply(lambda x: x.replace('T', ' ')))
        elif case == "ticker":
            df["datetime64_kst"] = df.apply(lambda x: x["trade_date_kst"]+x["trade_time_kst"], axis=1)
            df["datetime64_utc"] = df.apply(lambda x: x["trade_date"]+x["trade_time"], axis=1)
            df["datetime64_kst"] = pd.to_datetime(df["datetime64_kst"])
            df["datetime64_utc"] = pd.to_datetime(df["datetime64_utc"])
        elif case == "trade":
            df["datetime64_utc"] = df.apply(lambda x: x["trade_date_utc"]+x["trade_time_utc"], axis=1)
            df["datetime64_utc"] = pd.to_datetime(df["datetime64_utc"])

        return df


    def market(self, keyword=None, is_details=False):
        '''
        업비트에서 거래되는 자산 정보

        https://docs.upbit.com/reference/%EB%A7%88%EC%BC%93-%EC%BD%94%EB%93%9C-%EC%A1%B0%ED%9A%8C
        '''
        query = f'isDetails={is_details}'
        url = 'https://api.upbit.com/v1/market/all?'

        df = self.get_df(url, query, None)
        if keyword is not None:
            return df[df["market"].apply(lambda x: keyword in x)]
        else:
            return df
    

    def min_candle(self, unit=1, market="KRW-BTC", to="", count=10, ohlcv=True, index=None):
        '''
        분봉 정보

        https://docs.upbit.com/reference/%EB%B6%84minute-%EC%BA%94%EB%93%A4-1
        '''
        # assert unit in [1, 3, 5, 15, 10, 30, 60, 240], "invalid unit"
        url = f'https://api.upbit.com/v1/candles/minutes/{unit}?'
        cur_count = min(count, 200)
        query = f'market={market}&to={to}&count={cur_count}'
        df = self.get_df(url, query)
        assert df is not None, "candle api error, count might be too large"

        if count > 200:
            last_to = df.loc[len(df)-1]["candle_date_time_utc"]
            newdf = self.min_candle(unit=unit, market=market, to=last_to, count=count-199, ohlcv=False).iloc[1:]
            df = pd.concat([df, newdf], ignore_index=True)

        return self.get_ohlcv_candle(df) if ohlcv else df
    

    def day_candle(self, market="KRW-BTC", to="", count=10, converting_price_unit="KRW", ohlcv=True):
        '''
        일봉 정보

        https://docs.upbit.com/reference/%EC%9D%BCday-%EC%BA%94%EB%93%A4-1
        '''
        url = 'https://api.upbit.com/v1/candles/days?'
        cur_count = min(count, 200)
        query = f'market={market}&to={to}&count={cur_count}&convertingPriceUnit={converting_price_unit}'
        df = self.get_df(url, query)
        assert df is not None, "candle api error, count might be too large"

        if count > 200:
            last_to = df.loc[len(df)-1]["candle_date_time_utc"]
            newdf = self.day_candle(market=market, to=last_to, count=count-199, converting_price_unit=converting_price_unit, ohlcv=False).loc[1:]
            df = pd.concat([df, newdf], ignore_index=True)

        return self.get_ohlcv_candle(df) if ohlcv else df


    def week_candle(self, market="KRW-BTC", to="", count=10, ohlcv=True):
        '''
        주봉 정보

        https://docs.upbit.com/reference/%EC%A3%BCweek-%EC%BA%94%EB%93%A4-1
        '''
        url = 'https://api.upbit.com/v1/candles/weeks?'
        cur_count = min(count, 200)
        query = f'market={market}&to={to}&count={cur_count}'
        df = self.get_df(url, query)
        assert df is not None, "candle api error, count might be too large"

        if count > 200:
            last_to = df.loc[len(df)-1]["candle_date_time_utc"]
            newdf = self.week_candle(market=market, to=last_to, count=count-199, ohlcv=False).loc[1:]
            df = pd.concat([df, newdf], ignore_index=True)

        return self.get_ohlcv_candle(df) if ohlcv else df


    def month_candle(self, market="KRW-BTC", to="", count=10, ohlcv=True):
        '''
        월봉 정보

        https://docs.upbit.com/reference/%EC%9B%94month-%EC%BA%94%EB%93%A4-1
        '''
        url = 'https://api.upbit.com/v1/candles/months?'
        cur_count = min(count, 200)
        query = f'market={market}&to={to}&count={cur_count}'
        df = self.get_df(url, query)
        assert df is not None, "candle api error, count might be too large"

        if count > 200:
            last_to = df.loc[len(df)-1]["candle_date_time_utc"]
            newdf = self.month_candle(market=market, to=last_to, count=count-199, ohlcv=False).loc[1:]
            df = pd.concat([df, newdf], ignore_index=True)

        return self.get_ohlcv_candle(df) if ohlcv else df


    def trade_ticks(self, market="KRW-BTC", to='', count=1, cursor='', days_ago=''):
        '''
        최근 체결 내역

        https://docs.upbit.com/reference/%EC%B5%9C%EA%B7%BC-%EC%B2%B4%EA%B2%B0-%EB%82%B4%EC%97%AD
        '''
        url = 'https://api.upbit.com/v1/trades/ticks'
        query = f'market={market}&to={to}&count={count}&cursor={cursor}&daysAgo={days_ago}'

        return self.get_df(url, query, case="trade")


    def ticker(self, markets=["KRW-BTC"], ohlcv=True):
        '''
        티커 조회
        
        https://docs.upbit.com/reference/ticker%ED%98%84%EC%9E%AC%EA%B0%80-%EC%A0%95%EB%B3%B4
        '''
        url = 'https://api.upbit.com/v1/ticker?'
        query = f'markets={",".join(markets)}'
        df = self.get_df(url, query, case="ticker")
        cols = ["market", "opening_price", "high_price", "low_price", "trade_price", "prev_closing_price", "acc_trade_price_24h", "acc_trade_volume_24h"]

        return df.set_index("datetime64_kst")[cols] if ohlcv else df


    def orderbook(self, markets=["KRW-BTC"]):
        '''
        호가 정보 조회

        https://docs.upbit.com/reference/%ED%98%B8%EA%B0%80-%EC%A0%95%EB%B3%B4-%EC%A1%B0%ED%9A%8C
        '''
        url = 'https://api.upbit.com/v1/orderbook?'
        query = f'markets={",".join(markets)}'
        return self.get_df(url, query, case=None)
        

    def top_markets_by_trade_price(self, keyword="", k=-1, ohlcv=True):
        '''
        하루 기준 거래 대금이 가장 많은 시장 순으로 k개 출력하는 메소드, keyword로 마켓 필터
        '''
        markets = self.market(keyword)["market"].to_list()
        df = self.ticker(markets=markets, ohlcv=ohlcv).sort_values(by=["acc_trade_price_24h"], ascending=False)
        if k != -1:
            df = df.head(k)
        return df