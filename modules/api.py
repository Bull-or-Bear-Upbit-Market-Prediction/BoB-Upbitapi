import jwt    # PyJWT 
import uuid
import hashlib
import requests
import configparser
import json
import pandas as pd
from tqdm import tqdm

from time import sleep

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

    def to_df(self, text):
        '''
        Pandas 데이터프레임으로 변환하는 내부메소드
        '''
        data = json.loads(text)

        if len(data):
            data = pd.DataFrame({key: [row[key] for row in data] for key in data[0].keys()})
        else:
            data = None
        return data


    def get_df(self, url, query, ohlcv=False):
        '''
        요청헤더 생성하는 내부메소드
        '''
        headers = {
            'accept': 'application/json',
            'Authorization': self.generate_auth_token(query)
        }
        res = requests.get(url+query, headers=headers)
        assert res.status_code == 200, "requests fail"

        df = self.to_df(res.text)

        return self.get_ohlcv(df) if ohlcv else df


    def get_ohlcv(self, df):
        # 캔들 정보만 해당
        ohlcv = ["opening_price", "high_price", "low_price", "trade_price", "candle_acc_trade_price", "candle_acc_trade_volume"]

        return df[ohlcv]


    def market(self, keyword=None, is_details=False):
        '''
        업비트에서 거래되는 자산 정보

        https://docs.upbit.com/reference/%EB%A7%88%EC%BC%93-%EC%BD%94%EB%93%9C-%EC%A1%B0%ED%9A%8C
        '''
        query = f'isDetails={is_details}'
        url = 'https://api.upbit.com/v1/market/all?'

        df = self.get_df(url, query)
        if keyword is not None:
            return df[df["market"].apply(lambda x: keyword in x)]
        else:
            return df


    def min_candle(self, unit=1, market="KRW-BTC", to="", count=10, ohlcv=True):
        '''
        분봉 정보

        https://docs.upbit.com/reference/%EB%B6%84minute-%EC%BA%94%EB%93%A4-1
        '''
        url = f'https://api.upbit.com/v1/candles/minutes/{unit}?'
        query = f'market={market}&to={to}&count={count}'

        return self.get_df(url, query, ohlcv)
    
    def day_candle(self, market="KRW-BTC", to="", count=10, converting_price_unit="KRW", ohlcv=True):
        '''
        일봉 정보

        https://docs.upbit.com/reference/%EC%9D%BCday-%EC%BA%94%EB%93%A4-1
        '''
        url = 'https://api.upbit.com/v1/candles/days?'
        query = f'market={market}&to={to}&count={count}&convertingPriceUnit={converting_price_unit}'

        return self.get_df(url, query, ohlcv)


    def week_candle(self, market="KRW-BTC", to="", count=10, ohlcv=True):
        '''
        주봉 정보

        https://docs.upbit.com/reference/%EC%A3%BCweek-%EC%BA%94%EB%93%A4-1
        '''
        url = 'https://api.upbit.com/v1/candles/weeks?'
        query = f'market={market}&to={to}&count={count}'

        return self.get_df(url, query, ohlcv)


    def month_candle(self, market="KRW-BTC", to="", count=10, ohlcv=True):
        '''
        월봉 정보

        https://docs.upbit.com/reference/%EC%9B%94month-%EC%BA%94%EB%93%A4-1
        '''
        url = 'https://api.upbit.com/v1/candles/months?'
        query = f'market={market}&to={to}&count={count}'

        return self.get_df(url, query, ohlcv)


    def trade_ticks(self, market="KRW-BTC", to='', count=1, cursor='', days_ago=''):
        '''
        최근 체결 내역

        https://docs.upbit.com/reference/%EC%B5%9C%EA%B7%BC-%EC%B2%B4%EA%B2%B0-%EB%82%B4%EC%97%AD
        '''
        url = 'https://api.upbit.com/v1/trades/ticks'
        query = f'market={market}&to={to}&count={count}&cursor={cursor}&daysAgo={days_ago}'

        return self.get_df(url, query)


    def ticker(self, markets=["KRW-BTC"], ohlcv=True):
        '''
        티커 조회
        
        https://docs.upbit.com/reference/ticker%ED%98%84%EC%9E%AC%EA%B0%80-%EC%A0%95%EB%B3%B4
        '''
        url = 'https://api.upbit.com/v1/ticker?'
        query = f'markets={",".join(markets)}'
        df = self.get_df(url, query, ohlcv=False)
        cols = ["market", "opening_price", "high_price", "low_price", "trade_price", "prev_closing_price", "acc_trade_price_24h", "acc_trade_volume_24h"]

        return df[cols] if ohlcv else df


    def orderbook(self, markets=["KRW-BTC"]):
        '''
        호가 정보 조회

        https://docs.upbit.com/reference/%ED%98%B8%EA%B0%80-%EC%A0%95%EB%B3%B4-%EC%A1%B0%ED%9A%8C
        '''
        url = 'https://api.upbit.com/v1/orderbook?'
        query = f'markets={",".join(markets)}'
        return self.get_df(url, query)
        

    def top_markets_by_trade_price(self, keyword="", k=-1, ohlcv=True):
        '''
        하루 기준 거래 대금이 가장 많은 시장 순으로 k개 출력하는 메소드, keyword로 마켓 필터
        '''
        markets = self.market(keyword)["market"].to_list()
        df = self.ticker(markets=markets).sort_values(by=["acc_trade_price_24h"], ascending=False)
        if k != -1:
            df = df.head(k)
        return df