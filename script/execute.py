from lstm2 import Bob_demo
from tech import applytech

import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from modules.api import UpbitAPI

api=UpbitAPI()
coins=api.top_markets_by_trade_price(keyword='KRW').iloc[2:13]['market'].values
board=dict()
for i in coins:
    board[i]=Bob_demo(i)
print(board)