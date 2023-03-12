from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.models import load_model
from tech import applytech
import numpy as np
import pandas as pd
import os.path
import pickle
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from modules.api import UpbitAPI

def BoB_demo(coin,batchsize=50,api=UpbitAPI()):
    if os.path.isfile('./temp/'+coin+'.h5') and os.path.isfile('./temp/'+coin+'_scaler.pkl'):
        model=load_model('./temp/'+coin+'.h5')
        with open('./temp/'+coin+'_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        df=applytech(api.min_candle(unit=3,market=coin,count=1000))
        df.sort_index(inplace=True)
        data = df.filter(['close','open','high','low','Daily return'])
        data['Daily return']=data['Daily return'].shift(1,fill_value=0)
        dataset = data.values
        training_data_len = int(np.ceil( len(dataset) * .9 ))


        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(dataset)

    else:
        df=applytech(api.min_candle(unit=3,market=coin,count=1000))
        df.sort_index(inplace=True)
        data = df.filter(['close','open','high','low','Daily return'])
        data['Daily return']=data['Daily return'].shift(1,fill_value=0)
        dataset = data.values
        training_data_len = int(np.ceil( len(dataset) * .9 ))

        breakpoint()
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(dataset)

        train_data = scaled_data[0:int(training_data_len), :]
        
        x_train = []
        y_train = []
        breakpoint()
        for i in range(batchsize, len(train_data)):
            x_train.append(train_data[i-batchsize:i, :])
            y_train.append(train_data[i, :])

            

        x_train, y_train = np.array(x_train), np.array(y_train)
        breakpoint()
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], x_train.shape[2])))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(x_train.shape[2]))

        model.compile(optimizer='adam', loss='mean_squared_error')


        model.fit(x_train, y_train,batch_size=10,epochs=2)
        
        model.save('./temp/'+coin+'.h5')

        with open('./temp/'+coin+'_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

    breakpoint()
    test_data = scaled_data[training_data_len - batchsize: , :]

    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(batchsize, len(test_data)):
        x_test.append(test_data[i-batchsize:i, :])
        
    breakpoint()
    x_test = np.array(x_test)
    
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    breakpoint()
    rmse = np.sqrt(np.mean(((predictions[:-1,:] - y_test[:-1,:]) ** 2)))

    df_pred=pd.DataFrame(index=df.index[training_data_len:],columns=['close','open','high','low','Daily return'],data=predictions)
    breakpoint()
    acc=pd.DataFrame(pd.concat([df['Daily return'],df_pred['Daily return']],axis=1).values,columns=['y','ypred'])
    acc=acc.dropna(axis=0)
    acc['correct']=acc.apply(lambda x: 1 if ((x.y*x.ypred)>0) else 0,axis=1)

    
    score=(acc['correct'].sum()/len(acc))

    return score