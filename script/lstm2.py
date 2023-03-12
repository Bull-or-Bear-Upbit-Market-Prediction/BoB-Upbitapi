from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.models import load_model
from keras.layers import Dropout, Activation
from tech import applytech
import numpy as np
import pandas as pd
import os.path
import pickle
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from modules.api import UpbitAPI

def Bob_demo(coin="KRW-DOGE",batchsize=50,api=UpbitAPI()):
    df=api.min_candle(unit=240,market="KRW-DOGE",count=2000)
    df.sort_index(inplace=True)
    df=applytech(df)


    data = df.filter(['Daily return','rsi','volume','stochastic_oscillators'])
    data['Daily return']=data['Daily return'].shift(1,fill_value=0)
    data.dropna(inplace=True)



    df_y=data.apply(lambda x: 1 if x['Daily return']>=0 else 0,axis=1)

    dataset = data.values
    dataset_y=df_y.values

    training_data_len = int(np.ceil( len(dataset) * .9 ))

    if  os.path.isfile('./temp/'+coin+'_scaler.pkl'):
        with open('./temp/'+coin+'_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    else:
        scaler = MinMaxScaler(feature_range=(0,1))
        with open('./temp/'+coin+'_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)


    scaled_data = scaler.fit_transform(dataset)

    if os.path.isfile('./temp/'+coin+'.h5') and os.path.isfile('./temp/'+coin+'_scaler.pkl'):
        model=load_model('./temp/'+coin+'.h5')
    else:
        train_data = scaled_data[0:int(training_data_len), :]
        train_data_y=dataset_y[0:int(training_data_len)]

        x_train = []
        y_train = []

        for i in range(batchsize, len(train_data)):
            x_train.append(train_data[i-batchsize:i, :])
            y_train.append(train_data_y[i])


            

        x_train, y_train = np.array(x_train), np.array(y_train)
        y_train=np.reshape(y_train,(y_train.shape[0],1))
        


        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], x_train.shape[2]),activation='ReLU'))
        model.add(Dropout(0.2))
        model.add(LSTM(64, return_sequences=False,activation='ReLU'))
        model.add(Dense(16))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(x_train, y_train,batch_size=x_train.shape[0]//batchsize,epochs=5)

        model.save('./temp/'+coin+'.h5')


    test_data = scaled_data[training_data_len - batchsize: , :]
    # Create the data sets x_test and y_test
    x_test = []
    y_test = dataset_y[training_data_len:]
    for i in range(batchsize, len(test_data)):
        x_test.append(test_data[i-batchsize:i, :])
        
    # Convert the data to a numpy array
    x_test = np.array(x_test)


    # Reshape the data
    #x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2] ))
    #print(x_test.shape)
    # Get the models predicted price values 
    predictions = model.predict(x_test)

    #predictions = scaler.inverse_transform(predictions)

    # Get the root mean squared error (RMSE)
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))


    df_pred=pd.DataFrame(index=df.index[-len(predictions):],columns=['up_down'],data=predictions)
    df=pd.DataFrame(index=df.index[-len(y_test):],columns=['up_down'],data=y_test)


    acc=pd.DataFrame(pd.concat([df['up_down'],df_pred['up_down']],axis=1).values,columns=['y','ypred'])
    acc=acc.dropna(axis=0)
    acc['correct']=acc.apply(lambda x: 1 if (((x['y']-0.5)*(x['ypred']-0.5))>0) else 0,axis=1)

    score=(acc['correct'].sum()/len(acc))

    return score

