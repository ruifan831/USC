import pandas as pd 
import os
from sklearn import preprocessing
from collections import deque
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,LSTM,CuDNNLSTM,BatchNormalization
from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint
import time


SEQ_LEN=60
FUTURE_PERIOD_PREDICT=3
RATIO_TO_PREDICT='LTC-USD'
EPOCHS=10
BATCH_SIZE=64

NAME=f'{RATIO_TO_PREDICT}-{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}'
data_files=os.listdir('crypto_data')
columns_name=['time','low','high','open','close','volume']
print(data_files)
df=pd.read_csv('crypto_data/LTC-USD.csv',names=columns_name)


def classify(current,future):
    if float(future)> float(current):
        return 1
    else:
        return 0


# def preprocess_df(df):
#     df = df.drop("future", 1)  # don't need this anymore.

#     for col in df.columns:  # go through all of the columns
#         if col != "target":  # normalize all ... except for the target itself!
#             df[col] = df[col].pct_change()  # pct change "normalizes" the different currencies (each crypto coin has vastly diff values, we're really more interested in the other coin's movements)
#             df.dropna(inplace=True)  # remove the nas created by pct_change
#             df[col] = preprocessing.scale(df[col].values)  # scale between 0 and 1.

#     df.dropna(inplace=True)  # cleanup again... jic.


#     sequential_data = []  # this is a list that will CONTAIN the sequences
#     prev_days = deque(maxlen=SEQ_LEN)  # These will be our actual sequences. They are made with deque, which keeps the maximum length by popping out older values as new ones come in

#     for i in df.values:  # iterate over the values
#         prev_days.append([n for n in i[:-1]])  # store all but the target
#         if len(prev_days) == SEQ_LEN:  # make sure we have 60 sequences!
#             sequential_data.append([np.array(prev_days), i[-1]])  # append those bad boys!

#     random.shuffle(sequential_data)  # shuffle for good measure.

#     buys = []  # list that will store our buy sequences and targets
#     sells = []  # list that will store our sell sequences and targets

#     for seq, target in sequential_data:  # iterate over the sequential data
#         if target == 0:  # if it's a "not buy"
#             sells.append([seq, target])  # append to sells list
#         elif target == 1:  # otherwise if the target is a 1...
#             buys.append([seq, target])  # it's a buy!

#     random.shuffle(buys)  # shuffle the buys
#     random.shuffle(sells)  # shuffle the sells!

#     lower = min(len(buys), len(sells))  # what's the shorter length?

#     buys = buys[:lower]  # make sure both lists are only up to the shortest length.
#     sells = sells[:lower]  # make sure both lists are only up to the shortest length.

#     sequential_data = buys+sells  # add them together
#     random.shuffle(sequential_data)  # another shuffle, so the model doesn't get confused with all 1 class then the other.

#     X = []
#     y = []

#     for seq, target in sequential_data:  # going over our new sequential data
#         X.append(seq)  # X is the sequences
#         y.append(target)  # y is the targets/labels (buys vs sell/notbuy)

#     return np.array(X), y  # return X and y...and make X a numpy array!

def preprocess_df(df):
    df = df.drop('future',1)
    for col in df.columns:
        if col != 'target':
            df[col]=df[col].pct_change()
            df.dropna(inplace=True)
            df[col]= preprocessing.scale(df[col].values)
    df.dropna(inplace=True)
    
    sequential_data=[]
    prev_days=deque(maxlen=SEQ_LEN)
    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days)==SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])
    random.shuffle(sequential_data)

    buys=[]
    sells=[]

    for seq,target in sequential_data:
        if target==0:
            sells.append([seq,target])
        elif target==1:
            buys.append([seq,target])
    lower = min(len(buys),len(sells))

    buys= buys[:lower]
    sells=sells[:lower]

    sequential_data=buys + sells
    random.shuffle(sequential_data)
    X=[]
    Y=[]
    for seq,target in sequential_data:
        X.append(seq)
        Y.append(target)
    return np.array(X),np.array(Y)
    




main_df=pd.DataFrame()
for i in data_files:
    dataset=f'crypto_data/{i}'
    coin=i.split('.')[0]
    df=pd.read_csv(dataset,names=columns_name)
    df.rename(columns={'close':f'{coin}_close','volume':f'{coin}_volume'},inplace=True)

    df.set_index('time',inplace=True)
    df=df[[f'{coin}_close',f'{coin}_volume']]
    if len(main_df)==0:
        main_df=df
    else:
        main_df=main_df.join(df)

main_df['future']=main_df[f"{RATIO_TO_PREDICT}_close"].shift(-FUTURE_PERIOD_PREDICT)


main_df['target']=list(map(classify,main_df[f"{RATIO_TO_PREDICT}_close"],main_df['future']))

times = sorted(main_df.index.values)
last_5pct= times[-int(0.05*len(times))]

validation_main_df=main_df[(main_df.index>= last_5pct)]
main_df=main_df[(main_df<last_5pct)]

train_x, train_y = preprocess_df(main_df)
val_x, val_y = preprocess_df(validation_main_df)

model=Sequential()
model.add(CuDNNLSTM(128,input_shape=(train_x.shape[1:]),return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(CuDNNLSTM(128,input_shape=(train_x.shape[1:]),return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(CuDNNLSTM(128,input_shape=(train_x.shape[1:])))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2,activation='softmax'))

opt=tf.keras.optimizers.Adam(lr=0.001,decay=1e-6)

model.compile(loss='sparse_categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

tensorboard=TensorBoard(log_dir=f'logs/{NAME}')

filepath='RNN_Final-{epoch:02d}'

checkpoint = ModelCheckpoint('models/{}.model'.format(filepath,monitor='val_acc',verbose=1,save_best_only=True,mode='max'))

hist = model.fit(train_x,train_y,batch_size= BATCH_SIZE, epochs=EPOCHS, validation_data=(val_x,val_y),callbacks=[tensorboard,checkpoint])