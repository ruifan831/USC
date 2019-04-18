#!/usr/bin/env python
# coding: utf-8

# In[221]:


import os
import re


# In[222]:


# only call once
def corpus():
    appended_content=""
    for i in os.listdir():
        if ".txt" in i:
            print(i)
            file=open(i,"r",encoding="utf-8-sig")
            data=file.read().replace("\n","")
            appended_content=appended_content+data
            file.close()
    return appended_content

        


# In[223]:


corpus_result=corpus()


# In[224]:


out=[ord(i) for i in corpus_result]


# In[225]:


import numpy as np
out_scaled=list(map(lambda x: x/255,list(filter(lambda x: x<255,out))))
out_unscaled=list(filter(lambda x: x<255,out))
W=100
S=1


# In[254]:


X=[]
Y=[]
for i in range(len(out)):
    try:
        X.append(out_scaled[i:i+W-S])
        Y.append(out_unscaled[i+W-S])
    except IndexError:
        break
X=X[:-1]


# In[255]:


len(X)


# In[256]:


np.shape(X)


# In[257]:


X=np.reshape(X,(-1,99,1))


# In[258]:


np.shape(X)


# In[259]:


A=np.diag(np.ones(256))


# In[260]:


Y=[A[i] for i in Y]


# In[261]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
from sklearn.model_selection import train_test_split


# In[262]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


# In[263]:


np.shape(X_train)


# In[268]:


model= Sequential()

model.add(LSTM(128,input_shape=(99,1),activation="sigmoid",return_sequences=True))
model.add(Dense(256,activation="softmax"))
opt=tf.keras.optimizers.Adam(lr=1e-3,decay=1e-5)
model.compile(loss="sparse_categorical_crossentropy",optimizer=opt,
             metrics=['accuracy'])
model.fit(X_train,Y_train,epochs=3,batch_size=50,validation_data=(X_test,Y_test))


# In[ ]:




