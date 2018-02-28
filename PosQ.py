from keras.models import Sequential
from keras.layers import Dropout, Dense, LSTM
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras import initializers
from keras import backend as K
import numpy as np
from keras.optimizers import Adam

def getCol(i, tensor):
    col = tensor[:,i]
    col = K.reshape(col, (-1,1))
    return col

def myMD(y_true, y_pred):
    e = y_true - y_pred
    e0 = getCol(3,e)**2
    e1 = getCol(4,e)**2
    e2 = getCol(5,e)**2
    q0 = getCol(0,y_pred)**2
    q1 = getCol(1,y_pred)**2
    q2 = getCol(2,y_pred)**2
    md0 = e0/q0
    md1 = e1/q1
    md2 = e2/q2
    normQ = K.sqrt(q0**2+ q1**2+ q2**2)
    logQ = K.log(normQ + 0.9)
    #e = K.concatenate([logQ, md0, md1, md2, e0, e1, e2], axis=1)
    #e = K.concatenate([logQ, md0, md1, md2], axis=1)
    e = logQ + md0 + md1 + md2
    e = K.mean(e)
    return e

def myLoss():
    def callBack(y_true, y_pred):
        return myMD(y_true, y_pred)
    return callBack

def myMSE(y_true, y_pred):
    e = y_true - y_pred
    e0 = getCol(3,e)**2
    e1 = getCol(4,e)**2
    e2 = getCol(5,e)**2
    e = K.concatenate([e0, e1, e2], axis=1)
    e = K.mean(e)
    return e

def myLoss2():
    def callBack(y_true, y_pred):
        return myMSE(y_true, y_pred)
    return callBack

def getPosQ(input_dim, T):
    myinit = 'uniform'#initializers.RandomNormal(stddev=0.1)
    model = Sequential()
    model.add(LSTM(T, activation=LeakyReLU(), return_sequences = True, input_shape=input_dim))
    model.add(LSTM(T, activation=LeakyReLU(), return_sequences = False))
    #model.add(Dense(10, kernel_initializer="uniform", activation=LeakyReLU()))
    model.add(Dense(10, kernel_initializer="uniform", activation=LeakyReLU()))
    model.add(Dense(6, kernel_initializer="uniform", activation='linear'))
    model.compile(loss=myLoss(), optimizer='adam', metrics=[myLoss2()])
    return model
