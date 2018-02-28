from keras.models import Model, Sequential
from keras.layers import Input, Dropout, Dense, Conv2D, Flatten, Merge, Activation, Conv2DTranspose, concatenate, Flatten, LSTM
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
import keras.backend as K
from keras.optimizers import RMSprop, Adam

def getCol(i, tensor):
    col = tensor[:,i]
    col = K.reshape(col, (-1,1))
    return col

def myMD(y_true, y_pred):
    e = y_true - y_pred
    e0 = getCol(3,e)**2
    e1 = getCol(4,e)**2
    e2 = getCol(5,e)**2

    errForMD0 = (getCol(0,y_true) - getCol(3,y_true))**2
    errForMD1 = (getCol(1,y_true) - getCol(4,y_true))**2
    errForMD2 = (getCol(2,y_true) - getCol(5,y_true))**2
    q0 = getCol(0,y_pred)**2
    q1 = getCol(1,y_pred)**2
    q2 = getCol(2,y_pred)**2
    md0 = errForMD0/q0
    md1 = errForMD1/q1
    md2 = errForMD2/q2
    normQ = K.sqrt(q0**2 + q1**2+ q2**2)

    logQ = K.log(normQ + 0.8)
    e = logQ + md0 + md1 + md2  + e0 + e1 + e2
    e = K.mean(e)
    return e

def customMD():
    def md(y_true, y_pred):
        return myMD(y_true, y_pred)
    return md


def myMSE(y_true, y_pred):
    e = y_true - y_pred
    e0 = getCol(3,e)**2
    e1 = getCol(4,e)**2
    e2 = getCol(5,e)**2
    e = K.concatenate([e0, e1, e2], axis=1)
    e = K.mean(e)
    return e

def myLoss2():
    def mse(y_true, y_pred):
        return myMSE(y_true, y_pred)
    return mse

def getPosQ(input_dim):
    model = Sequential()
    model.add(Dense(10, input_dim=input_dim, kernel_initializer="glorot_uniform", activation=LeakyReLU()))
    model.add(Dense(10, kernel_initializer="glorot_uniform", activation=LeakyReLU()))
    model.add(Dense(6, kernel_initializer="glorot_uniform", activation='linear'))
    adam = Adam(lr=10**-4, beta_1=0.9, beta_2=0.999)
    model.compile(loss=customMD(), optimizer=adam)
    return model








    #end
