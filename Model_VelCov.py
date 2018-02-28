from keras.models import Model, Sequential
from keras.layers import Input, Dropout, Dense, Conv2D, Flatten, Merge, Activation, Conv2DTranspose, concatenate, Flatten
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
import keras.backend as K
from keras.optimizers import RMSprop, Adam

def getCol(i, tensor):
    col = tensor[:,i]
    col = K.reshape(col, (-1,1))
    return col

def myMD(y_true, y_pred):
    e = y_true
    e0 = getCol(0,e)**2
    e1 = getCol(1,e)**2
    e2 = getCol(2,e)**2

    q0 = K.abs(getCol(0,y_pred))
    q1 = K.abs(getCol(1,y_pred))
    q2 = K.abs(getCol(2,y_pred))

    md0 = e0/q0
    md1 = e1/q1
    md2 = e2/q2

    normQ = K.sqrt(q0**2+ q1**2+ q2**2)
    logQ = K.log(normQ)
    e_final = logQ + md0 + md1 + md2
    e_final = K.mean(e_final)
    return e_final

def customMD():
    def md(y_true, y_pred):
        return myMD(y_true, y_pred)
    return md

def getVelQ():
    model = Sequential()
    model.add(Dense(20, input_shape=(3,), activation=LeakyReLU()))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(3, activation='linear'))
    rms = RMSprop(lr=10**-3, rho=0.9, epsilon=10**-6, decay=0.0)
    model.compile(loss=customMD(), optimizer=rms)
    return model

    #end
