from keras.models import Model
from keras.layers import Input, Dropout, Dense, Conv2D, Flatten, Merge, Activation, Conv2DTranspose, concatenate, Flatten
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
import keras.backend as K
from keras.optimizers import RMSprop, Adam

def getCol(i, tensor):
    col = tensor[:,i]
    col = K.reshape(col, (-1,1))
    return col

# def myMD(y_true, y_pred):
#     e = y_true - y_pred
#     e0 = getCol(3,e)**2
#     e1 = getCol(4,e)**2
#     e2 = getCol(5,e)**2
#     q0 = getCol(0,y_pred)**2
#     q1 = getCol(1,y_pred)**2
#     q2 = getCol(2,y_pred)**2
#     md0 = e0/q0
#     md1 = e1/q1
#     md2 = e2/q2
#     normQ = K.sqrt(q0**2+ q1**2+ q2**2)
#     logQ = K.log(normQ + 0.5)
#     #e = K.concatenate([logQ, md0, md1, md2, e0, e1, e2], axis=1)
#     #e = K.concatenate([logQ, md0, md1, md2], axis=1)
#     e = logQ + md0 + md1 + md2
#     e = K.mean(e)
#     return e
#
# def myLoss():
#     def callBack(y_true, y_pred):
#         return myMD(y_true, y_pred)
#     return callBack

def myMSE(y_true, y_pred):
    e = y_true - y_pred
    e0 = getCol(3,e)**2
    e1 = getCol(4,e)**2
    e2 = getCol(5,e)**2
    e = K.concatenate([e0, e1, e2], axis=1)
    e = K.mean(e)
    return e

def customMSE():
    def mse(y_true, y_pred):
        return myMSE(y_true, y_pred)
    return mse

def myMAE(y_true, y_pred):
    print y_pred.shape
    e = y_true - y_pred
    e0 = K.abs(getCol(3,e))
    e1 = K.abs(getCol(4,e))
    e2 = K.abs(getCol(5,e))
    e = K.concatenate([e0, e1, e2], axis=1)
    e = K.mean(e)
    return e

def customMAE():
    def mae(y_true, y_pred):
        return myMAE(y_true, y_pred)
    return mae

def getCNN(h,w):
    input0 = Input(shape=(h, w, 3))
    input1 = Input(shape=(h, w, 3))
    input = concatenate([input0, input1], axis=3)
    conv0 =   Conv2D(64,   (3, 3), name = 'conv0',   strides = 1, padding='same', activation=LeakyReLU())(input)
    conv1 =   Conv2D(64,   (3, 3), name = 'conv1',   strides = 2, padding='same', activation=LeakyReLU())(conv0)
    conv1_1 = Conv2D(128,  (3, 3), name = 'conv1_1', strides = 1, padding='same', activation=LeakyReLU())(conv1)
    conv2 =   Conv2D(128,  (3, 3), name = 'conv2',   strides = 2, padding='same', activation=LeakyReLU())(conv1_1)
    conv2_1 = Conv2D(128,  (3, 3), name = 'conv2_1', strides = 1, padding='same', activation=LeakyReLU())(conv2)
    conv3 =   Conv2D(256,  (3, 3), name = 'conv3',   strides = 1, padding='same', activation=LeakyReLU())(conv2_1)
    conv3_1 = Conv2D(256,  (3, 3), name = 'conv3_1', strides = 2, padding='same', activation=LeakyReLU())(conv3)
    conv4 =   Conv2D(512,  (3, 3), name = 'conv4',   strides = 1, padding='same', activation=LeakyReLU())(conv3_1)
    conv4_1 = Conv2D(512,  (3, 3), name = 'conv4_1', strides = 2, padding='same', activation=LeakyReLU())(conv4)
    conv5 =   Conv2D(512,  (3, 3), name = 'conv5',   strides = 1, padding='same', activation=LeakyReLU())(conv4_1)
    conv5_1 = Conv2D(512,  (3, 3), name = 'conv5_1', strides = 2, padding='same', activation=LeakyReLU())(conv5)
    flow =    Conv2D(100,  (1, 1), name = 'flow2',   strides = 1, padding='same', activation='tanh')(conv5_1)
    flow =    Conv2D(2,    (1, 1), name = 'flow0',   strides = 1, padding='same', activation='linear')(flow)
    flat =    Flatten()(flow)
    dense =   Dense(500, activation=LeakyReLU())(flat)
    dense =   Dense(100, activation='tanh')(dense)
    input2 =  Input(shape=(9,))
    merged =  concatenate([input2, dense], axis=1)
    dense =   Dense(10,  activation=LeakyReLU())(merged)
    vel =     Dense(3,   activation='linear')(dense)

    model = Model(inputs=[input0, input1, input2], outputs=[flow, vel])
    rms = RMSprop(lr=0.001, rho=0.9, epsilon=10**-6, decay=0.0)
    model.compile(loss=['mae', 'mae'], optimizer=rms)
    model.summary()
    return model
