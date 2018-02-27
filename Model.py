from keras.models import Model, Sequential
from keras.layers import Input, Dropout, Dense, Conv2D, Flatten, Merge, Activation, Conv2DTranspose, concatenate, Flatten
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
import keras.backend as K
from keras.optimizers import RMSprop, Adam

def getModel(h,w):
    input0 = Input(shape=(h, w, 3))
    input1 = Input(shape=(h, w, 3))
<<<<<<< HEAD
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
=======
    input =  concatenate([input0, input1], axis=3)
    conv1 =  Conv2D(64,   (3, 3), name = 'conv0',   strides = 2, padding='same', activation=LeakyReLU())(input)
    conv2 =  Conv2D(128,  (3, 3), name = 'conv1',   strides = 2, padding='same', activation=LeakyReLU())(conv1)
    conv3 =  Conv2D(256,  (3, 3), name = 'conv2',   strides = 2, padding='same', activation=LeakyReLU())(conv2)
    conv4 =  Conv2D(512,  (3, 3), name = 'conv3',   strides = 2, padding='same', activation=LeakyReLU())(conv3)
    conv5 =  Conv2D(1024, (3, 3), name = 'conv4',   strides = 2, padding='same', activation=LeakyReLU())(conv4)
    trans =  Conv2D(2,    (1, 1), name = 'op_flow', strides = 1, padding='same', activation='tanh')(conv5)

    input2 = Input(shape=(9,))
    merged = concatenate([input2, Flatten()(trans)], axis=1)
    dense =  Dense(128, activation=LeakyReLU())(merged)
    dense =  Dense(64,  activation=LeakyReLU())(dense)
    dense =  Dense(16,  activation=LeakyReLU())(dense)
    vel =    Dense(3,   activation='linear')(dense)

    model = Model(inputs=[input0, input1, input2], outputs=vel)
    rms = RMSprop(lr=10**-5, rho=0.9, epsilon=10**-6, decay=0.0)
    model.compile(loss='mae', optimizer=rms)
>>>>>>> b3_evenlight_sg
    model.summary()
    return model
