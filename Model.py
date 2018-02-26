from keras.models import Model, Sequential
from keras.layers import Input, Dropout, Dense, Conv2D, Flatten, Merge, Activation, Conv2DTranspose, concatenate, Flatten
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
import keras.backend as K
from keras.optimizers import RMSprop, Adam

def getCNN(h,w):
    input0 = Input(shape=(h, w, 3))
    input1 = Input(shape=(h, w, 3))
    input = concatenate([input0, input1], axis=3)
    conv1 =   Conv2D(64,   (3, 3), name = 'conv1',   strides = 2, padding='same', activation=LeakyReLU())(input)
    conv2 =   Conv2D(128,  (3, 3), name = 'conv2',   strides = 2, padding='same', activation=LeakyReLU())(conv1)
    conv3 =   Conv2D(256,  (3, 3), name = 'conv3_1', strides = 2, padding='same', activation=LeakyReLU())(conv2)
    conv4 =   Conv2D(512,  (3, 3), name = 'conv4_1', strides = 2, padding='same', activation=LeakyReLU())(conv3)
    conv5 =   Conv2D(512,  (3, 3), name = 'conv5_1', strides = 2, padding='same', activation=LeakyReLU())(conv4)
    conv6 =   Conv2D(100,  (1, 1), name = 'flow2',   strides = 1, padding='same', activation='tanh')(conv5)
    flow =    Conv2D(2,    (1, 1), name = 'flow0',   strides = 1, padding='same', activation='linear')(conv6)
    fn = Model(inputs=[input0, input1], outputs=flow)
    rms = RMSprop(lr=2*10**-4, rho=0.9, epsilon=10**-6, decay=0.0)
    fn.compile(loss='mae', optimizer=rms)
    fn.load_weights('Weights/b2_light_cnn.h5')
    return fn

def getModel(h,w):
    input0 = Input(shape=(h, w, 3))
    input1 = Input(shape=(h, w, 3))
    input = concatenate([input0, input1], axis=3)
    conv1 =   Conv2D(64,   (3, 3), name = 'conv1',   strides = 2, padding='same', activation=LeakyReLU())(input)
    conv2 =   Conv2D(128,  (3, 3), name = 'conv2',   strides = 2, padding='same', activation=LeakyReLU())(conv1)
    conv3 =   Conv2D(256,  (3, 3), name = 'conv3_1', strides = 2, padding='same', activation=LeakyReLU())(conv2)
    conv4 =   Conv2D(512,  (3, 3), name = 'conv4_1', strides = 2, padding='same', activation=LeakyReLU())(conv3)
    conv5 =   Conv2D(512,  (3, 3), name = 'conv5_1', strides = 2, padding='same', activation=LeakyReLU())(conv4)
    conv6 =   Conv2D(100,  (1, 1), name = 'flow2',   strides = 1, padding='same', activation='tanh')(conv5)
    flow =    Conv2D(2,    (1, 1), name = 'flow0',   strides = 1, padding='same', activation='linear')(conv6)
    fn = Model(inputs=[input0, input1], outputs=flow)
    fn.load_weights('Weights/b2_light_cnn.h5')

    input2 =  Input(shape=(9,))
    flat =    Flatten()(flow)
    dense =   Dense(720, activation='tanh')(flat)

    merged =  concatenate([input2, dense], axis=1)
    dense =   Dense(180,  activation=LeakyReLU())(merged)
    dense =   Dense(40,  activation=LeakyReLU())(dense)
    dense =   Dense(20,  activation=LeakyReLU())(dense)
    dense =   Dense(10,  activation='tanh')(dense)
    vel =     Dense(3,   activation='linear')(dense)

    model = Model(inputs=[input0, input1, input2], outputs=[flow, vel])
    rms = RMSprop(lr=10**-4, rho=0.9, epsilon=10**-6, decay=0.0)
    model.compile(loss='mae', optimizer=rms)
    model.summary()
    return model


def getNewModel(input_dim):
    myinit = 'uniform'#initializers.RandomNormal(stddev=0.1)
    model = Sequential()
    model.add(Dense(input_dim/4, input_dim=input_dim, kernel_initializer="uniform", activation=LeakyReLU()))
    model.add(Dense(int(input_dim/16), kernel_initializer="uniform", activation=LeakyReLU()))
    model.add(Dense(int(input_dim/32), kernel_initializer="uniform", activation=LeakyReLU()))
    model.add(Dense(int(input_dim/64), kernel_initializer="uniform", activation='tanh'))
    model.add(Dense(3, kernel_initializer="uniform", activation='linear'))
    model.compile(loss='mae', optimizer='adam')
    return model
