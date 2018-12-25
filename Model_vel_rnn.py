from keras.models import Model
from keras.layers import *
from keras.layers.advanced_activations import *
from keras.losses import *
import keras.backend as K
from keras.optimizers import RMSprop, Adam
import tensorflow as tf
from keras.utils.training_utils import multi_gpu_model


def getSeqElem(i):
    def func(x):
        return x[:, None,i, :]
    return Lambda(func)

def AttachDense():
    def func(x):
        h = BatchNormalization()(x)
        h = Dense(64, kernel_regularizer=regularizers.l2(0.01),activation='linear')(h)
        h = LeakyReLU()(h)
        h = Dense(3, kernel_regularizer=regularizers.l2(0.01),activation='linear')(h)
        return h
    return Lambda(func)

def rnnModel(T):
    input_dim = (T, 6)
    input = Input(shape=input_dim)
    h = BatchNormalization()(input)
    h = LSTM(100, kernel_regularizer=regularizers.l2(0.01), return_sequences = True, activation='linear')(h)
    h = LeakyReLU()(h)
    h = BatchNormalization()(h)
    h = LSTM(100, kernel_regularizer=regularizers.l2(0.01), return_sequences = True, activation='linear')(h)
    h = LeakyReLU()(h)
    h = BatchNormalization()(h)
    h = LSTM(6, kernel_regularizer=regularizers.l2(0.01),return_sequences = True, activation='linear')(h)


    # h = BatchNormalization()(h)
    # h = LSTM(3, return_sequences = True, activation='linear')(h)


    #seq = [AttachDense()(getSeqElem(i)(h)) for i in range(0,T)]
    seq = [getSeqElem(i)(h) for i in range(0,T)]
    output = concatenate(seq, axis=1)

    single_model = Model(input=[input], output=[output])
    rms = RMSprop(lr=10**-3, rho=0.9, epsilon=10**-6, decay=0.0)
    single_model.compile(loss=mae, optimizer=rms, loss_weights=[1])

    single_model.summary()
    return single_model


if __name__ =='__main__':
    m = rnnModel(20)
