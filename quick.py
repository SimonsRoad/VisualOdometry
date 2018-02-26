from keras.models import Model, Sequential
from keras.layers import Input, Dropout, Dense, Conv2D, Flatten, Merge, Activation, Conv2DTranspose, concatenate, Flatten
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
import keras.backend as K
from keras.optimizers import RMSprop, Adam
import matplotlib.pyplot as plt
from PrepData import *
from Model import *
import time
import sys
import cv2
from keras.callbacks import EarlyStopping
import pickle

def getModel(h,w):
    input0 = Input(shape=(h, w, 3))
    input1 = Input(shape=(h, w, 3))
    input = concatenate([input0, input1], axis=3)
    conv1 =   Conv2D(64,   (3, 3), name = 'conv1',   strides = 2, padding='same', activation=LeakyReLU())(input)
    conv2 =   Conv2D(128,  (3, 3), name = 'conv2',   strides = 2, padding='same', activation=LeakyReLU())(conv1)
    conv3 =   Conv2D(256,  (3, 3), name = 'conv3_1', strides = 2, padding='same', activation=LeakyReLU())(conv2)
    conv4 =   Conv2D(512,  (3, 3), name = 'conv4_1', strides = 2, padding='same', activation=LeakyReLU())(conv3)
    conv5 =   Conv2D(1024, (3, 3), name = 'conv5_1', strides = 2, padding='same', activation=LeakyReLU())(conv4)
    trans =   Conv2D(2,    (1, 1), name = 'trans',   strides = 1, padding='same', activation='tanh')(conv5)

    input2 =  Input(shape=(9,))
    merged =  concatenate([input2, Flatten()(trans)], axis=1)
    dense =   Dense(128, activation=LeakyReLU())(merged)
    dense =   Dense(16,  activation=LeakyReLU())(dense)
    dense =   Dense(20,  activation=LeakyReLU())(dense)
    dense =   Dense(10,  activation='tanh')(dense)
    vel =     Dense(3,   activation='linear')(dense)

    model = Model(inputs=[input0, input1, input2], outputs=vel)
    rms = RMSprop(lr=10**-4, rho=0.9, epsilon=10**-6, decay=0.0)
    model.compile(loss='mae', optimizer=rms)
    model.summary()
    return model


of, vel, pos, DCM, img1, img2 = getMergedData([7])
m = getModel(320, 1152)
earlystop = EarlyStopping(monitor='loss', min_delta=10**-5, patience=5, verbose=1, mode='auto')
callbacks_list = [earlystop]
history = m.fit([img1, img2, DCM], vel, epochs=40, batch_size=32, verbose=1,  shuffle=True, callbacks=callbacks_list)

pred_vel_list = []
i = 0
while i < img1.shape[0]:
    inputImg1 = img1[i:i+10,:,:,:]
    inputImg2 = img2[i:i+10,:,:,:]
    dcm = DCM[i:i+10,:]
    pred_vel = m.predict([inputImg1, inputImg2, dcm])
    pred_vel_list.append(pred_vel)
    i += 10
    if i%100 == 0:
        print i

pred_vel = np.array(pred_vel_list)
print pred_vel.shape
pred_vel = np.reshape(pred_vel, (vel.shape[0], 3))
#pred_vel = pred_vel[:,3:]
diff = vel-pred_vel

print pred_vel.shape
print np.mean(diff)

pred_pos = pos2vel(pred_vel)
rmse = np.sqrt((np.asarray((np.subtract(pred_vel, vel))) ** 2).mean())
print 'rmse=%f' %(rmse)
rmse = np.sqrt((np.asarray((np.subtract(pred_pos, pos))) ** 2).mean())
print 'rmse=%f' %(rmse)


plt.figure()
plt.plot(vel, 'r')
plt.plot(pred_vel, 'b')
plt.show()


pred_pos = vel2pos(pred_vel)
plt.figure()
plt.plot(pos[:,0], pos[:,2], 'ro')
plt.plot(pred_pos[:,0], pred_pos[:,2], 'b.')
plt.show()
