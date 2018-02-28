import matplotlib.pyplot as plt
from PrepData_post import *
from Model_VelCov import *
from Model_PosCov import *
import time
import sys
from keras.callbacks import EarlyStopping
import pickle

def train_vel_cov():
    vel_gt, pos_gt, vel_pr, pos_pr = getMergedData([0, 2, 4, 6])
    input = vel_pr
    output = vel_gt - vel_pr

    m_cov = getVelQ()
    m_cov.load_weights('Weights/vel_cov.h5')
    earlystop = EarlyStopping(monitor='md', min_delta=10**-5, patience=5, verbose=1, mode='auto')
    history = m_cov.fit(input, output, epochs=500, batch_size=1024,
                                       verbose=1, shuffle=True,
                                       callbacks=[earlystop])
    loss = history.history['loss']
    plt.figure()
    plt.plot(loss, 'r')
    plt.show()
    m_cov.save_weights('Weights/vel_cov.h5')

def test_vel_cov():
    vel_gt, pos_gt, vel_pr, pos_pr = getMergedData([3])
    m_cov = getVelQ()
    m_cov.load_weights('Weights/vel_cov.h5')
    vel_cov = np.abs(m_cov.predict(vel_pr))

    error = vel_gt - vel_pr
    ed = np.zeros((error.shape[0], 1))
    md = np.zeros((error.shape[0], 1))
    for i in range(0, error.shape[0]):
        err = error[i,:]
        q = vel_cov[i,:]
        ed[i,:] = np.sqrt(err[0]**2 + err[1]**2 + err[2]**2)
        md[i,:] = np.sqrt(err[0]**2/q[0] + err[1]**2/q[1] + err[2]**2/q[2])

    plt.figure()
    plt.plot(ed, 'ro')
    plt.plot(md, 'b')
    plt.show()

def train_pos_cov():
    vel_gt, pos_gt, vel_pr, pos_pr = getMergedData([0, 2, 4, 6, 7])

    k = np.arange(pos_pr.shape[0])
    k = np.reshape(k, (-1, 1))
    x = np.concatenate([k, pos_pr], axis=1)
    y = np.concatenate((pos_gt, pos_pr), axis=1)

    model = getPosQ(x.shape[1])
    model.load_weights('Weights/pos_cov.h5')
    history = model.fit(x, y, epochs=1000, batch_size=512, verbose=1, shuffle=False)
    model.save_weights('Weights/pos_cov.h5')
    loss = history.history['loss']
    plt.figure()
    plt.plot(loss, 'r')
    plt.show()


def test_pos_cov():
    for seq in range(0,11):
        vel_gt, pos_gt, vel_pr, pos_pr = getMergedData([seq])

        k = np.arange(pos_pr.shape[0])
        k = np.reshape(k, (-1, 1))
        x = np.concatenate([k, pos_pr], axis=1)
        y = np.concatenate((pos_gt, pos_pr), axis=1)

        m_cov = getPosQ(x.shape[1])
        m_cov.load_weights('Weights/pos_cov.h5')
        pos_cov = (m_cov.predict(x))**2

        # print pos_cov
        np.savetxt('Results/Pred_Data/seq' + str(seq) + '_poscov.txt', pos_cov)
        np.savetxt('Results/Pred_Data/seq' + str(seq) + '_posgt.txt', pos_gt)
        np.savetxt('Results/Pred_Data/seq' + str(seq) + '_pospr.txt', pos_pr)

        error = vel_gt - vel_pr
        ed = np.zeros((error.shape[0], 1))
        md = np.zeros((error.shape[0], 1))
        for i in range(0, error.shape[0]):
            err = error[i,:]
            q = pos_cov[i,:]
            ed[i,:] = np.sqrt(err[0]**2 + err[1]**2 + err[2]**2)
            md[i,:] = np.sqrt(err[0]**2/q[0] + err[1]**2/q[1] + err[2]**2/q[2])

        # plt.figure()
        # plt.plot(ed, 'ro')
        # plt.plot(md, 'b')
        # plt.show()



if __name__=='__main__':
    type = int(sys.argv[1])
    if type==0:
        train_vel_cov()
    elif type==1:
        test_vel_cov()
    elif type==2:
        train_pos_cov()
    elif type==3:
        test_pos_cov()
#end
