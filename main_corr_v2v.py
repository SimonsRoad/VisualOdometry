from PrepData_post import *
import matplotlib.pyplot as plt
import numpy as np
from Model_vel_rnn import *
import sys

def main():
    T = 10
    x, y = getMergedSeqData([0, 2, 4, 6], T)
    print(x.shape)
    print(y.shape)
    model = rnnModel(T)
    for i in range (0, 10):
        model.fit(x,y, epochs = 5, batch_size = 256,  validation_split=0.2, shuffle=True)
        model.save_weights('quickTestLSTM.h5')

def test():
    T = 10
    x, y = getMergedSeqData([0], T)
    model = rnnModel(T)
    model.load_weights('quickTestLSTM.h5')
    vp_seq = model.predict(x)
    print(vp_seq.shape)
    vp = np.zeros((vp_seq.shape[0]+T, 6))
    for i in range(0, vp_seq.shape[0]):
        vseq = vp_seq[i,:]
        vp[i:i+T,:] = vseq

    DCM, pos, vel = readGT(0)
    plt.figure()
    plt.plot(vp[:,0:3], 'b')
    plt.plot(pos, 'r')
    plt.show()





if __name__=='__main__':
    type = int(sys.argv[1])
    if type==0:
        main()
    elif type==1:
        test()
