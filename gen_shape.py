import argparse
import numpy as np
import scipy.io as sio
from numpy import save
import matplotlib.pyplot as plt

from ihgan import Model
from utils import ElapsedTimer
from utils import create_dir
from visualization_metrics import visualization

import statistics as st

if __name__ == "__main__":
    
    # Arguments
    parser = argparse.ArgumentParser(description='Shape generation')
    parser.add_argument('--mode', type=str, default='train', help='train or evaluate')
    args = parser.parse_args()
    assert args.mode in ['train', 'evaluate']
    
    # Hyperparameters for GAN
    noise_dim = 2
    train_steps = 3000
    save_interval = 0
    batch_size = 16
    
    DATA2=np.loadtxt('TAB_DATA_3.txt')
    X=DATA2[:,0:7:]
    C=DATA2[:,7:]
    
    
    
    # Train/test split
    N = X.shape[0]
    split = int(0.8 * N)
    X_train = X[:split]
    C_train = C[:split]
    X_test = X[split:]
    C_test = C[split:]
        
    #Create a folder for the trained model
    model_dir = 'trained_model'
    create_dir(model_dir)

    # Train GAN
    model = Model(noise_dim, X.shape[1], C.shape[1], n_classes=2)
    if args.mode == 'train':
        timer = ElapsedTimer()
        model.train(X_train, C_train, X_test, C_test, batch_size=batch_size, train_steps=train_steps, 
                    save_interval=save_interval, save_dir=model_dir)
        elapsed_time = timer.elapsed_time()
        runtime_mesg = 'Wall clock time for training: %s' % elapsed_time
        print(runtime_mesg)
    else:
        model.restore(save_dir=model_dir)
        
    # Generate synthetic design variables given target material properties
    opt_dir = 'U:\STAGE2A\MMAE'
    C_tgt = C
    X_synth = model.synthesize(C_tgt)
    np.save('{}/dvar_synth.npy'.format(opt_dir), X_synth)
    sio.savemat('{}/dvar_synth.mat'.format(opt_dir), {'vect':X_synth})
    
    
    pca=visualization(X,X_synth,"pca")
    tsne=visualization(X,X_synth,"tsne")
    
    plt.figure(3)
    plt.plot(X_synth[:,0],color='r')
    plt.plot(X[:,0],color='b')
    plt.show()
    
    plt.figure(4)
    plt.plot(X[:,1],color='b')
    plt.plot(X_synth[:,1],color='r')
    plt.show()
    
    plt.figure(5)
    plt.plot(X_synth[:,2],color='r')
    plt.plot(X[:,2],color='b')
    plt.show()
    
    plt.figure(6)
    plt.plot(X[:,3],color='b')
    plt.plot(X_synth[:,3],color='r')
    plt.show()
    
    plt.figure(7)
    plt.plot(X_synth[:,4],color='r')
    plt.plot(X[:,4],color='b')
    plt.show()
    
    plt.figure(8)
    plt.plot(X[:,5],color='b')
    plt.plot(X_synth[:,5],color='r')
    plt.show()
    
    plt.figure(9)
    plt.plot(X_synth[:,6],color='r')
    plt.plot(X[:,6],color='b')
    plt.show()

    #calcul d'écart relatif
    ER_1=[]
    ER_2=[]
    ER_3=[]
    ER_4=[]
    ER_5=[]
    ER_6=[]
    ER_7=[]
    
    for i in range(len(X_synth)):
        er1=abs((X_synth[:,0][i]-X[:,0][i])/X[:,0][i])
        er2=abs((X_synth[:,1][i]-X[:,1][i])/X[:,1][i])
        er3=abs((X_synth[:,2][i]-X[:,2][i])/X[:,2][i])
        er4=abs((X_synth[:,3][i]-X[:,3][i])/X[:,3][i])
        er5=abs((X_synth[:,4][i]-X[:,4][i])/X[:,4][i])
        er6=abs((X_synth[:,5][i]-X[:,5][i])/X[:,5][i])
        er7=abs((X_synth[:,6][i]-X[:,6][i])/X[:,6][i])

        ER_1.append(er1)
        ER_2.append(er2)
        ER_3.append(er3)
        ER_4.append(er4)
        ER_5.append(er5)
        ER_6.append(er6)
        ER_7.append(er7)

        
    plt.figure(10)
    plt.plot(ER_1)
    plt.show()
    
    plt.figure(11)
    plt.plot(ER_2)
    plt.show()
    
    plt.figure(12)
    plt.plot(ER_3)
    plt.show()
    
    plt.figure(13)
    plt.plot(ER_4)
    plt.show()
    
    plt.figure(14)
    plt.plot(ER_5)
    plt.show()
    
    plt.figure(15)
    plt.plot(ER_6)
    plt.show()
    
    plt.figure(16)
    plt.plot(ER_7)
    plt.show()
    
    Mean_ER_1=st.mean(ER_1)
    Mean_ER_2=st.mean(ER_2)
    Mean_ER_3=st.mean(ER_3)
    Mean_ER_4=st.mean(ER_4)
    Mean_ER_5=st.mean(ER_5)
    Mean_ER_6=st.mean(ER_6)
    Mean_ER_7=st.mean(ER_7)
    
Mean=(Mean_ER_1+Mean_ER_2+Mean_ER_3+Mean_ER_4+Mean_ER_5+Mean_ER_6+Mean_ER_7)/7

    
    

