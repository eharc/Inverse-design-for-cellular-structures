# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 11:44:14 2022

@author: eharc
"""
"""ORIGAMI"""

import argparse
import numpy as np
import scipy.io as sio
from numpy import save
import matplotlib.pyplot as plt


from ihgan import Model
from utils import ElapsedTimer
from utils import create_dir
from visualization_metrics import visualization



if __name__ == "__main__":
    
    # Arguments
    parser = argparse.ArgumentParser(description='Shape generation')
    parser.add_argument('--mode', type=str, default='train', help='train or evaluate')
    args = parser.parse_args()
    assert args.mode in ['train', 'evaluate']
    
    # Hyperparameters for GAN
    noise_dim = 2
    train_steps =110000
    save_interval = 0
    batch_size = 128
    

    DATA1=np.loadtxt('U:\STAGE2A\ORIGAMI\origami_random.txt')
    X=DATA1[:,:2]   
    C=DATA1[:,2:]

#bon paramètres
#(10000,128):2.16min / (15000,128):3.29 min / (20000,128):4.41 min + / (30000,128): 11.39 min ++ /

#(20000,200) 0.65/
#(70000,128)0.25/
#(100000,128) 0.18/
    
    
    
    # Train/test split
    N = X.shape[0]
    split = int(0.9*N)
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
    opt_dir = 'U:\STAGE2A\ORIGAMI'
    

    C_tgt = C_test
    X_synth = model.synthesize(C_tgt)



    
    np.save('{}/dvar_synth.npy'.format(opt_dir), X_synth)
    sio.savemat('{}/dvar_synth.mat'.format(opt_dir), {'vect':X_synth})
    
    plt.figure(1)
    pca=visualization(X_test,X_synth,"pca")
    plt.show()
    
    plt.figure(2)
    tsne=visualization(X_test,X_synth,"tsne")
    plt.show()

from plot import plot

trace=plot(X_test,X_synth)
    



