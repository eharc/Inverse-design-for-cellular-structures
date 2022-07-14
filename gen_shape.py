import argparse
import numpy as np
import scipy.io as sio
from numpy import save
import matplotlib.pyplot as plt

from ihgan import Model
from utils import ElapsedTimer
from utils import create_dir
from visualization_metrics import visualization

from plot import plot

if __name__ == "__main__":
    
    # Arguments
    parser = argparse.ArgumentParser(description='Shape generation')
    parser.add_argument('--mode', type=str, default='train', help='train or evaluate')
    args = parser.parse_args()
    assert args.mode in ['train', 'evaluate']
    
    # Hyperparameters for GAN
    noise_dim = 2
    train_steps = 9000
    save_interval = 0
    batch_size = 32
    
    #(3000,16) 34%
    #(4000,32) 30%
    #(9000,32) 26%
    #(9000,16) 28%
    
    DATA2=np.loadtxt('TAB_DATA_4.txt')
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
    C_tgt = C_test
    X_synth = model.synthesize(C_tgt)
    np.save('{}/dvar_synth.npy'.format(opt_dir), X_synth)
    sio.savemat('{}/dvar_synth.mat'.format(opt_dir), {'vect':X_synth})
    
    
    pca=visualization(X_test,X_synth,"pca")
    tsne=visualization(X_test,X_synth,"tsne")
    
    plo=plot(X_test,X_synth)