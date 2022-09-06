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
    train_steps =200000
    save_interval = 0
    batch_size =32
    
#datall2908.txt
#dat3108.txt

    DATA1=np.loadtxt('D:\STAGE2A\ORIGAMI\datall2908.txt')
    X=DATA1[:,:2]   
    C=DATA1[:,2:]
    
    
    # Train/test split
    N = X.shape[0]
    split = int(0.7*N)
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
    opt_dir = 'D:\STAGE2A'
    

    C_tgt = C_test
    X_synth = model.synthesize(C_tgt)



    
    np.save('{}/dvar_synth.npy'.format(opt_dir), X_synth)
    sio.savemat('{}/dvar_synth.mat'.format(opt_dir), {'vect':X_synth})
    
    plt.figure(10)
    pca=visualization(X_test,X_synth,"pca")
    plt.show()
    
    plt.figure(11)
    tsne=visualization(X_test,X_synth,"tsne")
    plt.show()

from plot import plot

trace=plot(X_test,X_synth)
    