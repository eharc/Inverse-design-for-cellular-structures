
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import statistics as st
import numpy as np
from data_processing import Normalizer
from data_processing import postprocess_design_variables, preprocess_design_variables, preprocess_material_properties
from scipy.signal import lfilter


class Model(object):
    
    def __init__(self, dim_Z, dim_X, dim_C,dim_C_test,X_test, n_classes=2):
        self.X_test=X_test
        self.dim_C_test=dim_C_test
        self.dim_Z = dim_Z
        self.dim_X = dim_X
        self.dim_C = dim_C
        self.n_classes = n_classes
        tf.reset_default_graph()
        
    def generator(self, z, c, reuse=tf.AUTO_REUSE):
        
        with tf.variable_scope('Generator', reuse=reuse):
            
            zc = tf.concat([z, c], axis=1)
            
            x = tf.layers.dense(zc, 64)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            
            x = tf.layers.dense(x, 128)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            
            x = tf.layers.dense(x, 256)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            
            x1 = tf.layers.dense(x, self.n_classes-1)

            x = tf.concat([x1], axis=1, name='gen')

            return x
        
        
    def discriminator(self, x, c, reuse=tf.AUTO_REUSE):
        
        with tf.variable_scope('Discriminator', reuse=reuse):
            
            xc = tf.concat([x, c], axis=1)
            
            y = tf.layers.dense(xc, 256)
            y = tf.nn.leaky_relu(y, alpha=0.2)
            
            y = tf.layers.dense(y, 128)
            y = tf.nn.leaky_relu(y, alpha=0.2)
            
            y = tf.layers.dense(y, 64)
            y = tf.nn.leaky_relu(y, alpha=0.2)
            
            y = tf.layers.dense(y, 1)
            
            return y
        
    def predictor(self, x, reuse=tf.AUTO_REUSE):
        
        with tf.variable_scope('Predictor', reuse=reuse):
            
            y = tf.layers.dense(x, 256)
            y = tf.nn.leaky_relu(y, alpha=0.2)
            
            y = tf.layers.dense(y, 128)
            y = tf.nn.leaky_relu(y, alpha=0.2)
            
            y = tf.layers.dense(y, 64)
            y = tf.nn.leaky_relu(y, alpha=0.2)
            
            y = tf.layers.dense(y, self.dim_C)
            
            return y
        
            
    def physics_loss1(self, c,batch_size):
        f = []
        for i in range(batch_size):
            P1 = 1.5766*c[i,0]**(-0.448)
            f.append(P1)
        return f
        
    def physics_loss2(self, c,batch_size):
        f = []
        for i in range(batch_size):
            P2 =  1.7983*c[i,1]**(-0.505)
            f.append(P2)
        return f
            

    def train(self, X_train, C_train, X_test, C_test,X,C, train_steps, batch_size, 
              save_interval=0, save_dir='.'):
        
        print('Training model ...')

        
        # Normalize training data
        self.normalizer_X = Normalizer(data=X)
        self.normalizer_C = Normalizer(data=C)
        np.save('{}/bounds_dvar.npy'.format(save_dir), self.normalizer_X.bounds)
        np.save('{}/bounds_mat_prp.npy'.format(save_dir), self.normalizer_C.bounds)
        
        X_train_ = preprocess_design_variables(X_train, self.normalizer_X)
        X_test_ = preprocess_design_variables(X_test, self.normalizer_X)
        C_train_ = preprocess_material_properties(C_train, self.normalizer_C)
        C_test_ = preprocess_material_properties(C_test, self.normalizer_C)
        
        # Inputs
        self.x = tf.placeholder(tf.float32, shape=[None, self.dim_X], name='data')
        self.z = tf.placeholder(tf.float32, shape=[None, self.dim_Z], name='noise')
        self.c = tf.placeholder(tf.float32, shape=[None, self.dim_C], name='condition')
        
        # Outputs

        d_real = self.discriminator(self.x, self.c)
        self.x_fake = self.generator(self.z, self.c)


        d_fake = self.discriminator(self.x_fake, self.c)
        c_pred_real = self.predictor(self.x)
        c_pred_fake = self.predictor(self.x_fake)

        
        loss_phy1=tf.reduce_mean(tf.abs(tf.math.square(self.x_fake-self.physics_loss1(C_test,self.dim_C_test))))
        loss_phy2=tf.reduce_mean(tf.abs(tf.math.square(self.x_fake-self.physics_loss2(C_test,self.dim_C_test))))
        
        # Losses
        # Cross entropy losses for D
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=tf.ones_like(d_real)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.zeros_like(d_fake)))
        
        # Cross entropy losses for G

        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.ones_like(d_fake)))

        # L1 loss for c
        c_loss_real = tf.reduce_mean(tf.abs(c_pred_real-self.c))
        c_loss_fake = tf.reduce_mean(tf.abs(c_pred_fake-self.c))
        

        # Optimizers
        d_optimizer = tf.train.AdamOptimizer(learning_rate=0.000003, beta1=0.5)
        g_optimizer = tf.train.AdamOptimizer(learning_rate=0.000003, beta1=0.5)
        
        # Generator variables
        gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
        # Discriminator variables
        dis_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')
        # Predictor variables
        pred_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Predictor')
        
        gamma = 20
        eta=5
        beta=1
 
        # Training operations

        d_train = d_optimizer.minimize(d_loss_real+d_loss_fake+gamma*c_loss_real, var_list=[dis_vars, pred_vars])
        g_train = g_optimizer.minimize(beta*g_loss+gamma*c_loss_fake+eta*(loss_phy1+loss_phy2), var_list=[gen_vars])
        
        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()
        
        # Create summaries to monitor losses
        tf.summary.scalar('D_loss_for_real', d_loss_real)
        tf.summary.scalar('D_loss_for_fake', d_loss_fake)
        tf.summary.scalar('G_loss', g_loss)
        tf.summary.scalar('C_loss_for_real', c_loss_real)
        tf.summary.scalar('C_loss_for_fake', c_loss_fake)
        tf.summary.scalar('loss_phy1', loss_phy1)
        tf.summary.scalar('loss_phy2', loss_phy2)

        # Merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()
        
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
        # Start training
        self.sess = tf.Session()
        
        # Run the initializer
        self.sess.run(init)
        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter('{}/logs'.format(save_dir), graph=self.sess.graph)
        DLR=[]
        DLF=[]
        GL=[]
        CLR=[]
        CLF=[]
        PL1=[]
        PL2=[]
        
        Output_gen=np.zeros(shape=[batch_size, 0])
    
        for t in range(train_steps):

            
            ind = np.random.choice(X_train_.shape[0], size=batch_size, replace=False)
            x_batch = X_train_[ind]
            c_batch = C_train_[ind]





            noise = np.random.normal(scale=0.5, size=(batch_size, self.dim_Z))
            summary_str, _, _, dlr, dlf, gl, clr, clf ,pl1,pl2= self.sess.run([merged_summary_op, d_train, g_train, 
                                                                       d_loss_real, d_loss_fake, g_loss, c_loss_real, c_loss_fake,loss_phy1,loss_phy2], 
                                                                      feed_dict={self.x: x_batch, self.z: noise,
                                                                                 self.c: c_batch})
            DLR.append(dlr)
            DLF.append(dlf)
            GL.append(gl)
            CLR.append(clr)
            CLF.append(clf)
            PL1.append(pl1)
            PL2.append(pl2)
            
            i=self.synthesize(c_batch,noise)
            Output_gen=np.concatenate((Output_gen,i),axis=1)

            summary_writer.add_summary(summary_str, t+1)
            
            # Show messages
            log_mesg = "%d: [D] real %f fake %f L1 %f" % (t+1, dlr, dlf, clr)
            log_mesg = "%s  [G] fake %f L1 %f pl1 %f pl2 %f" % (log_mesg, gl, clf,pl1,pl2)
            print(log_mesg)
            
            if save_interval>0 and (t+1)%save_interval==0 or t+1 == train_steps:
                # Save the variables to disk.
                save_path = saver.save(self.sess, '{}/model'.format(save_dir))
                print('Model saved in path: %s' % save_path)


        plt.figure(6)
        plt.plot(DLR)
        plt.title('Discriminator loss real')
        
        plt.figure(7)
        plt.plot(DLF)
        plt.title('Discriminator loss fake')
        
        plt.figure(8)
        plt.plot(GL)
        plt.title('Generator loss')
        
        plt.figure(9)
        plt.plot(CLR)
        plt.title('Predictor loss real')
        
        plt.figure(15)
        plt.plot(CLF)
        plt.title('Predictor loss fake')
        
        plt.figure(25)
        plt.plot(PL1)
        plt.title('Ph1')
        
        plt.figure(26)
        plt.plot(PL2)
        plt.title('Ph2')

        Savefile = np.savetxt('npfile.txt', np.transpose(Output_gen))
                    
    def restore(self, save_dir='.'):
        
        print('Loading model ...')
        
        self.normalizer_X = Normalizer(bounds=np.load('{}/bounds_dvar.npy'.format(save_dir)))
        self.normalizer_C = Normalizer(bounds=np.load('{}/bounds_mat_prp.npy'.format(save_dir)))
        
        self.sess = tf.Session()
        # Load meta graph and restore weights
        saver = tf.train.import_meta_graph('{}/model.meta'.format(save_dir))
        saver.restore(self.sess, tf.train.latest_checkpoint('{}/'.format(save_dir)))
        
        # Access and create placeholders variables            
        graph = tf.compat.v1.get_default_graph()
        self.x = graph.get_tensor_by_name('data:0')
        self.z = graph.get_tensor_by_name('noise:0')
        self.c = graph.get_tensor_by_name('condition:0')
        self.x_fake = graph.get_tensor_by_name('Generator/gen:0')

    def synthesize(self, condition, noise=None):
        condition = preprocess_material_properties(condition, self.normalizer_C)
        if noise is None:
           noise = np.random.normal(scale=0.5, size=(condition.shape[0], self.dim_Z))
        X = self.sess.run(self.x_fake, feed_dict={self.z: noise, self.c: condition})
        X = postprocess_design_variables(X, self.normalizer_X)
        return X