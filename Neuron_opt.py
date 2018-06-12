import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from Neuron import Neuron_tf, V_pos, Ca_pos
from Optimizer import Optimizer
import tensorflow as tf
import numpy as np
from utils import  plots_output_double, plot_loss_rate, plot_vars
from data import get_data_dump, FILE_LV, DUMP_FILE, SAVE_PATH, FILE_NEUR
import pickle
import params
from tqdm import tqdm



class HH_opt(Optimizer):
    """Full Hodgkin-Huxley Model implemented in Python"""

    dim_batch= 1

    def __init__(self, neuron=None, init_p=params.give_rand(), fixed=[], constraints=params.CONSTRAINTS, loop_func=None, dt=0.1):
        Optimizer.__init__(self)
        if(neuron is not None):
            self.neuron = neuron
        else:
            self.neuron = Neuron_tf(init_p, loop_func=loop_func, dt=dt, fixed=fixed, constraints=constraints)
        self.optimized = self.neuron
        self.plot_vars = plot_vars


    """Define how the loss is computed"""
    def build_loss(self, w):
        cac = self.res[:, Ca_pos]
        out = self.res[:, V_pos]
        # if (self.neuron.num > 1):
        #     self.loss = []
        #     for i in range(self.neuron.num):
        #         losses_v = w[0] * tf.square(tf.subtract(out[:,:,i], self.ys_[0,:,:,i]))
        #         losses_ca = w[1] * tf.square(tf.subtract(cac[:,:,i], self.ys_[-1,:,:,i]))
        #         self.loss.append(tf.reduce_mean(losses_v + losses_ca))
        #     self.loss = tf.stack(self.loss)
        if(self.neuron.num > 1):
            losses_v = w[0] * tf.square(tf.subtract(out, self.ys_[0]))
            losses_ca = w[1] * tf.square(tf.subtract(cac, self.ys_[-1]))
            self.loss = losses_v + losses_ca
            self.loss = tf.reduce_mean(self.loss, axis=[0,1])
        else:
            losses_v = w[0] * tf.square(tf.subtract(out, self.ys_[0]))
            losses_ca = w[1] * tf.square(tf.subtract(cac, self.ys_[-1]))
            self.loss = losses_v + losses_ca
            self.loss = tf.reduce_mean(self.loss)



    def optimize(self, subdir, w=[1,0], epochs=500, l_rate=[0.9,9,0.95], suffix='', step=None, file=DUMP_FILE, reload=False):
        print(suffix, step)
        self.init(subdir, suffix, l_rate, w, neur=self.neuron)
        self.T, self.X, self.V, self.Ca = get_data_dump(file)

        if(self.neuron.loop_func == self.neuron.ik_from_v):
            self.Ca = self.V

        n_batch = self.X.shape[self.dim_batch]
        assert(self.neuron.dt == self.T[1] - self.T[0])
        # inputs

        #Xshape = [time, n_batch]
        xshape = [None, None]
        yshape = [2, None, None]
        if(self.neuron.num > 1):
            #add dimension for neurons trained in parallel
            #[time, n_batch, neuron]
            self.X = np.stack([self.X for _ in range(self.neuron.num)], axis=self.X.ndim)
            self.V = np.stack([self.V for _ in range(self.neuron.num)], axis=self.V.ndim)
            self.Ca = np.stack([self.Ca for _ in range(self.neuron.num)], axis=self.Ca.ndim)
            xshape.append(self.neuron.num)
            yshape.append(self.neuron.num)
        self.xs_ = tf.placeholder(shape=xshape, dtype=tf.float32, name='input_current')
        self.ys_ = tf.placeholder(shape=yshape, dtype=tf.float32, name='voltage_Ca')
        init_state = self.neuron.init_state
        initshape = list(init_state.shape)
        #reshape init state : [state, n_batch, n_neuron]
        initshape.insert(self.dim_batch, n_batch)
        self.init_state = np.stack([init_state for _ in range(n_batch)], axis=self.dim_batch)

        self.init_state_ = tf.placeholder(shape=initshape, dtype=tf.float32, name='init_state')

        self.res = tf.scan(self.neuron.step,
                      self.xs_,
                     initializer=self.init_state_)

        self.build_loss(w)
        self.build_train()

        summary = tf.summary.merge_all()

        with tf.Session() as sess:
            if(reload):
                """Get variables and measurements from previous steps"""
                self.saver.restore(sess, '%s%s'%(self.dir, SAVE_PATH))
                with open(self.dir+FILE_LV, 'rb') as f:
                    l,r,vars = pickle.load(f)
                losses = np.concatenate((l, np.zeros((epochs, self.neuron.num))))
                rates = np.concatenate((r, np.zeros(epochs)))
                len_prev = len(l)
            else:
                sess.run(tf.global_variables_initializer())
                vars = dict([(var, [val]) for var, val in self.neuron.init_p.items()])
                losses = np.zeros((epochs, self.neuron.num))
                rates = np.zeros(epochs)
                len_prev = 0

            vars = dict([(var, np.vstack((val, np.zeros((epochs, self.neuron.num))))) for var, val in vars.items()])

            for i in tqdm(range(epochs)):

                results = self.train_and_gather(sess, len_prev+i, losses, rates, vars)

                # if(losses[len_prev+i]<self.min_loss):
                #     self.plots_dump(sess, losses, rates, vars, len_prev + i)
                #     return i+len_prev

                for b in range(n_batch):
                    plots_output_double(self.T, self.X[:,b,0], results[:,V_pos,b], self.V[:,b,0], results[:,Ca_pos,b],
                                        self.Ca[:,b, 0], suffix='%s_trace%s_%s_%s' % (suffix, b, step, i + 1), show=False,
                                        save=True, l=0.7, lt=2)
                if(i%10==0 or i==epochs-1):
                    self.plots_dump(sess, losses, rates, vars, len_prev+i)

        return  -1



if __name__ == '__main__':
    pass
