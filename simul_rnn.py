import tensorflow as tf
import numpy as np
from random import sample

import pandas as pd
import matplotlib.pyplot as plt

import argparse
import random

FILE = 'AVAL1.csv'
PATH_NET = './'

PARAM_CHANNELS = {
        'k_fast__p__tau': 2.25518e-3,
        'k_fast__p__rate': 1,
        'k_fast__p__scale': 7.42636e-3,
        'k_fast__p__midpoint': -8.05232e-3,

        'k_fast__q__tau': 149.963e-3,
        'k_fast__q__rate': 1,
        'k_fast__q__scale': -9.97468e-3,
        'k_fast__q__midpoint': -15.6456e-3,

        'k_slow__n__tau': 25.0007e-3,
        'k_slow__n__rate': 1,
        'k_slow__n__scale': 15.8512e-3,
        'k_slow__n__midpoint': 19.8741e-3,

        'ca_boyle__e__tau': 0.100027e-3,
        'ca_boyle__e__rate': 1,
        'ca_boyle__e__scale': 6.74821e-3,
        'ca_boyle__e__midpoint': -3.3568e-3,

        'ca_boyle__f__tau': 150.88e-3,
        'ca_boyle__f__rate': 1,
        'ca_boyle__f__scale': -5.03176e-3,
        'ca_boyle__f__midpoint': 25.1815e-3,

        'ca_simple__e__tau': 0.100027e-3,
        'ca_simple__e__rate': 1,
        'ca_simple__e__scale': 6.74821e-3,
        'ca_simple__e__midpoint': -3.3568e-3,

        'ca_boyle__h__alpha' : 0.282473,
        'ca_boyle__h__k' : -1.00056e-11,
        'ca_boyle__h__ca_half' : 6.41889e-11
}

####
# disable logs
tf.logging.set_verbosity(tf.logging.ERROR)
###
# get data
df = pd.read_csv(FILE)
Y = np.array(df['trace'])
X = np.array(df['inputCurrent'])*10e-12
T = np.array(df['timeVector'])
#
# params
seqlen = X.shape[0]
state_size = 1
BATCH_SIZE = 1
num_classes = 1



INIT_POT = -45e-3 #V
INIT_CONC = 1e-13 #mol_per_cm3
RADIUS = 2.5e-4 #cm
SURFACE = RADIUS*RADIUS*3.14159 #cm2
CAPA = 1e-6 * SURFACE #F
DECAY_CA = 11.6e-3 #s
RHO_CA = 0.000239e-2 #mol_per_cm_per_A_per_s
REST_CA = 0 #M

EREV_k = -60e-3
EREV_ca = 40e-3
EREV_leak = -50e-3

N_HILL = 189e-9 #M

RESISTIVITY = 0.1 #kohm_cm


#computes one gate activation rate
def compute_rate(prefix, vprev, qprev, defaults, dt):

    mdp = tf.get_variable('%s__midpoint'%prefix, initializer=PARAM_CHANNELS['%s__midpoint'%prefix], dtype=tf.float32)
    scale = tf.get_variable('%s__scale'%prefix, initializer=PARAM_CHANNELS['%s__scale'%prefix], dtype=tf.float32)
    tau = tf.get_variable('%s__tau'%prefix, initializer=PARAM_CHANNELS['%s__tau'%prefix], dtype=tf.float32)

    sig = (vprev - mdp) / scale
    inf = tf.sigmoid(sig)
    #inf = tf.Print(inf, [inf], '%s steady state : '%prefix)

    #simple solver
    q = qprev + dt*(inf - qprev)/tau
    #more complex solver
    q = ((tau*dt) / (tau+dt)) * (qprev/dt + inf/tau)

    #q = tf.Print(q, [q], '%s : '%prefix)

    return q

def special_rate(prefix, cacprev, defaults):

    mdp = tf.get_variable('%s__midpoint' % prefix, initializer=PARAM_CHANNELS['%s__ca_half' % prefix], dtype=tf.float32)
    scale = tf.get_variable('%s__scale' % prefix, initializer=PARAM_CHANNELS['%s__k' % prefix], dtype=tf.float32)
    alpha = tf.get_variable('%s__alpha' % prefix, initializer=PARAM_CHANNELS['%s__alpha' % prefix], dtype=tf.float32)

    sig = (cacprev - mdp) / scale
    inf = tf.sigmoid(sig)
    q = 1 + (inf-1)*alpha

    return q


def update(hprev, i_sin, index, dt):
    vprev = hprev[0]
    cacprev = hprev[-1]
    index = index + 1

    g_leak = tf.get_variable('g_leak', initializer=5e-6) #S_per_cm2
    g_leak_res = g_leak * SURFACE

    g_kf = tf.get_variable('g_k_fast', initializer=7e-5)
    p = compute_rate('k_fast__p', vprev, hprev[1], PARAM_CHANNELS, dt)
    q = compute_rate('k_fast__q', vprev, hprev[2], PARAM_CHANNELS, dt)
    g_kf_res = g_kf * q * p ** 4 * SURFACE

    g_ks = tf.get_variable('g_k_slow', initializer=3e-3)
    n = compute_rate('k_slow__n', vprev, hprev[3], PARAM_CHANNELS, dt)
    g_ks_res = n * g_ks * SURFACE

    g_cab = tf.get_variable('g_ca_boyle', initializer=3e-3)
    e = compute_rate('ca_boyle__e', vprev, hprev[4], PARAM_CHANNELS, dt)
    f = compute_rate('ca_boyle__f', vprev, hprev[5], PARAM_CHANNELS, dt)
    h = special_rate('ca_boyle__h', cacprev, PARAM_CHANNELS)
    g_cab_res = g_cab * e ** 2 * f * h * SURFACE

    i_k = (EREV_k - vprev) * (g_ks_res+g_kf_res)
    i_leak = (EREV_leak - vprev) * g_leak_res
    i_ca = (EREV_ca - vprev) * g_cab_res
    i = i_ca + i_k + i_leak + i_sin

    v = vprev + (i / CAPA) * dt

    cac = cacprev + ((i_ca / SURFACE) * RHO_CA - ((cacprev - REST_CA) / DECAY_CA)) * dt
    cac = tf.maximum(cac, 0.) #mol per cm3

    #v = tf.Print(v, [v], 'Potential : ')
    #cac = tf.Print(cac, [cac], 'Calcium concentration : ')

    return tf.stack([v, p, q, n, e, f, h, cac], 0), i_sin, index, dt

def condition(hprev, x, index, div):
    return tf.less(index*div, DT)

# step operation
def step(hprev, x):

    div = tf.constant(500.)
    dt = DT/div
    index = tf.constant(0.)


    h = tf.while_loop(condition, update, (hprev, x, index, dt))[0]

    #h = update(0, DT, hprev)

    return h


if __name__ == '__main__':
    #
    # build graph
    tf.reset_default_graph()
    # inputs
    xs_ = tf.placeholder(shape=[None], dtype=tf.float32)
    ys_ = tf.placeholder(shape=[None], dtype=tf.float32)
    #
    # initial hidden state
    init_state = tf.placeholder(shape=[8], dtype=tf.float32, name='initial_state')


    DT = tf.constant(0.344, dtype=tf.float32)
    REST_CA = tf.constant(0., dtype=tf.float32)


    #
    # here comes the scan operation; wake up!
    #   tf.scan(fn, elems, initializer)

    states = tf.scan(step,
                     xs_,
                     initializer=init_state)
    #
    # optimization
    cacs = states[:,-1] * 1000
    cacs_pow = tf.pow(cacs, 3.8)
    cac_lum = cacs / (cacs_pow + N_HILL)
    losses = tf.square(tf.subtract(cac_lum, ys_))
    loss = tf.reduce_mean(losses)
    loss = tf.Print(loss, [loss], 'loss : ')
    train_op = tf.train.AdamOptimizer(learning_rate=0.02).minimize(loss)
    #
    # to generate or to train - that is the question.
    if True:
        #
        # training
        #  setup batches for training
        epochs = 1
        #
        # set batch size
        batch_size = BATCH_SIZE
        # training session
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_loss = 0
            try:
                for i in range(epochs):
                    for j in range(1):
                        xs, ys, t = X, Y, T
                        _, train_loss_, res, cac_l = sess.run([train_op, loss, states, cac_lum], feed_dict={
                            xs_: xs,
                            ys_: ys,
                            init_state : [INIT_POT, 0.5, 0.5, 0.5, 0.5, 0.5, 0, INIT_CONC]
                        })

                        plt.subplot(411)
                        plt.title('Potential')
                        plt.plot(t, res[:, 0], 'b')
                        plt.subplot(412)
                        plt.title('calcium concentration')
                        plt.plot(t, cac_l, 'r')
                        plt.subplot(413)
                        plt.title('input current')
                        plt.plot(t, xs, 'g')
                        plt.subplot(414)
                        plt.title('p')
                        plt.plot(t, ys, 'b')
                        #plt.plot(t, res[:,-4], 'black', t, res[:,-3], 'orange', t, res[:,-2], 'purple')
                        plt.show()



                    print('[{}] loss : {}'.format(i, train_loss))
                    train_loss = 0


                    """plt.subplot(221)
                    plt.plot(t, res[:, 0], 'b')
                    plt.subplot(222)
                    plt.plot(t, cac_l, 'r')
                    plt.subplot(223)
                    plt.plot(t, xs, 'g')
                    plt.subplot(224)
                    plt.plot(t, ys, 'r')
                    plt.show()"""



            except KeyboardInterrupt:
                print('interrupted by user at ' + str(i))
                #
                # training ends here;
                #  save checkpoint
                saver = tf.train.Saver()
                saver.save(sess, PATH_NET + 'vanilla1.ckpt', global_step=i)
    else:
        #
        # generate text
        """random_init_word = random.choice(idx2ch)
        current_word = ch2idx[random_init_word]
        #
        # start session
        with tf.Session() as sess:
            # init session
            sess.run(tf.global_variables_initializer())
            #
            # restore session
            ckpt = tf.train.get_checkpoint_state(PATH_NET)
            saver = tf.train.Saver()
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            # generate operation
            words = [current_word]
            state = None
            # set batch_size to 1
            batch_size = 1
            num_words = args['num_words'] if args['num_words'] else 111
            # enter the loop
            for i in range(num_words):
                if state:
                    feed_dict = {xs_: np.array(current_word).reshape([1, 1]),
                                 init_state: state_}
                else:
                    feed_dict = {xs_: np.array(current_word).reshape([1, 1])
                        , init_state: np.zeros([batch_size, state_size])}
                #
                # forward propagation
                preds, state_ = sess.run([predictions, last_state], feed_dict=feed_dict)
                #
                # set flag to true
                state = True
                #
                # set new word
                current_word = np.random.choice(preds.shape[-1], 1, p=np.squeeze(preds))[0]
                # add to list of words
                words.append(current_word)"""
        #########
        # text generation complete
        #
