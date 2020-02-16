import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle
import time
import pdb
import csv

LR_A = 0.0005       # learning rate for actor
LR_C = 0.001        # learning rate for critic
beta = 0.01			# e^(-beta t)
SRrate = 0.1		# soft replacement

# ---------------------- RNN Actor Critic NNs ----------------------------
class FeedForward():
    def __init__(self, s_dim, a_dim, LenSeq, a_bound, MEMORY_CAPACITY = 10000, var = 0.1):
        self.s_dim, self.a_dim, self.LenSeq, self.a_bound, self.MEMORY_CAPACITY, self.var = s_dim, a_dim, LenSeq, a_bound, MEMORY_CAPACITY, var

        # store experiences for experience replay
        self.memory = np.zeros((self.MEMORY_CAPACITY, self.s_dim * self.LenSeq * 2 + self.a_dim + 1), dtype=np.float32)  # Initialize FIFO
        self.pointer = 0            # The number of experiences we've seen so far (just a counter)

        self.learnCnt = 0           # Training starts after seeing 1280 experiences, learnCnt counts the number of trainings

        tf.reset_default_graph()
        self.sess = tf.Session()

        ####################################################################################################### DNNs
        # inputs
        self.state = tf.placeholder(tf.float32, [None, self.LenSeq * self.s_dim])          # Input tensor of the Actor
        self.state_ = tf.placeholder(tf.float32, [None, self.LenSeq * self.s_dim])
        self.immediateReward = tf.placeholder(tf.float32, [None, 1])               # Input tensor of (Target) Reward

        # Real network
        with tf.variable_scope('Real'):
            self.action = self._build_a(self.state, scope='actor', trainable=True)
            self.reward = self._build_c(self.state, self.action, scope='critic', trainable=True)

        # Target Network
        with tf.variable_scope('Target'):
            self.action_ = self._build_a(self.state_, scope='actor', trainable=False)
            reward_ = self._build_c(self.state_, self.action_, scope='critic', trainable=False)

        # parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Real/actor')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Real/critic')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Target/actor')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Target/critic')

        self.saverA = tf.train.Saver(self.ae_params)
        self.saverC = tf.train.Saver(self.ce_params)

        # target net replacement
        self.soft_replace1 = [tf.assign(t, (1 - SRrate) * t + SRrate * e)
                             for t, e in zip(self.at_params, self.ae_params)]
        self.soft_replace2 = [tf.assign(t, (1 - SRrate) * t + SRrate * e)
                             for t, e in zip(self.ct_params, self.ce_params)]
        # MSE
        action1 = self.action * self.a_bound + self.a_bound
        
        # ==============================================================================================================
        # ==============================================================================================================
        # q_target = tf.div(self.immediateReward, action1)                      # MYOPIC
        q_target = tf.multiply(self.immediateReward, tf.div(1-tf.exp(-beta * action1), beta * action1)) + tf.multiply(tf.exp(-beta * action1), reward_)	# future
        # ==============================================================================================================
        # ==============================================================================================================

        self.v_loss = tf.losses.mean_squared_error(labels=q_target, predictions=self.reward)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(self.v_loss, var_list=self.ce_params)

        self.a_loss = - tf.reduce_mean(self.reward)    # maximize the reward
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(self.a_loss, var_list=self.ae_params)

        self.vloss = []
        self.aloss = []

        self.sess.run(tf.global_variables_initializer())

        # make sure the real and evaluate networks are the same
        assignSameNetworks1 = [tf.assign(t, e)
                             for t, e in zip(self.at_params, self.ae_params)]
        assignSameNetworks2 = [tf.assign(t, e)
                             for t, e in zip(self.ct_params, self.ce_params)]
        self.sess.run([assignSameNetworks1,assignSameNetworks2])


    def _build_a(self, state, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(state, 256, name='l1', trainable=trainable)
            net = tf.nn.relu(net)
            net = tf.layers.dense(net, 128, name='l2', trainable=trainable)
            net = tf.nn.relu(net)
            net = tf.layers.dense(net, 64, name='l3', trainable=trainable)
            net = tf.nn.relu(net)
            net = tf.layers.dense(net, 32, name='l4', trainable=trainable)
            net = tf.nn.relu(net)
            action = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return action

    def _build_c(self, state, action, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 256
            w1_s = tf.get_variable('w1_s', [self.LenSeq * self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.matmul(state, w1_s) + tf.matmul(action, w1_a) + b1
            net = tf.nn.relu(net)
            net = tf.layers.dense(net, 128, name='l11', trainable=trainable)
            net = tf.nn.relu(net)
            net = tf.layers.dense(net, 64, name='l12', trainable=trainable)
            net = tf.nn.relu(net)
            net = tf.layers.dense(net, 32, name='l13', trainable=trainable)
            net = tf.nn.relu(net)
            return tf.layers.dense(net, 1, trainable=trainable)  # reward(s,a)

    def choose_action(self, inputState, numSamples):
        sampledAction = self.sess.run(self.action, {self.state: inputState.reshape([numSamples, self.LenSeq * self.s_dim])})
        return sampledAction

    def store_transition(self, state, action, reward, state_):
        transition = np.hstack((state, action, reward, state_))
        index = self.pointer % self.MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def learn(self):
        # sample a mini-batch
        if self.pointer >= self.MEMORY_CAPACITY:
            indices = np.random.choice(self.MEMORY_CAPACITY, size=self.BATCH_SIZE)
        else:
        	indices = np.random.randint(0, self.pointer, self.BATCH_SIZE)
        batch = self.memory[indices, :]
        bs = batch[:, :self.LenSeq*self.s_dim]                          # batch state 
        ba = batch[:, self.LenSeq*self.s_dim: self.LenSeq*self.s_dim + self.a_dim]  # batch action
        br = batch[:, self.LenSeq*self.s_dim + self.a_dim: self.LenSeq*self.s_dim + self.a_dim + 1] # reward
        bs_ = batch[:, -self.LenSeq*self.s_dim:] # next action
        
        #### update critic network (MSE)
        lossv, _ = self.sess.run([self.v_loss, self.ctrain], {self.state: bs, self.action: ba, self.immediateReward: br, self.state_: bs_})

        #### update actor network (policy gradient)
        lossa, _ = self.sess.run([self.a_loss, self.atrain], {self.state: bs})
        
        if np.isnan(lossv):
            pdb.set_trace()

        if self.learnCnt % 100 == 0 and self.var > 0.0001:
            self.var = self.var * 0.99

        self.learnCnt += 1

        self.sess.run(self.soft_replace1)
        self.sess.run(self.soft_replace2)

        self.vloss.append(lossv)
        self.aloss.append(lossa)

    def saveParams(self, path1, path2):
        self.saverA.save(self.sess, path1)
        self.saverC.save(self.sess, path2)

    def loadParams(self, path1, path2):
        self.saverA.restore(self.sess, path1)
        self.saverC.restore(self.sess, path2)

