from stochastic.diffusion import OrnsteinUhlenbeckProcess
from scipy.integrate import simps
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle
import time
import pdb
import csv

noiseFlag = 1   # 0 -> no noise 1 -> add noise
var = 1       # random Guassian noise to guarantee exploration

# --------------------- hyper parameters -----------------------

sampleCost = 0.1
MAX_STEPS = 10000           # steps to collect experiences
MEMORY_CAPACITY = 32768     # Memory Size (Size of FIFO (M))
LR_A = 0.0005       # learning rate for actor
LR_C = 0.001        # learning rate for critic
BATCH_SIZE = 128

s_dim = 1       # only one observations: x(t_i-1)
a_dim = 1       # only one action, T_i
a_bound = 20    # nomalize the value range of T_i -x~x -> 0~2x    (This controls Tmax, Tmax = 2 * a_bound)

# ---------------------- Gen OU process ------------------------
# takes time
precision = 0.01            # time precision, round t to this precision
mu = 0
sigma = 0.5
theta = 0.025
aa = 2*sigma/np.sqrt(2*theta)       # 2 standard dev from mu (for plotting)
Tall = MAX_STEPS * a_bound          # Length of training trajectory 
# OU = OrnsteinUhlenbeckProcess(speed=theta, mean=mu, vol=sigma, t=Tall)
# env = OU.sample(int(Tall/precision))
# times = OU.times(int(Tall/precision))
# plt.plot(times, env)
# plt.show()
# pickle.dump(env, open('OUdata/2Path_original_realReward', 'wb'))
# pdb.set_trace()

# ---------------------- load generated OU process ------------------------
env = pickle.load(open('OUdata/2Path_original_realReward', 'rb'))           # Training Trajectory
testEnv = pickle.load(open('OUdata/2Path_original_realReward_TEST', 'rb'))  # Evaluation Trajectory
testLen = len(testEnv) * precision

# ---------------------- Actor Critic NNs ----------------------------
class DNN_AC():
    def __init__(self, a_dim, s_dim):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim + a_dim + 1), dtype=np.float32)  # Initialize FIFO
        self.pointer = 0            # The number of experiences we've seen so far (just a counter)
        self.learnCnt = 0           # Training starts after seeing 1280 experiences, learnCnt counts the number of trainings
        self.var = var 
        self.sess = tf.Session()

        self.a_dim, self.s_dim = a_dim, s_dim
        self.state = tf.placeholder(tf.float32, [None, s_dim], 's')         # Input tensor of the Actor
        self.targetReward = tf.placeholder(tf.float32, [None, 1], 'r')      # Input tensor of (Target) Reward

        self.action = self._build_a(self.state, scope='actor', trainable=True)              #
        reward = self._build_c(self.state, self.action, scope='critic', trainable=True)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor')    # get all parameters of the Actor network 
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic')   
        self.saverA = tf.train.Saver(self.ae_params)
        self.saverC = tf.train.Saver(self.ce_params)

        # MSE
        self.v_loss = tf.losses.mean_squared_error(labels=self.targetReward, predictions= reward)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(self.v_loss, var_list=self.ce_params)

        self.a_loss = - tf.reduce_mean(reward)    # maximize the reward
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(self.a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, inputState):
        return self.sess.run(self.action, {self.state: inputState[np.newaxis, :]})[0]

    def learn(self):
        # sample a mini-batch
        if self.pointer >= MEMORY_CAPACITY:
            indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        else:
        	indices = np.random.randint(0, self.pointer, BATCH_SIZE)
        batch = self.memory[indices, :]
        bs = batch[:, :self.s_dim]                          # batch state 
        ba = batch[:, self.s_dim: self.s_dim + self.a_dim]  # batch action
        br = batch[:, -1: ]                                 # batch reward

        # update critic network (MSE)
        lossv, _ = self.sess.run([self.v_loss, self.ctrain], {self.state: bs, self.action: ba, self.targetReward: br})
        # update actor network (policy gradient)
        lossa, _ = self.sess.run([self.a_loss, self.atrain], {self.state: bs})


        if self.learnCnt % 100 == 0:
            self.var = self.var * 0.99

        self.learnCnt += 1

    def store_transition(self, state, action, reward):
        transition = np.hstack((state, action, reward))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, state, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(state, 32, name='l1', trainable=trainable)
            net = tf.nn.relu(net)
            net = tf.layers.dense(net, 64, name='l2', trainable=trainable)
            net = tf.nn.relu(net)
            net = tf.layers.dense(net, 32, name='l3', trainable=trainable)
            net = tf.nn.relu(net)
            action = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return action

    def _build_c(self, state, action, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 32
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.matmul(state, w1_s) + tf.matmul(action, w1_a) + b1
            net = tf.nn.relu(net)
            net = tf.layers.dense(net, 64, name='l11', trainable=trainable)
            net = tf.nn.relu(net)
            net = tf.layers.dense(net, 32, name='l12', trainable=trainable)
            net = tf.nn.relu(net)
            return tf.layers.dense(net, 1, trainable=trainable)  # reward(s,a)


    def saveParams(self, path1, path2):
        self.saverA.save(self.sess, path1)
        self.saverC.save(self.sess, path2)

    def loadParams(self, path1, path2):
        self.saverA.restore(self.sess, path1)
        self.saverC.restore(self.sess, path2)

# ---------------------- Actor Critic RL -----------------------------
def main_AC(DNN):
    # ---------------------------------------------------- initialization
    # initial state
    startIndex = 10             # Because we use X[t=0.1] to start, 0.1 = startIndex*precision
    cummulativeT = 0            # Keeps track of time progressed since 0
    TStore = []         # This array holds the sampling epochs (in time/seconds)
    costArray = []      # Cost of running the ith policy on the Evaluation Trajectory
    costArray1 = []
    xAxis1, app_t1, opt_t1 = optPolicy()    # If plotted we get the optimal and approximate policies
    policyStore = np.transpose(np.array([np.arange(mu-aa, mu+aa, 0.01)]))
    policyStore = np.column_stack([policyStore, opt_t1])        # The array of policies
    policy_cnt = 0      # Corresponds to j in Alg. 1
    # ---------------------------------------------------- start to run steps
    for index in range(MAX_STEPS):  # Index corresponds to i in Alg. 1
        try:
            state1 = np.array([env[startIndex]])
        except:
            break
        action = DNN.choose_action(state1) # the output of DNN is a scaled fraction (-1 to 1)
        if noiseFlag == 1:
            action = np.clip(np.random.normal(action, DNN.var), -1, 1) # add random noise to guarantee exploration
        
        action1 = action * a_bound + a_bound        # Convert the action \in [-1,1] to the appropriate range
        if action1 < precision:
            action1 = np.array([precision])         # Basically we don't allow actions to be smaller than precision. (round up)

        nextIndex = startIndex + np.rint(action1[0]/precision).astype(int)
        
        # ---------------------------------------------------- reward function
        reward = (-sampleCost - calcCost(env[startIndex:nextIndex]))/action1[0]

        # ---------------------------------------------------- store experience
        DNN.store_transition(state1, action, reward)    # This gets stored in the FIFO

        if DNN.pointer > BATCH_SIZE * 10:
            # ---------------------------------- evaluate
            if DNN.learnCnt % 200 == 0:
                cost_to_store, cost_to_store1 = Assess_Model(DNN.choose_action, testEnv)
                costArray.append(cost_to_store/testLen)
                costArray1.append(cost_to_store1/testLen)

                policy = calcPolicy(DNN.choose_action)
                policyStore = np.column_stack([policyStore, policy])
                policy_cnt += 1

                TStore.append(cummulativeT)
                
            # ---------------------------------- learn
            DNN.learn()

        print("%s: %s, %s"%(index, cummulativeT, action1))

        startIndex = nextIndex
        cummulativeT = cummulativeT + action1[0]
        # print(time.time()-start)

    # ---------------------------------------------------- Total cost VS time consumed
    optCost, _ = Assess_Optimal(testEnv)
    Steps = np.arange(len(TStore)) * 500

    xAxis = np.array(Steps)
    bench_opt = np.zeros(len(xAxis)) + optCost/testLen
    plt.figure()
    plt.plot(xAxis, np.array(costArray), 'b*-')
    plt.xlabel('Training steps')
    plt.ylabel('Cost')
    plt.plot(xAxis, bench_opt, 'k.-')
    plt.grid(True)

    plt.figure()
    plt.plot(TStore, np.array(costArray), 'b*-')
    plt.xlabel('Time')
    plt.ylabel('Cost')
    plt.plot(TStore, np.zeros(len(TStore)) + optCost/testLen, 'k.-')
    plt.grid(True)

    plt.show()
    pdb.set_trace()


    costArray_now = pickle.load(open("plot_policy/costArray", "rb"))
    Steps = np.arange(len(costArray_now)) * 200
    xAxis1 = Steps
    bench_opt = np.zeros(len(xAxis1)) + 0.0515

    costArray1_now = costArray1[:len(xAxis1)]
    plt.figure()
    plt.plot(xAxis1, np.array(costArray_now), 'bo-', markersize=5, mec='r', label='Learned Policy')
    plt.plot(xAxis1, np.array(costArray1_now), 'y-')
    plt.plot(xAxis1, np.array(costArray_now) - np.array(costArray1_now), 'r-')
    plt.xlabel('Training Steps')
    plt.ylabel('Evaluation Cost (unit time)')
    plt.plot(xAxis1, bench_opt, 'k.-', label='Optimal Policy')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.title('(a)')
    plt.show()

    # store - benchmark
    # pickle.dump(policyStore, open('plot_policy/policyOverTraining20', 'wb'))
    # pickle.dump(costArray, open('plot_policy/costArray', 'wb'))
    # pickle.dump(TStore, open('plot_policy/TStore', 'wb'))




def calcCost(data):
    if data[0] == 0:
        negativeEnv = (data - np.abs(data))/2
        area = np.abs(simps(negativeEnv, dx=precision))
        positiveEnv = (-data - np.abs(data))/2
        area = np.abs(simps(positiveEnv, dx=precision))
        area = (area1 + area2)/2
    elif data[0] > 0:
        negativeEnv = (data - np.abs(data))/2
        area = np.abs(simps(negativeEnv, dx=precision))
    else:
        positiveEnv = (-data - np.abs(data))/2
        area = np.abs(simps(positiveEnv, dx=precision))

    return area


# ---------- Assess the Trained Model using Prestored Data -----------
def Assess_Model(DNNactor, testEnv):
    # flag = 0 => period 1; flag = 1 => period 2
    startTime = time.time()
    state = testEnv[0]
    currentIndex = 0
    samplePoint = [0]
    while 1:
        action = DNNactor(np.array([state]))
        action1 = action * a_bound + a_bound
        if action1 < precision:
            action1 = np.array([precision])

        currentIndex = currentIndex + np.rint(action1[0]/precision).astype(int)
        try:
            state = testEnv[currentIndex]
        except:
            break
        samplePoint.append(currentIndex)

    # -------------------------------------------------------------  calcute the overall cost
    cost = len(samplePoint) * sampleCost
    cost1 = cost
    testEnv = np.array(testEnv)
    for index in range(1, len(samplePoint)):
        ii = samplePoint[index - 1]
        jj = samplePoint[index]
        area = calcCost(testEnv[ii:jj])
        cost += area

    return cost, cost1

def calcPolicy(DNNactor):       # Only for plotting purposes
    # the learned policy policy / sigma varies
    learned_t = []
    xAxis = np.arange(mu-aa, mu+aa, 0.01)

    for index in range(len(xAxis)):
        action = DNNactor(np.array([xAxis[index]]))
        action1 = action * a_bound + a_bound # now the precision is 0.01
        if action1 < precision:
            action1 = np.array([precision])
        learned_t.append(action1[0])
    return learned_t

def optPolicy():                # Only for plotting purposes
    xAxis1 = np.arange(mu-aa, mu+aa, 0.01)

    # optimal policy / sigma = 0.5
    with open('OptPolicy/OU_Process_0Mean_OptimalSampling_Numerical.csv', newline='') as csvfile:
        data = list(csv.reader(csvfile))
    opt_t1 = []
    for i in range(len(data)):
        if float(data[i][0]) >= mu-aa and float(data[i][0]) < mu+aa:
            opt_t1.append(float(data[i][1]))

   

    # approximate optimal policy / sigma = 0.5
    app_t1 = []
    T_star1 = (18*np.pi*sampleCost**2/sigma/sigma)**(1/3.)
    for index in range(len(xAxis1)):
        action = T_star1 + (1-np.exp(-theta*T_star1))*np.sqrt(np.pi)*abs(xAxis1[index])/np.sqrt(1-np.exp(-2*theta*T_star1))/sigma/np.sqrt(theta)
        app_t1.append(action)


    return xAxis1, app_t1, opt_t1

def Assess_Optimal(testEnv):
    # load optimal sampling policy
    with open('OptPolicy/OU_Process_0Mean_OptimalSampling_Numerical.csv', newline='') as csvfile:
        data = list(csv.reader(csvfile))

    xx = []
    yy = []
    for i in range(len(data)):
        xx.append(float(data[i][0]))
        yy.append(float(data[i][1]))

    foptimal = interp1d(xx, yy, kind='cubic')

    state = testEnv[0]
    currentIndex = 0
    samplePoint = [0]
    while 1:
        state = np.round(state,2)

        # use the optimal scheme
        if np.abs(state) > np.max(xx):
            action = a_bound * 2
        else:
            action = foptimal(state)

        if action <= np.array([0.01]):
            action1 = 1
        else:
            action1 = np.rint(np.round(action,2)/precision).astype(int)

        currentIndex = currentIndex + action1

        try:
            state = testEnv[currentIndex]
        except:
            break
        samplePoint.append(currentIndex)

    # -------------------------------------------------------------  calcute the overall cost
    cost = len(samplePoint) * sampleCost
    testEnv = np.array(testEnv)
    for index in range(1, len(samplePoint)):
        ii = samplePoint[index - 1]
        jj = samplePoint[index]
        area = calcCost(testEnv[ii:jj])
        cost += area
    return cost, len(samplePoint) * sampleCost

def plot_policy(DNNactor):
    xAxis, app_t1, opt_t1 = optPolicy()
    learnedPolicy = calcPolicy(DNNactor)
    ## ------------------------------------------------- PLOT CURVE
    plt.xlabel("State (current observation)")
    plt.ylabel("Policy (update interval)")
    plt.plot(xAxis, opt_t1, 'k-', label='Optimal Policy')
    plt.plot(xAxis, learnedPolicy, 'b-', label='Learned Policy')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()


if __name__=="__main__":
    # -------------------------------------------------------------  Actor Critic Main
    DNN = DNN_AC(a_dim, s_dim)
    main_AC(DNN)
    