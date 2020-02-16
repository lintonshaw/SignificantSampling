from DNNModel import FeedForward
from stochastic.diffusion import OrnsteinUhlenbeckProcess
from scipy.integrate import simps
from scipy import interpolate
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle
import time
import pdb
import csv


noiseFlag = 1  # 0 -> no noise, 1 -> add noise
var = 1      # random Guassian noise to guarantee exploration

# --------------------- hyper parameters -----------------------
sampleCost = 0.1    
MAX_STEPS = 2000          # steps to collect experiences
numSamples = 31           # number of samples per step
MEMORY_CAPACITY = 32768    # Memory Size (Size of FIFO (M))
BATCH_SIZE = 128
CELL_SIZE = 128
attention_size = 128 * 2
num_stacked_layers = 1
numBatch = 5				# number of training pper time step

s_dim = 6       # number of paths we are considering
a_dim = 1       # only one action, T_i
a_bound = 20    # nomalize the value range of T_i -x~x -> 0~2x    (This controls Tmax, Tmax = 2 * a_bound)

# ---------------------- Input a period of history to the NN ----------------------
LenSeq = 1      # Input sequence length (= 1 for OU process, longer for process with memory)
# 100 -> 1 sample/s
# 10 -> 10 sample/s
reso_interval = 100
start_Index = 30 + LenSeq * reso_interval

# ---------------------- Import Training and Evaluation Data ------------------------
precision = 0.01
precision1 = 0.01         
eva_period = 1000                                   # time precision, round t to this precision

mu = 0
sigma = 0.5
theta = 0.025
aa = 2*sigma/np.sqrt(2*theta)
xAxis0 = np.arange(mu-aa, mu+aa, 0.01)

# Load training data and evaluation data
env = pickle.load(open('./Gen_process_data/OU6path_training', 'rb'))         # Training Trajectory
bestEnv = np.min(env,0)
testEnv = pickle.load(open('./Gen_process_data/OU6path_evaluation', 'rb'))   # Evaluation Trajectory
testBestEnv = np.min(testEnv,0)
testLen = (len(testBestEnv)-start_Index) * precision

# ---------------------- Actor Critic RL -----------------------------
def main_AC(DNN):
    # ---------------------------------------------------- initialization
    # initial state
    costArray = []      # Cost of running the ith policy on the Evaluation Trajectory
    costArray_sample = []
    # ---------------------------------------------------- start to run steps
    for index in range(MAX_STEPS):
        # randomly sample experience from the training trajectory
        reStartIndex = np.random.randint(130 + LenSeq * reso_interval, len(bestEnv) - a_bound * 2 * 100, numSamples) # shape = (numSamples,)
        AllIndex = get_Index(reStartIndex) # shape = (LenSeq*numSamples,)
        state1 = np.transpose(env[:, AllIndex]) # shape = (LenSeq*numSamples, s_dim)

        # input is (numSamples,), output is (numSamples, 1)
        action = DNN.choose_action(state1, numSamples)
        
        if noiseFlag == 1:
            action = np.clip(np.random.normal(action, DNN.var), -1, 1) # add random noise to guarantee exploration
        
        action1 = action * a_bound + a_bound        # Convert the action \in [-1,1] to the appropriate range
        
        # minimum sampling rate
        action1[action1 < precision] = np.array([precision])
        

        # for storage
        action = (action1 - a_bound) / a_bound  # important, guarantee no zero T (will lead to nan values)
        state1 = state1.reshape(numSamples, LenSeq * s_dim)

        reNextIndex = reStartIndex + np.rint(action1/precision).astype(int)
        AllIndex = get_Index(reNextIndex)
        state2 = np.transpose(env[:, AllIndex]).reshape(numSamples, LenSeq * s_dim)
        # ---------------------------------------------------- reward function
        for ii in range(numSamples):
            # ---------------------------------------------------- reward function
            reReward = -sampleCost - calcCost(reStartIndex[ii], reNextIndex[ii], 0)
            DNN.store_transition(state1[ii], action[ii], reReward, state2[ii])

        if DNN.pointer > BATCH_SIZE * 10:
            # ---------------------------------- evaluate
            if DNN.learnCnt % eva_period == 0:
                cost_to_store, OnlysampleCost = Assess_Model(DNN.choose_action, testEnv, sampleCost)
                costArray.append(cost_to_store/testLen)
                costArray_sample.append(OnlysampleCost/testLen)
                print("DNN.var = ", DNN.var)
                print("Cost achieved on the evaluation set =", cost_to_store/testLen)
                DNN.saveParams('./params/actor.ckpt', './params/critic.ckpt')
                
            # ---------------------------------- learn
            for _ in range(numBatch):
                DNN.learn()

        
        if len(costArray) == 0:
            print("%s: state=%s, action=%s, cost=%s"%(index, state1[0], action1[0], 1))
        else:
            print("%s: state=%s, action=%s, cost=%s"%(index, state1[0], action1[0], costArray[-1]))

    cost_to_store, OnlysampleCost = Assess_Model(DNN.choose_action, testEnv, sampleCost)
    print("Cost achieved on the evaluation set =", cost_to_store/testLen)

    # plot how the cost decreases in the RL process
    xAxis = np.arange(len(costArray)) * eva_period
    xAxis1, costArray1 = xAxis, np.array(costArray)
    plt.figure()
    plt.plot(xAxis1, costArray1, 'b*-')
    plt.xlabel('Training steps')
    plt.ylabel('Cost')
    plt.grid(True)
    plt.legend(loc='upper right')
    # plt.yscale('log')
    # plt.show()

    pickle.dump(costArray1, open('./cost', 'wb'))
    DNN.saveParams('./params/actor.ckpt', './params/critic.ckpt')
    verify_policy(DNN.choose_action, testEnv[:,145000:155000])
    
    pdb.set_trace()

def calcCost(index1, index2, flag = 0):
    if flag == 0: # training
        chosenIndex = np.argmin(env[:,index1])
        costFunc = env[chosenIndex,index1:index2]-bestEnv[index1:index2]
        area = simps(costFunc, dx=precision)
    else: # evaluation
        chosenIndex = np.argmin(testEnv[:,index1])
        costFunc = testEnv[chosenIndex,index1:index2]-testBestEnv[index1:index2]
        area = simps(costFunc, dx=precision)

    return area

def get_Index(reStartIndex):
    AllIndex = []
    for ii in range(len(reStartIndex)):
        startIndex = reStartIndex[ii]
        AllIndex = np.r_[AllIndex, reso_interval + np.arange(startIndex - LenSeq * reso_interval, startIndex, reso_interval)]
    return AllIndex.astype(int)

def verify_policy(choose_action, testEnv):
    currentIndex = start_Index
    samplePoint = [currentIndex]
    while currentIndex < len(testEnv[0]):
        try:
            state1 = np.array(testEnv[:,reso_interval + np.arange(currentIndex - LenSeq * reso_interval, currentIndex, reso_interval)])
            action = choose_action(state1, 1)[0]
        except:
            break

        action1 = action * a_bound + a_bound
        if action1 < precision1:
            action1 = np.array([precision1])

        currentIndex = currentIndex + np.rint(action1[0]/precision).astype(int)
        samplePoint.append(currentIndex)

    plt.figure()
    for ii in range(len(testEnv)):
        plt.plot(np.arange(len(testEnv[0])) * precision, testEnv[ii,:])
    for index in range(len(samplePoint)-1):
        plt.plot([samplePoint[index] * precision,samplePoint[index] * precision],[-4,4], 'k', alpha = 0.5)
    plt.show()

def Assess_Model(choose_action, testEnv, sampleCost, tolerance = 40):
    startTime = time.time()
    currentIndex = start_Index
    samplePoint = [currentIndex]
    while currentIndex < len(testBestEnv):
        try:
            state1 = testEnv[:,reso_interval + np.arange(currentIndex - LenSeq * reso_interval, currentIndex, reso_interval)]
            action = choose_action(state1, 1)[0]
        except:
            break

        action1 = action * a_bound + a_bound
        if action1 < precision1:
            action1 = np.array([precision1])

        currentIndex = currentIndex + np.rint(action1[0]/precision).astype(int)
        samplePoint.append(currentIndex)

        if time.time() - startTime > tolerance:
            return 0, 0

    # -------------------------------------------------------------  calcute the overall cost
    Allcost = len(samplePoint) * sampleCost
    OnlysampleCost = Allcost
    for index in range(1, len(samplePoint)):
        ii = samplePoint[index - 1]
        jj = samplePoint[index]
        area = calcCost(ii,jj,1)
        Allcost += area
    print("Time consumed =", time.time() - startTime)
    return Allcost, OnlysampleCost

if __name__=="__main__":
    # -------------------------------------------------------------  Actor Critic Main
    print("-------------------------------------------------------")
    print("-------------------------------------------------------")
    print("Input Traffic = 6-Path OU")
    print("DNN Type = FeedForward")
    print("Reward = Future")
    print("L = 1")
    print("-------------------------------------------------------")
    print("-------------------------------------------------------")
    # -------------------------------------------------------------  Actor Critic Main
    DNN = FeedForward(s_dim, a_dim, LenSeq, a_bound, MEMORY_CAPACITY, var, CELL_SIZE, attention_size, BATCH_SIZE, num_stacked_layers)
    # DNN = RNN(s_dim, a_dim, LenSeq, a_bound, MEMORY_CAPACITY, var, CELL_SIZE, attention_size, BATCH_SIZE, num_stacked_layers)

    isTraining = 1

    if isTraining == 1:
        main_AC(DNN)
    else:
        DNN.loadParams('./params1/actor.ckpt', './params1/critic.ckpt')
        # cost_to_store, OnlysampleCost = Assess_Model(DNN.choose_action, testEnv, sampleCost, tolerance = 100)
        # print("Cost achieved on the evaluation set =", cost_to_store/testLen)
        verify_policy(DNN.choose_action, testEnv[:,128300:136300])
        pickle.dump(testEnv[:,128300:136300], open('./SixPathEnv', 'wb'))
        