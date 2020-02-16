from stochastic.diffusion import OrnsteinUhlenbeckProcess
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.interpolate import interp1d
import numpy as np
import pickle
import time
import pdb
import csv


precision = 0.01

def OU_process(theta = 0.025, mu = 0, sigma = 0.5, Tall = 100):
	OU = OrnsteinUhlenbeckProcess(speed=theta, mean=mu, vol=sigma, t=Tall)
	return OU.sample(int(Tall/precision))

def Gen_6OU(Tall):
    # # ---------------------------- Gen 3-path sin+OU ----------------------------
    mu = 0
    sigma = 0.3
    theta = 0.025
    one_env = OU_process(theta, mu, sigma, Tall)
    for _ in range(5):
    	temp = OU_process(theta, mu, sigma, Tall)
    	env = np.row_stack([one_env[100000:-1], temp[100000:-1]])
    return env

if __name__ == '__main__':
    Tall_training = 51000
    Tall_evaluation = 21000

    data = Gen_6OU(Tall = Tall_training)
    print(data.shape)
    pickle.dump(data, open('OU6path_training', 'wb'))

    data = Gen_6OU(Tall = Tall_evaluation)
    print(data.shape)
    pickle.dump(data, open('OU6path_evaluation', 'wb'))
    # pdb.set_trace()
    plt.plot(data[0,0:10000])
    plt.plot(data[1,0:10000])
    plt.plot(data[2,0:10000])
    plt.plot(data[3,0:10000])
    plt.plot(data[4,0:10000])
    plt.plot(data[5,0:10000])
    plt.show()