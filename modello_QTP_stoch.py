'''
by Andrea Macri'
last update 04/09/2023
'''

import numpy as np
import math as m
import statistics as stat
import matplotlib.pyplot as plt
import random as rnd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import deque

from tqdm import tqdm
import copy
import time
from statistics import pvariance, variance, stdev
import seaborn as sns

'''constrained'''

in_features = 4

PERM_IMP = 0.001 #permantent impact
TEMP_IMP = 0.002 #temporary impact
VOLA = 0.00001 #volatility
MU = 0 # drift
LAMBDA = 1 # non in uso
TIME_VAR = False
IMP_DECR = False
INV = 20 # inventario
PASSI = 10 #discretizzazione
STEP = 100 # abbassa la epsilon ogni 100 azioni compiute
NUMIT = 10#_000# traiettorie di train, quelle di test sono sempre la meta', il numero totale delle azioni compiute in train sarà NUMIT x PASSI
LR = 0.0001 # learning rate
BATCH = 16 # batch size
P_MIN = 9.97 # historical min
P_MAX = 10.03 # historical max
LAMBDA_A = 5
LAMBDA_B = 5
SIGMA_B = 0.002
SIGMA_A = 0.002    
#np.random.seed(14)

class DQN(nn.Module):
    '''
    Fully connected NN with 30 nodes in each layer, for 6 layers using sequential layers
    '''
    def __init__(self, in_size, hidden_layers_size):
        super(DQN, self).__init__()

        layers = []
        layers.append(nn.Linear(in_size, hidden_layers_size))
        for _ in range(5):  # Adding 5 hidden layers
            layers.append(nn.Linear(hidden_layers_size, hidden_layers_size))
            layers.append(nn.LeakyReLU())

        layers.append(nn.Linear(hidden_layers_size, 1))
        
        self.sequential_layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.sequential_layers(x)

class Ambiente():

    def __init__(self, S0 = 10, mu = MU, kappa = PERM_IMP, action = 0, sigma = VOLA, lambd = LAMBDA, t0 = 0, t = 1, T = 3600, inv = INV): #T = 1.0, M = 7200, I = 1_000
        
        self.S0 = S0
        self.mu = mu
        self.kappa = kappa
        self.action = action
        self.sigma = sigma
        self.dt = 1/T
        self.T = T
        self.t0 = t0
        self.tau = t-t0
        self.lambd = lambd
        self.initial_capital = inv

    def abm(self, seed = 14, numIt=10):
        '''
        returns a matrix of Arithmetic Brownian Motion paths
        '''
        N = self.T
        I = numIt
        dt= 1.0 / self.T
        X = np.zeros((N + 1, I), dtype=float)
        X[0] = self.S0
        for i in range(N):
            X[i + 1] = X[i] + (self.mu - ((self.kappa) * self.action)) * dt + self.sigma * np.sqrt(dt) * np.random.standard_normal(I) 

        return np.abs(X)
    
    def stoch_imp(self, retImp = True, seed=14, numIt=100, k_0 = PERM_IMP, a_0 = TEMP_IMP):
        '''
        Returns a matrix of Arithmetic Brownian Motion paths
        '''
        lambda_a = LAMBDA_A
        lambda_b = LAMBDA_B
        theta_b = PERM_IMP
        theta_a = TEMP_IMP
        sigma_b = SIGMA_B
        sigma_a = SIGMA_A                      
        N = self.T
        I = numIt
        dt = 1.0 / N
        X = np.zeros((N, I), dtype=float)
        k = np.zeros((N, I), dtype=float)
        a = np.zeros((N, I), dtype=float)
        a[0] = a_0
        k[0] = k_0
        X[0] = 10

        # Generate correlated standard normal draws
        corr_matrix = np.array([[1.0, 0.9], [0.9, 1.0]])  # Define the correlation matrix
        L = np.linalg.cholesky(corr_matrix)  # Perform Cholesky decomposition

        rand_nums = np.random.standard_normal((2, I))  # Generate standard normal draws for each iteration

        rand_nums_corr = np.matmul(L, rand_nums)  # Apply Cholesky decomposition

        for i in range(1,N):
            k[i] = k[i-1] + (lambda_b * (theta_b - k[i-1])) * dt + sigma_b * np.sqrt(k[i-1]) * np.sqrt(dt) * rand_nums_corr[0, :]
            a[i] = a[i-1] + (lambda_a * (theta_a - a[i-1])) * dt + sigma_a * np.sqrt(a[i-1]) * np.sqrt(dt) * rand_nums_corr[1, :]
            #print(f"Step {i + 1}: k = {k[i + 1]}, a = {a[i + 1]}")
            # Generate new correlated standard normal draws for the next time step
            rand_nums = np.random.standard_normal((2, I))
            rand_nums_corr = np.matmul(L, rand_nums)

        return k, a

    def inventory_action_transform(self, q, x):

        q_0 = self.initial_capital + 1

        q = q / q_0 - 1
        x = x / q_0
        r = m.sqrt(q ** 2 + x ** 2)
        theta = m.atan((-x / q))
        z = -x / q

        if theta <= m.pi / 4:
            r_tilde = r * m.sqrt((pow(z, 2) + 1) * (2 * (m.cos(m.pi / 4 - theta)) ** 2))

        else:

            r_tilde = r * m.sqrt((pow(z, -2) + 1) * (2 * (m.cos(theta - m.pi / 4)) ** 2))

        return 2 * (-r_tilde * m.cos(theta)) + 1, 2 * (r_tilde * m.sin(theta)) - 1

    def time_transform(self, t):

        tc = (PASSI - 1) / 2
        return (t - tc) / tc

    def price_normalise(self, price, min_p, max_p):
        """
        Normalizes a matrix of real numbers between 1 and -1 domain using min-max normalization.
        """
        # Takes the minimum and maximum values for the whole dataset
        min_vals = min_p 
        max_vals = max_p 
        range_vals = max_vals - min_vals

        # Perform column-wise -1,1 normalization
        normalized_matrix = 2 * (price - min_vals) / range_vals - 1

        if normalized_matrix   > 1: normalized_matrix = 1
        elif normalized_matrix <-1: normalized_matrix = -1
         

        return normalized_matrix

    def normalise(self, inventory, time, price, x, min_p, max_p):
        '''
        performs the normalisation in the range [-1,1] for the feature of the NN
        '''
        q, x = self.inventory_action_transform(inventory, x)
        t = self.time_transform(time)
        p = self.price_normalise(price, min_p, max_p)
        return q, t, p, x#

class ReplayMemory():
    '''
    Experience replay memory
    '''

    def __init__(self, capacity):

        self.memory = deque([], maxlen=capacity)

    def add(self, inv, time, p, x, next_inv, next_time, next_price, reward): 
        
        self.memory.append([inv, time, p, x, next_inv, next_time, next_price, reward])

    def sample(self, batch_size):
        
        return rnd.sample(self.memory, batch_size)
    
    def halve(self):

        for i in range(round(self.__len__()/2)):
            del self.memory[i]

    def __len__(self):
        
        return len(self.memory)

class Agente():

    def __init__(self, inventario, numTrain):

        self.train = numTrain
        self.maxlen = 15_000
        self.memory = ReplayMemory(self.maxlen)   
        self.env = Ambiente()
        self.main_net = DQN(in_size=in_features, hidden_layers_size=30)
        self.target_net = DQN(in_size=in_features, hidden_layers_size=30)

        for p in self.target_net.parameters():
            p.requires_grad = False

        self._update_target_net()

        self.learning_rate = LR
        self.optimizer = optim.Adam(params=self.main_net.parameters(), lr=self.learning_rate)
        self.time_subdivisions = PASSI
        self.inventory = inventario
        self.a_penalty = TEMP_IMP
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.batch_size = BATCH
        self.gamma = 1
        self.timestep = 0
        self.update_target_steps = STEP
        self.lots_size = 100
        self.matrix = np.zeros((self.inventory + 1, self.time_subdivisions))

    def _update_target_net(self):
        '''
        private method of the class: it refreshes the weight matrix of the target NN 
        '''

        self.target_net.load_state_dict(self.main_net.state_dict())

    def eval_Q(self, state, act, p_min, p_max, type = 'tensor', net = 'main'):
        '''
        Evaluates the Q-function
        '''
        if type == 'scalar' :

            q, t, p, x = Ambiente().normalise(state[0], state[1], state[2], act, p_min, p_max)
            in_features = torch.tensor([q, t, p, x], dtype=torch.float)

        if type == 'tensor':

            features = []

            for i in range(len(state)):

                q, t, p, x = Ambiente().normalise(state[i][0], state[i][1], state[i][2], act[i], p_min, p_max)
                features.append(torch.tensor([q, t, p, x], dtype=torch.float))

            in_features = torch.stack(features)
            in_features.type(torch.float)

        if net == 'main':

            retval = self.main_net(in_features).type(torch.float)
            return retval

        elif net == 'target':

            retval = self.target_net(in_features).type(torch.float)
            return retval

    def q_action(self, state, min_p, max_p):
        '''
        Chooses the best action by argmax_x Q(s,x)
        '''
        features = []
        with torch.no_grad():

            for i in range(int(state[0] + 1)):

                q, t, p, x = Ambiente().normalise(state[0], state[1], state[2], i, min_p, max_p) 
                features.append(torch.tensor([q, t, p, x], dtype=torch.float))

            qs_value = self.main_net.forward(torch.stack(features))
            action = torch.argmax(qs_value).item()

            return round(action)

    def action(self, state, min_p, max_p):
        '''
        does the exploration in the action space
        '''
        if state[0] <= 0:
            action = 0

        elif np.random.rand() <= self.epsilon and state[1] < (PASSI - 1):#
            n = state[0]
            p = 1 / (self.time_subdivisions - state[1])
            action = np.random.binomial(n, p)
            #action = round(np.linspace(0, self.inventory, self.inventory)[action])        
            
        elif state[1] >= (PASSI - 1) and state[0] >= 0:
            action = state[0]

        elif state[1] >= (PASSI - 1) and state[0] <= 0:
            action = 0

        else:

            action = self.q_action(state, min_p, max_p)

        return action

    def reward(self, inv, x, data, alpha):
        '''
        calculates the reward of going slice and dice between the intervals with the quantity chosen to be traded within the intervals
        '''
        reward = -20 + data[0] * x - alpha * (x ** 2)

        return reward
    
    def train_1(self, transitions, data, p_min, p_max):
        '''
        performs the training of the Q-Function approximated as a NN, manages the corner cases of interval = 4 or = 5
        '''
        state       = [tup[:3] for tup in transitions ]
        act         = [tup[3] for tup in transitions  ]
        next_state  = [tup[4:7] for tup in transitions]
        reward      = [tup[7] for tup in transitions  ]

        current_Q = self.eval_Q(state, act, p_min, p_max,'tensor', 'main')
        target = []

        self.matrix     [state[:][1][0], state[:][1][1]] +=  1

        for (s, next_s, rew, ac) in zip(state, next_state, reward, act):

            if next_s[1] >= (PASSI - 1):
                target_value = rew
                target.append(target_value)

            else :
                best_future_action = self.q_action(next_s, p_min, p_max)
                target_value = rew + self.gamma * torch.max(self.eval_Q(next_s, best_future_action, p_min, p_max, 'scalar', 'target'))
                target.append(target_value)

        total_norm = 0
        for p in self.main_net.parameters():

            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2

            grad_norm = total_norm ** 0.5

        target = torch.tensor(target, dtype=torch.float32).reshape(-1,1)
        loss = F.mse_loss(target, current_Q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.main_net.parameters(), 1)
        self.optimizer.step()

        if self.timestep % self.update_target_steps == 0:
            self._update_target_net()
            self.epsilon = self.epsilon * self.epsilon_decay
        
        return loss.cpu().item(), grad_norm, self.matrix, self.epsilon

    def test(self, inv, tempo, dati, p_min, p_max, alpha):
        '''
        does the testing on new data using the weights of the already trained NN
        '''
        state = [inv, tempo, dati[0]]

        if tempo == PASSI:
            x = inv
        else:
            x = self.action(state, p_min, p_max)

        reward = self.reward(inv, x, dati, alpha)

        new_inv = inv - x

        return (new_inv, x, reward)

    def step(self, inv, tempo, data, p_min, p_max, alpha):
        '''
        function that manages the states, performs the action and calculates the reward, in addition it fills up the replay buffer 
        and halves it when it fills up
        '''

        self.timestep += 1
        state = [inv, tempo, data[0]]
        x = self.action(state, p_min, p_max)
        r = self.reward(inv, x, data, alpha)
        new_inv = inv - x
        next_state = [new_inv, tempo + 1, data[-1]]
        self.memory.add(state[0], state[1], state[2], x, next_state[0], next_state[1], next_state[2],  r)

        if len(self.memory) == self.maxlen:

            self.memory.halve()

        if len(self.memory) < self.batch_size:

            return 1, 0, np.zeros((21,PASSI)), 1, new_inv, x, r

        else:

            transitions = self.memory.sample(self.batch_size)

        return *self.train_1(transitions, data, p_min, p_max), new_inv, x, r

if __name__ == '__main__': 

    def sliceData(price, slici):

        step = int(len(price)/slici)
        y = np.zeros((slici,step))

        for i, ii in zip(range(slici), range(step, len(price), step)):
            it = step * i
            y[i, :] = price[it:ii]

        return y

    def doAve(a):
        aa = np.asarray(a)
        ai = aa.reshape(-1, PASSI)
        mean = ai.mean(axis = 0)
        std = np.empty(PASSI)
        for i in range(ai.shape[1]):
            std[i] = stdev(np.double(ai[:,i]))
            
        return np.double(mean), np.double(std)

    def doTrain(age, numIt = 200):

        act_hist = []
        loss_hist = []
        rew_hist = []
        grad_hist = []

        for j in tqdm(range(numIt)):
            slices = PASSI
            inv = INV
            tempo = 0
            s_0 = 10#np.random.uniform(9.9, 10.1)#
            action = 0
            a_hist = []
            k_hist = []
            kappa, alpha = Ambiente(T=PASSI).stoch_imp(numIt = 1, k_0 = PERM_IMP , a_0 = TEMP_IMP)

            for i in tqdm(range(slices)):
                
                k = kappa[i][0]
                a = alpha[i][0]

                dati = Ambiente(T = 1, sigma = VOLA, S0 = s_0, action = action, kappa = k).abm(numIt = 1).flatten()
                p_min, p_max = P_MIN, P_MAX #dati.min(), dati.max() # 
                loss, grad, state , epsilon, new_inv, action, reward = age.step(inv, tempo, dati, p_min, p_max,a) 
                inv = new_inv 
                s_0 = dati[-1]
                ke0 = kappa[-1][-1]
                ey0 = alpha[-1][-1]
                tempo += 1
                rew_hist.append(reward)
                act_hist.append(action)
                grad_hist.append(grad)
                loss_hist.append(loss)
                a_hist.append(a)
                k_hist.append(k)
                

        act_mean, act_sd = doAve(act_hist)
        loss_mean, loss_sd = doAve(loss_hist)
        rew_mean, rew_sd = doAve(rew_hist)

        np.save('./Desktop/ennesima/loss', np.asarray(loss_hist)) 

        return (act_mean, act_sd, act_hist, loss_mean, loss_sd, rew_mean, rew_sd, loss_hist, rew_hist, state, epsilon)

    def doTest(age, numIt = 100):

        act = []
        re = []
        transaction_cost_balance = []
        std_list = []
        mean_list = []
        states = []
        dat = []
        a_hist = []
        k_hist = []

        for j in tqdm(range(numIt)):
            slices = PASSI
            inv = INV
            tempo = 0
            s_0 = 10#np.random.uniform(9.9, 10.1)#
            x = 0
            xx = np.array([2.        , 2.06299434, 1.8064479 , 1.7419047 , 1.80245613,
       1.93107335, 2.06800596, 2.11708072, 2.19547761, 2.25838744])
            kappa, alpha = Ambiente(T=PASSI).stoch_imp(numIt = 1, k_0 = PERM_IMP , a_0 = TEMP_IMP)
            for i in tqdm(range(slices)):

                k = kappa[i][0]
                a = alpha[i][0]

                dati = Ambiente(T = 1, sigma = VOLA, S0 = s_0, action = xx[i], kappa = k).abm(numIt = 1).flatten()

                p_min, p_max =  P_MIN, P_MAX #dati.min(), dati.max() #
                selling_strategy = []
                (new_inv, x, reward) = age.test(inv, tempo, dati, p_min, p_max, a)
                states.append([inv, tempo, s_0, x])
                tempo += 1
                s_0 = dati[-1]
                a_hist.append(a)
                k_hist.append(k)
                inv = new_inv
                re.append(reward)
                act.append(x)
                dat.append(dati)
                selling_strategy.append(x)

        np.save('./Desktop/ennesima/dati', np.asarray(dat)) 
        np.save('./Desktop/ennesima/kappa', np.asarray(k_hist)) 
        np.save('./Desktop/ennesima/alpha', np.asarray(a_hist)) 

        return mean_list, std_list, act, *doAve(act), *doAve(re), re, transaction_cost_balance, states, dat , a_hist,  k_hist
    
    def impl_IS(a_hist, k_hist):
        azioni = np.load('C:/Users/macri/Desktop/ennesima/azioni.npy')
        dati =   np.load('C:/Users/macri/Desktop/ennesima/dati.npy')
        alpha =  np.asarray(a_hist)     
        a = []
        for i in range(dati[:,0].reshape(-1,PASSI).shape[0]):
            a.append(sum(dati[:,0].reshape(-1,PASSI)[i]* azioni.reshape(-1,PASSI)[i] - alpha.reshape(-1,PASSI)[i]* azioni.reshape(-1,PASSI)[i]**2))

        return 200-np.asarray(a)

    def impl_IS_twap(a_hist, k_hist):
        alpha =  np.asarray(a_hist)         
        if PERM_IMP == 0.001:
            dati =   np.load('C:/Users/macri/Desktop/ennesima/dati_TWAP_stoch.npy')
        else:
            dati =   np.load('C:/Users/macri/Desktop/ennesima/dati_TWAP_stoch.npy')

        azioni_tw = np.ones((5000,10)) *2
        a = []
        for i in range(dati[:,0].reshape(-1,10).shape[0]):
            00
            
        return 200-np.asarray(a)

    def run(n, test = False):
        numIt = n
        numTr = 5_000#int(numIt * 0.5)
        age = Agente(inventario = INV, numTrain = numIt)
        
        act_mean, act_sd, act_hist, loss_mean, loss_sd, rew_mean, rew_sd, loss_hist, rew_hist, state, epsilon = doTrain(age, numIt)

        if test == True:
            pel, sdPeL, azioni, azioni_med, sdaz, ricompensa, sdRic, re, tr, states, dat, a_hist, k_hist  = doTest(age, numTr) 
            np.save('./Desktop/ennesima/azioni', azioni) 
            np.save('./Desktop/ennesima/azioniTrain', act_hist) 
            np.save('./Desktop/ennesima/stati', states) 
            np.save('./Desktop/ennesima/stati_train', state)
            np.save('./Desktop/ennesima/trans', tr)

            np.save('./Desktop/ennesima/re', re)

            ql_IS =      impl_IS(a_hist, numTr)
#            ql_TW = impl_IS_twap(a_hist, numTr)

            np.save('./Desktop/ennesima/ql', ql_IS)
            #np.save('./Desktop/ennesima/ac', ql_TW)


            print('med_act  =' , azioni[-PASSI:] ,', act sd = ', sdaz ,
            '\n', ', rew ave =', ricompensa, ', rew sd =', sdRic,
            '\n', ', rew_train =', rew_mean, ', rew sd_train =', rew_sd,
            '\n',', average action chosen from train =', act_mean, 
            '\n', ', ave act test =', azioni_med, ', sd test act =', sdaz,
            '\n', ', IS_QL =',ql_IS.mean(),
            #'\n', ', IS_AC =',ql_TW.mean(),
            '\n', ', epsilon', epsilon,
            '\n', ', TIPO = ', 'Q,T,P stoch',
            '\n', "--- %s minutes ---" % ((time.time() - start_time)    /   60))

            sns.heatmap(np.asarray(state), cmap="YlGnBu" )
            plt.title('states explored in train')
            plt.savefig('./Desktop/ennesima/statiTrain')
            plt.close() # train state

            ranger = np.arange(0,PASSI)
            plt.bar(ranger, azioni_med)
            plt.title('average action test')
            plt.savefig('./Desktop/ennesima/avgActTest')
            plt.close()

            plt.plot(np.asarray(re))
            plt.ylabel('rewards')
            plt.xlabel('iterations')
            plt.savefig('./Desktop/ennesima/rewards')
            plt.close()
            ql_TW=0
            return azioni, azioni_med, sdaz, ricompensa, sdRic, re, tr, states, ql_IS, ql_TW, dat
        
        if test == False:
            print('average action chosen from train =', act_mean, ',actions train sd = ', act_sd ,
            '\n',', reward train =' ,rew_mean ,',reward sd = ', rew_sd ,
            '\n', ',average loss from NN =', loss_mean, ',loss sd =', loss_sd,
            '\n', "--- %s minutes ---" % ((time.time() - start_time)/60))#

            plt.plot(loss_hist)
            plt.xlabel('iterations')
            plt.ylabel('loss')
            plt.show()
            plt.plot(rew_hist)
            plt.xlabel('iterations')
            plt.ylabel('rewards')
            plt.show()
            plt.plot(np.asarray(act_mean))
            plt.title('average action train')
            plt.show()
            sns.heatmap(np.asarray(state))
            plt.show()
    
    start_time = time.time()
    azioni_tot = []
    azioni_med = [] 
    sdaz = []
    ricompensa= [] 
    sdRic = []
    re = []
    dati_tot = []
    stati_tot = []
    ql_tot = []
    ac_tot = []

    for _ in range(1):

        azioni, azioni_med, sdaz, ricompensa, sdRic, re, tr, states, ql, ac, dati = run(n = NUMIT, test = True)
        azioni_tot.append(azioni_med)
        ql_tot.append(ql) 
        ac_tot.append(ac) 
        dati_tot.append(dati)

    np.save('./Desktop/ennesima/ql_tot', ql_tot)
    np.save('./Desktop/ennesima/ac_tot', ac_tot)
    np.save('./Desktop/ennesima/azioni_tot', azioni_tot)
    np.save('./Desktop/ennesima/stati_tot', states)
    np.save('./Desktop/ennesima/dati_tot', dati_tot)
