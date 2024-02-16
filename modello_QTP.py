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
'''constrained '''
in_features = 4

PERM_IMP = 0.001 # entra negativo nel ABM però ricordati #0.0001
TEMP_IMP = 0.002
VOLA = 0.0001#0.0001# era 0.01
MU = 0#0.0002
LAMBDA = 1 # non in uso
TIME_VAR = False
IMP_DECR = False
INV = 20 # inventario
PASSI = 10 #discretizzazione
STEP = 100#00 # abbassa la epsilon ogni 150 azioni compiute
NUMIT = 10_000 # traiettorie di train, quelle di test sono sempre la meta', il numero totale delle azioni compiute in train sarà NUMIT x PASSI
MIN_P = 9.94  
MAX_P = 10.06

class DQN(nn.Module):
    '''
    Fully connected NN with 30 nodes in each layer, for 5 layers using sequential layers
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
        #self.numIt = numIt
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
        #np.random.seed(seed)
        k_0 = 1e-6#0.009
        for i in range(N):
            #X[i + 1] = X[i] + (self.mu - ((1e-6 + self.kappa) * self.action)) * dt + self.sigma * np.sqrt(dt) * np.random.standard_normal(I) 
            #X[i+1] = X[i+1] + TEMP_IMP * self.action/N
            if TIME_VAR == True:
                if IMP_DECR == True:
                    imp = 0.009 - (self.kappa * dt * i) * (self.action)
                else:
                    imp = 1e-6 + (self.kappa * dt * i) * (self.action)
                if imp < 0: imp = 1e-6
                X[i + 1] = X[i] + (self.mu - imp) * dt + self.sigma * np.sqrt(dt) * np.random.standard_normal(I) # impatto sale 
            else:
                X[i + 1] = X[i] + (self.mu - ((1e-6 + self.kappa) * self.action)) * dt + self.sigma * np.sqrt(dt) * np.random.standard_normal(I) 
                #X[i + 1] = X[i + 1] - (0.001* self.action)/N 

            #era  (self.mu - (self.kappa * i) * (self.action/N)**self.lambd)
            #X[i + 1] = X[i] + (self.mu - self.kappa  * np.sign(self.action) * np.abs(self.action/N) ** self.lambd) * dt + self.sigma * np.sqrt(dt) * np.random.standard_normal(I) 

        return np.abs(X)
    
    def updn(self, seed = 14, numIt=10):
        '''
        returns a matrix of Arithmetic Brownian Motion paths
        '''
        N = 3600
        I = numIt
        dt= 1.0 / 3600
        X =np.zeros((N + 1, I), dtype=float)
        X[0] = 10
        #np.random.seed(seed)
        step = 5
        for i in range(N):

            if (i < N/step) or (N/step*2 < i < N/step*3) or (N/step*4 < i < N/step*5) :
                sgn = -self.mu
            elif (N/step < i < N/step*2) or  (N/step*3 < i < N/step*4):
                sgn = self.mu

            X[i + 1] = X[i] + sgn * dt + self.sigma * np.sqrt(dt) * np.random.standard_normal(I) 

        return X
    
    def abmUD(self, seed = 14, numIt = 1):

        X = np.zeros((3600 + 1, numIt), dtype=float)
        #np.random.seed(seed)

        def up():

            N = 3600
            I = 1
            dt= 1.0 / 3600
            X = np.zeros((N + 1, I), dtype=float)
            X[0] = 10

            for i in range(N):
                X[i + 1] = X[i] - self.mu * dt + self.sigma * np.sqrt(dt) * np.random.standard_normal(I)
            return X

        def dn():

            N = 3600
            I = 1
            dt= 1.0 / 3600
            X = np.zeros((N + 1, I), dtype=float)
            X[0] = 10
            for i in range(N):
                X[i + 1] = X[i] + self.mu * dt + self.sigma * np.sqrt(dt) * np.random.standard_normal(I)
            return X

        for i in range(numIt):

            #c = np.random.binomial(1, 0.5)
            if i%2 == 0:#c == 1:

                X[:,i] = up().reshape(-1)
            else :

                X[:,i] = dn().reshape(-1)

        return X

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

    def qdr_var_normalize(self, qdr_var, min_v, max_v):

        middle_point = (max_v + min_v) / 2
        half_length = (max_v - min_v) / 2

        qdr_var = (qdr_var - middle_point) / half_length

        return qdr_var

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
        #middle_point = (max_p + min_p) / 2
        #half_length = (max_p - min_p) / 2
#
        #price = (price - middle_point) / half_length
#
        #return price

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

    def add(self, inv, time, p, x, next_inv, next_time, next_price, reward): #inv, time, price, var, x, next_state, reward state, action, next_state, reward
        
        self.memory.append([inv, time, p, x, next_inv, next_time, next_price, reward])

    def sample(self, batch_size):
        
        return rnd.sample(self.memory, batch_size)
    
    def halve(self):

        for i in range(round(self.__len__()/2)):
            del self.memory[i]

        #self.memory.remove()

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

        #self.initial_capital = inventory

        self._update_target_net()

        self.learning_rate = 0.0001
        self.optimizer = optim.Adam(params=self.main_net.parameters(), lr=self.learning_rate)
        self.time_subdivisions = PASSI
        self.inventory = inventario
        self.a_penalty = TEMP_IMP#0.001#0.001
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.batch_size = 32#128/2#*3
        self.gamma = 1#0.99#1#e-10#0.9   ##############################################
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
        eps >= U(0,1) then tosses a coin -> 50%prob does TWAP, 50%prob does all in
        eps <= U(0,1) does the optimal Q action
        '''
        # azione da eseguire: estrae un numero a caso: se questo è minore di epsilon allora fa azione casuale x=(0,q_t), altrimenti fa argmax_a(Q(s,a))
        if state[0] <= 0:
            action = 0

        elif np.random.rand() <= self.epsilon and state[1] < (PASSI - 1):#
            n = state[0]
            p = 1 / (self.time_subdivisions - state[1])
            action = np.random.binomial(n, p)
            #action = round(np.linspace(0, self.inventory+1, self.inventory)[action])           
            
        elif state[1] >= (PASSI - 1) and state[0] >= 0:
            action = state[0]

        elif state[1] >= (PASSI - 1) and state[0] <= 0:
            action = 0

        else:

            action = self.q_action(state, min_p, max_p)

        return action

    def reward(self, inv, x, data):
        '''
        calculates the reward of going slice and dice between the intervals with the quantity chosen to be traded within the intervals
       
        reward = 0
        inventory_left = inv
        M = len(data)
        xs = x/M
        for i in range(1, M):

            reward += inventory_left * (data[i] - data[i - 1]) - self.a_penalty * (xs ** 2) #- PERM_IMP/2 * (inventory_left/M) ** 2

            inventory_left -= xs

        return reward
        '''
        reward = 0
        inventory_left = inv
        M = len(data)
        xs = x/M
        for i in range(1, M):

            reward = -20 +  data[0] * x - self.a_penalty * (x ** 2)#inventory_left * (data[i] - data[i - 1]) - self.a_penalty * (xs ** 2)# - PERM_IMP/2 * (inventory_left/M) ** 2 (data[i] - data[i - 1])

            inventory_left -= xs

        return reward
    
    def train_1(self, transitions, data, p_min, p_max):
        '''
        performs the training of the Q-Function approximated as a NN, manages the corner cases of interval = 4 or = 5
        '''
        #PROBLEMA E' QUI NEL SAMPLING E NELL'USO DI QUESTI SAMPLING

        state       = [tup[:3] for tup in transitions ]
        act         = [tup[3] for tup in transitions  ]
        next_state  = [tup[4:7] for tup in transitions]
        reward      = [tup[7] for tup in transitions  ]

        current_Q = self.eval_Q(state, act, p_min, p_max,'tensor', 'main')
        target = []

        self.matrix     [state[:][1][0], state[:][1][1]] +=  1

        for (s, next_s, rew, ac) in zip(state, next_state, reward, act):

            # qui aumenta il counter per gli states quando vengono sampled 

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
        
        torch.save(self.main_net.state_dict(), 'model.pth')

        return loss.cpu().item(), grad_norm, self.matrix, self.epsilon


    def test(self, inv, tempo, dati, p_min, p_max):
        '''
        does the testing on new data using the weights of the already trained NN
        '''
        model = self.main_net
        #model.load_state_dict(torch.load('model.pth'))
        #torch.load_state()
        state = [inv, tempo, dati[0]]
        #if dati.max() > p_max: p_max = dati.max()
        #if dati.min() < p_min: p_min = dati.min()

        if tempo == PASSI:
            x = inv
        else:
            x = self.action(state, p_min, p_max)

        reward = self.reward(inv, x, dati)

        new_inv = inv - x
        #next_state = [new_inv,    tempo + 1,    dati[-2]]

        return (new_inv, x, reward)

    def PeL_QL(self, strategy, data):
        '''
        calculates the Profit and Loss of the strategy found by DQN
        '''
        PeL = 0
        a = self.a_penalty
        M = len(data)

        for i in range(self.time_subdivisions):

            x = self.lots_size * strategy[0]
            xs = x / M

            for t in range(M):
                if t + 1 < len(data):
                    PeL += xs * data[t] - a * (xs ** 2) #-  PERM_IMP/2 * (self.inventory/M) ** 2 ###############################################################

        return np.double(PeL)

    def PeL_AC_with_drift(self,data):
        
        PeL = 0
        M = len(data)
        penalty = self.a_penalty
        
        T_f = self.time_subdivisions
        q_0=self.inventory

        mu = Ambiente().mu
        gamma = 1e-6
        V = 100
        eta = V * self.a_penalty #/ 10                                                         
        sigma = Ambiente().sigma
                
        b= V * (pow(sigma, 2)) * gamma / (2 * eta)
        a= - (V * mu) / (2 * eta)
        
        x_present=q_0
        
        for i in range(self.time_subdivisions):
            
            next_second = i+1
            x_future =  ((q_0+(a/b))*(m.sinh(m.sqrt(b)*(T_f-next_second)))+(a/b)*m.sinh(m.sqrt(b)*next_second))/m.sinh(m.sqrt(b)*T_f)-(a/b)


            xs=self.lots_size*(x_present-x_future)/M 
                
            for t in range(M):
                if t + 1 < len(data):

                    PeL += xs * data[t] - penalty * (xs ** 2) #- PERM_IMP/2 * (self.inventory/M) ** 2 + (PERM_IMP/2)*(xs ** 2)#xs * data[t] - penalty * (xs ** 2) - PERM_IMP/2 * (self.inventory/M) ** 2#########################################
                    
            x_present = x_future  

        return PeL  

    def PeL_TWAP(self, data):
        '''
        Calculates the Profit and Loss of the TWAP strategy
        '''
        PeL = 0
        M = len(data)
        a = self.a_penalty

        x = self.inventory / self.time_subdivisions * self.lots_size
        xs = x / M

        for i in range(self.time_subdivisions):

            for t in range(M):
                if t+1 < len(data):
                    PeL += xs * data[t] - a * (xs ** 2)
        return PeL

    def step(self, inv, tempo, data, p_min, p_max):
        '''
        function that manages the states, performs the action and calculates the reward, in addition it fills up the replay buffer 
        and halves it when it fills up
        '''
        #iter = 1
        self.timestep += 1
        state = [inv, tempo, data[0]]
        #p_min, p_max = data.min(), data.max()
        #if data.max() > p_max: p_max = data.max()
        #if data.min() < p_min: p_min = data.min()
        x = self.action(state, p_min, p_max)
        r = self.reward(inv, x, data)
        new_inv = inv - x
        next_state = [new_inv, tempo + 1, data[-1]]
        self.memory.add(state[0], state[1], state[2], x, next_state[0], next_state[1], next_state[2],  r)

        if len(self.memory) == self.maxlen:

            self.memory.halve()

        if len(self.memory) < self.batch_size:

            return 1, 0, np.zeros((21,PASSI)), 1, new_inv, x, r

        else:

            transitions = self.memory.sample(self.batch_size)
        
        #salva i pesi qui?
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
        mean = ai.mean(axis = 0)############
        std = np.empty(PASSI)
        for i in range(ai.shape[1]):
            std[i] = stdev(np.double(ai[:,i]))#/(ai.shape[0] - 1)
            
        return np.double(mean), np.double(std)

    def doTrain(age, numIt = 200):

        act_hist = []
        loss_hist = []
        rew_hist = []
        grad_hist = []

        #data = Ambiente().abm(numIt=numIt) #------> questa la faccio da fuori? NO

        for j in tqdm(range(numIt)):
            slices = PASSI
            #ss = sliceData(data[:,j], slices)
            inv = INV
            tempo = 0
            s_0 = 10
            action = 0

            for i in tqdm(range(slices)):

                dati = Ambiente(T = 1, sigma = VOLA, S0 = s_0, action = action).abm(numIt = 1).flatten()#ss[i,:] # considero slice da 5 (720 osservazioni alla volta)
                p_min, p_max = MIN_P , MAX_P#dati.min(),dati.max()#
                loss, grad, state , epsilon, new_inv, action, reward = age.step(inv, tempo, dati, p_min, p_max) 
                inv = new_inv 
                s_0 = dati[-1]
                tempo += 1
                rew_hist.append(reward)
                act_hist.append(action)
                grad_hist.append(grad)
                loss_hist.append(loss)

        act_mean, act_sd = doAve(act_hist)
        loss_mean, loss_sd = doAve(loss_hist)
        rew_mean, rew_sd = doAve(rew_hist)

        return (act_mean, act_sd, act_hist, loss_mean, loss_sd, rew_mean, rew_sd, loss_hist, rew_hist, state, epsilon)

    def doTest(age, numIt = 100):
        #data = Ambiente().abm(seed = 10, numIt=numIt)
        act = []
        re = []
        transaction_cost_balance = []
        std_list = []
        mean_list = []
        states = []
        ql = []
        ac = []
        dat = []
        for j in tqdm(range(numIt)):
            slices = PASSI
            #ss = sliceData(data[:,j], slices)
            inv = INV
            tempo = 0
            s_0 = 10#np.random.uniform(9,11)#######################################################################10#
            x = 0

            for i in tqdm(range(slices)):

                
                dati = Ambiente(T = 1, sigma = VOLA, S0 = s_0, action = x).abm(numIt = 1).flatten()
                p_min, p_max = MIN_P , MAX_P#dati.min(),dati.max()#MIN_P , MAX_P
                selling_strategy = []
                (new_inv, x, reward) = age.test(inv, tempo, dati, p_min, p_max)
                states.append([inv, tempo, s_0, x])
                tempo += 1
                s_0 = dati[-1]
                inv = new_inv
                re.append(reward)
                act.append(x)
                dat.append(dati)
                selling_strategy.append(x)

                #transaction_cost_balance.append((age.PeL_QL(selling_strategy, dati) - age.PeL_TWAP(dati)) / (age.PeL_TWAP(dati)))
                transaction_cost_balance.append((age.PeL_QL(selling_strategy, dati) - age.PeL_AC_with_drift(dati) ) / age.PeL_AC_with_drift(dati))
                ql.append(np.asarray(age.PeL_QL(selling_strategy, dati))/PASSI)
                ac.append(np.asarray(age.PeL_AC_with_drift(dati))/PASSI)
        ql = np.asarray(ql)
        ac = np.asarray(ac)
        mean_list = (((ql.reshape(-1,PASSI).sum(axis = 1) - ac.reshape(-1,PASSI).sum(axis = 1)) / ac.reshape(-1,PASSI).sum(axis = 1))/PASSI).mean()*100#.append(stat.mean(np.asarray(transaction_cost_balance)))
        std_list  = (((ql.reshape(-1,PASSI).sum(axis = 1) - ac.reshape(-1,PASSI).sum(axis = 1)) / ac.reshape(-1,PASSI).sum(axis = 1))/PASSI).std()*100  #.append(stat.stdev(np.asarray(transaction_cost_balance)))
        
        np.save('./Desktop/ennesima/dati', np.asarray(dat)) 

        #performance_list.append(performance(transaction_cost_balance))
        
        return mean_list, std_list, act, *doAve(act), *doAve(re), re, transaction_cost_balance, states, ql, ac# np.asarray(transaction_cost_balance).mean(), np.asarray(transaction_cost_balance).std()transaction_cost_balance#*doAve(mean_list)
    
    def get_heatmap(agent):

        def choose_best_action(q,t):   
            define_state = [q,t]
            return agent.action(define_state, 0, 1)

        array = np.zeros((21, PASSI))

        for q in range(21):
            for t in range(PASSI):
                x = choose_best_action(q,t)
                array[q][t] = x

        return array
    

    def impl_IS():
        azioni = np.load('C:/Users/macri/Desktop/ennesima/azioni.npy')
        dati =   np.load('C:/Users/macri/Desktop/ennesima/dati.npy')
        #azioni.reshape(-1,5),np.mean(sum(azioni.reshape(-1,5)**2))
        a = []
        for i in range(dati[:,0].reshape(-1,PASSI).shape[0]):
            a.append(sum(dati[:,0].reshape(-1,PASSI)[i]* azioni.reshape(-1,PASSI)[i] - TEMP_IMP* azioni.reshape(-1,PASSI)[i]**2))

        return 200-np.asarray(a)#*100

    def impl_IS_twap():
        #azioni = np.load('C:/Users/macri/Desktop/ennesima/azioni.npy')
        if PERM_IMP == 0.001:
            dati =   np.load('C:/Users/macri/Desktop/ennesima/dati_TWAP_10_0.001.npy')
        else:
            dati =   np.load('C:/Users/macri/Desktop/ennesima/dati_TWAP_10_0.003.npy')

        azioni_tw = np.ones((3_000,PASSI)) *2
        #azioni.reshape(-1,5),np.mean(sum(azioni.reshape(-1,5)**2))
        a = []
        for i in range(dati[:,0].reshape(-1,PASSI).shape[0]):
            a.append(sum(dati[:,0].reshape(-1,PASSI)[i]* azioni_tw.reshape(-1,PASSI)[i] - TEMP_IMP* azioni_tw.reshape(-1,PASSI)[i]**2))

        return 200-np.asarray(a)#*100




    def run(n, test = False):
        numIt = n
        numTr = int(numIt * 0.5)
        age = Agente(inventario = INV, numTrain = numIt)

        #for i in range(3):
        
        act_mean, act_sd, act_hist, loss_mean, loss_sd, rew_mean, rew_sd, loss_hist, rew_hist, state, epsilon = doTrain(age, numIt)

        #heat = get_heatmap(age)

        if test == True:
            #pel, sdPeL, azioni, azioni_med, sdaz, ricompensa, sdRic, re, tr = doTest(age, numTr)
            pel, sdPeL, azioni, azioni_med, sdaz, ricompensa, sdRic, re, tr, states, ql, ac  = doTest(age, numTr) #doAve(mean_list), act, doAve(act), doAve(re)
            #return pel, azioni, azioni_med, ricompensa#a, a_var, a_t, b, b_var, c, c_var, pel, azioni, azioni_med, ricompensa, loss_hist, rew_hist
            np.save('./Desktop/ennesima/azioni', azioni) 
            np.save('./Desktop/ennesima/azioniTrain', act_hist) 
            np.save('./Desktop/ennesima/stati', states) 
            np.save('./Desktop/ennesima/stati_train', state)
            np.save('./Desktop/ennesima/trans', tr)

            np.save('./Desktop/ennesima/re', re)
            
            mu = (((ql.reshape(-1,5).sum(axis = 1) - ac.reshape(-1,5).sum(axis = 1)) / ac.reshape(-1,5).sum(axis = 1))).mean()*100 #non in uso
            si = (((ql.reshape(-1,5).sum(axis = 1) - ac.reshape(-1,5).sum(axis = 1)) / ac.reshape(-1,5).sum(axis = 1))).std()


            ql_IS = impl_IS()
            ql_TW = impl_IS_twap()

            #mu = (((ql_IS.reshape(-1,5).sum(axis = 1) - ql_TW.reshape(-1,5).sum(axis = 1)) / ql_TW.reshape(-1,5).sum(axis = 1))).mean()*100
            #si = (((ql_IS.reshape(-1,5).sum(axis = 1) - ql_TW.reshape(-1,5).sum(axis = 1)) / ql_TW.reshape(-1,5).sum(axis = 1))).std()

            np.save('./Desktop/ennesima/ql', ql_IS)
            np.save('./Desktop/ennesima/ac', ql_TW)


            print('average PandL pct. =', mu, ', PandL sd = ', si ,
            '\n',', med_act  =' , azioni[-PASSI:] ,', act sd = ', sdaz ,
            '\n', ', rew ave =', ricompensa, ', rew sd =', sdRic,
            '\n', ', rew_train =', rew_mean, ', rew sd_train =', rew_sd,
            '\n',', average action chosen from train =', act_mean, 
            '\n', ', ave act test =', azioni_med, ', sd test act =', sdaz,
            '\n', ', IS_QL =',ql_IS.mean(),
            '\n', ', IS_AC =',ql_TW.mean(),
            '\n', ', epsilon', epsilon,
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

            plt.hist(   (((ql.reshape(-1,PASSI).sum(axis = 1) - ac.reshape(-1,PASSI).sum(axis = 1)) / ac.reshape(-1,PASSI).sum(axis = 1))))
            plt.axvline((((ql.reshape(-1,PASSI).sum(axis = 1) - ac.reshape(-1,PASSI).sum(axis = 1)) / ac.reshape(-1,PASSI).sum(axis = 1))).mean(), color = 'r')
            plt.title('$\Delta \, \, IS$ of QL and AC, ABM - Q,T,P $\mu=0,\,\, \sigma=0.1$')
            plt.ylabel('iterations')
            plt.xlabel('$\Delta \,\, IS$')
            plt.savefig('./Desktop/ennesima/deltaIS.png')
            plt.close()

        if test == False:
            print('average action chosen from train =', act_mean, ',actions train sd = ', act_sd ,
            '\n',', reward train =' ,rew_mean ,',reward sd = ', rew_sd ,
            '\n', ',average loss from NN =', loss_mean, ',loss sd =', loss_sd,
            '\n', "--- %s minutes ---" % ((time.time() - start_time)/60))#
            #'\n',',PeL performance=', pel,',average actions test=', azioni_med, ',ricompensa = ', ricompensa, 
            #'\n', ',last train=', a_t[-5:], ' last test = ', azioni[-5:])

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
            #sns.heatmap(np.asarray(heat))
            #plt.show()
            sns.heatmap(np.asarray(state))
            plt.show()
            #return act_mean, act_sd, act_hist, loss_mean, loss_sd, rew_mean, rew_sd, loss_hist, rew_hist, heat, state
    
    
    start_time = time.time()
    run(n = NUMIT, test = True)
    #act_mean, act_sd, act_hist, loss_mean, loss_sd, rew_mean, rew_sd, loss_hist, rew_hist, heat, state = run(False)
