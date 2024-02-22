import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import random
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from evaluate import evaluate_HIV, evaluate_HIV_population

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)


def greedy_action(network, state):
    device = "cuda" if next(network.parameters()).is_cuda else "cpu"
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()
   
   
device = "cuda" if torch.cuda.is_available() else "cpu"

# Declare network
state_dim = env.observation_space.shape[0]
n_action = env.action_space.n 
nb_neurons= 512
DQN = torch.nn.Sequential(nn.Linear(state_dim, nb_neurons),
                          nn.ReLU(),
                          nn.Linear(nb_neurons, nb_neurons),
                          nn.ReLU(), 
                          nn.Linear(nb_neurons, nb_neurons),
                          nn.ReLU(), 
                          nn.Linear(nb_neurons, n_action)).to(device)

# DQN config
config = {'nb_actions': env.action_space.n,
          'learning_rate': 0.001,
          'gamma': 0.95,
          'buffer_size': 10000,
          'epsilon_min': 0.01,
          'epsilon_max': 1.,
          'epsilon_decay_period': 20000,
          'epsilon_delay_decay': 500,
          'batch_size': 200,
          'gradient_steps': 4,
          'update_target_strategy': 'replace', # or 'ema'
          'update_target_freq': 500,
          'update_target_tau': 0.005,
          'criterion': torch.nn.SmoothL1Loss()} 

# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.nb_actions = config['nb_actions']
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.95
        self.batch_size = config['batch_size'] if 'batch_size' in config.keys() else 100
        buffer_size = config['buffer_size'] if 'buffer_size' in config.keys() else int(1e5)
        self.memory = ReplayBuffer(buffer_size,device)
        self.epsilon_max = config['epsilon_max'] if 'epsilon_max' in config.keys() else 1.
        self.epsilon_min = config['epsilon_min'] if 'epsilon_min' in config.keys() else 0.01
        self.epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
        self.epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.model = DQN 
        self.target_model = deepcopy(self.model).to(device)
        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
        self.update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
        self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005
    
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
            
    def train(self, env, max_episode):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        best_score = 0
        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action(self.model, state)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(self.nb_gradient_steps): 
                self.gradient_step()
            # update target network if needed
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0: 
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
                self.target_model.load_state_dict(target_state_dict)
            # next transition
            step += 1
            if done or trunc:
                episode += 1
                score = evaluate_HIV(agent=self, nb_episode=1)
                print("Episode ", '{:3d}'.format(episode), 
                      ", epsilon ", '{:6.2f}'.format(epsilon), 
                      ", batch size ", '{:5d}'.format(len(self.memory)), 
                      ", episode return ", '{:.4e}'.format(episode_cum_reward),
                      ", score ", '{:.4e}'.format(score),
                      sep='')
                if score > best_score:
                    best_score = score
                    self.save("DQN_new.pt")
                    print("Model saved")
                state, _ = env.reset()
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
            else:
                state = next_state
        return episode_return
              
    
    def act(self, observation, use_random=False):
        if use_random:
            return env.action_space.sample()
        else:
            a = greedy_action(self.model, observation)
            return a

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self):
        self.model.load_state_dict(torch.load("DQN_new.pt", map_location=torch.device('cpu')))


# agent = ProjectAgent()
# agent.load()
# # evaluate the agent
# print(evaluate_HIV(agent=agent, nb_episode=1))


### OLD CODE ####
############################################
############################################

# from gymnasium.wrappers import TimeLimit
# from env_hiv import HIVPatient
# import numpy as np
# from sklearn.ensemble import ExtraTreesRegressor
# from tqdm import tqdm
# from evaluate import evaluate_HIV, evaluate_HIV_population
# import pickle
# import random

# env = TimeLimit(
#     env=HIVPatient(domain_randomization=False), max_episode_steps=200
# )  # The time wrapper limits the number of steps in an episode at 200.
# # Now is the floor is yours to implement the agent and train it.


# # You have to implement your own agent.
# # Don't modify the methods names and signatures, but you can add methods.
# # ENJOY!

# # we will de a Fitted Q-Iteration agent
# class ProjectAgent:
#     def __init__(self): 
#         self.Q = None
        
#     def act(self, observation, use_random=False):
#         if use_random:
#             return env.action_space.sample()
#         else:
#             s = observation
#             a = self.greedy_action(self.Q,s,env.action_space.n)
#             return a

#     def save(self, path):
#         pickle.dump(self.Q, open(path, "wb"))

#     def load(self):
#         # self.Qfunctions = np.load("Qfunctions.npy", allow_pickle=True)
#         self.Q = pickle.load(open("Q4.pkl", "rb"))
        
    
#     def greedy_action(self,Q,s,nb_actions):
#         Qsa = []
#         for a in range(nb_actions):
#             sa = np.append(s,a).reshape(1, -1)
#             Qsa.append(Q.predict(sa))
#         return np.argmax(Qsa)
    
#     def rf_fqi(self, S, A, R, S2, D, iterations, nb_actions, gamma, disable_tqdm=False):
#         nb_samples = S.shape[0]
#         SA = np.append(S,A,axis=1)
#         for iter in tqdm(range(iterations), disable=disable_tqdm):
#             if iter==0:
#                 value=R.copy()
#             else:
#                 Q2 = np.zeros((nb_samples,nb_actions))
#                 for a2 in range(nb_actions):
#                     A2 = a2*np.ones((S.shape[0],1))
#                     S2A2 = np.append(S2,A2,axis=1)
#                     Q2[:,a2] = Q.predict(S2A2)
#                 max_Q2 = np.max(Q2,axis=1)
#                 value = R + gamma*(1-D)*max_Q2
#             # Q = ExtraTreesRegressor()
#             Q = ExtraTreesRegressor(n_estimators=200)
#             Q.fit(SA,value)
#         return Q
    
#     def collect_samples(self, env, horizon, disable_tqdm=False, print_done_states=False, use_random=True):
#         s, _ = env.reset()
#         #dataset = []
#         S = []
#         A = []
#         R = []
#         S2 = []
#         D = []
#         for _ in tqdm(range(horizon), disable=disable_tqdm):
#             # a = env.action_space.sample()
#             if random.random() < 0.5:
#                 a = env.action_space.sample()
#             else:
#                 a = self.act(s, use_random=use_random)
#             s2, r, done, trunc, _ = env.step(a)
#             #dataset.append((s,a,r,s2,done,trunc))
#             S.append(s)
#             A.append(a)
#             R.append(r)
#             S2.append(s2)
#             D.append(done)
#             if done or trunc:
#                 s, _ = env.reset()
#                 if done and print_done_states:
#                     print("done!")
#             else:
#                 s = s2
#         S = np.array(S)
#         A = np.array(A).reshape((-1,1))
#         R = np.array(R)
#         S2= np.array(S2)
#         D = np.array(D)
#         return S, A, R, S2, D
    
#     def train(self):
#         # Fitted Q-Iteration
#         horizon = 6000
#         S, A, R, S2, D = self.collect_samples(env, horizon)
#         nb_actions = env.action_space.n
#         gamma = 0.99
#         iterations = 1000
#         self.Q = self.rf_fqi(S, A, R, S2, D, iterations, nb_actions, gamma)
#         print(evaluate_HIV(agent=self, nb_episode=1))
#         # for _ in range(20):
#         #     S, A, R, S2, D = self.collect_samples(env, horizon, use_random=False)
#         #     self.Q = self.rf_fqi(S, A, R, S2, D, iterations, nb_actions, gamma)
#         #     print(evaluate_HIV(agent=self, nb_episode=1))
        
        
# agent = ProjectAgent()
# agent.train()
# # evaluate the agent
# print(evaluate_HIV(agent=agent, nb_episode=1))
# agent.save("Q4.pkl")

# # agent2 = ProjectAgent()
# # agent2.load()

