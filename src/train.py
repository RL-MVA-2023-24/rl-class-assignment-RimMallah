from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
from evaluate import evaluate_HIV, evaluate_HIV_population
import pickle

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

# we will de a Fitted Q-Iteration agent
class ProjectAgent:
    def __init__(self): 
        self.Q = None
        
    def act(self, observation, use_random=False):
        if use_random:
            return env.action_space.sample()
        else:
            s = observation
            a = self.greedy_action(self.Q,s,env.action_space.n)
            return a

    def save(self, path):
        pickle.dump(self.Q, open(path, "wb"))

    def load(self):
        # self.Qfunctions = np.load("Qfunctions.npy", allow_pickle=True)
        self.Q = pickle.load(open("Q.pkl", "rb"))
        
    
    def greedy_action(self,Q,s,nb_actions):
        Qsa = []
        for a in range(nb_actions):
            sa = np.append(s,a).reshape(1, -1)
            Qsa.append(Q.predict(sa))
        return np.argmax(Qsa)
    
    def rf_fqi(self, S, A, R, S2, D, iterations, nb_actions, gamma, disable_tqdm=False):
        nb_samples = S.shape[0]
        Qfunctions = []
        SA = np.append(S,A,axis=1)
        for iter in tqdm(range(iterations), disable=disable_tqdm):
            if iter==0:
                value=R.copy()
            else:
                Q2 = np.zeros((nb_samples,nb_actions))
                for a2 in range(nb_actions):
                    A2 = a2*np.ones((S.shape[0],1))
                    S2A2 = np.append(S2,A2,axis=1)
                    Q2[:,a2] = Qfunctions[-1].predict(S2A2)
                max_Q2 = np.max(Q2,axis=1)
                value = R + gamma*(1-D)*max_Q2
            Q = RandomForestRegressor()
            Q.fit(SA,value)
            Qfunctions.append(Q)
        return Qfunctions[-1]
    
    def collect_samples(self, env, horizon, disable_tqdm=False, print_done_states=False):
        s, _ = env.reset()
        #dataset = []
        S = []
        A = []
        R = []
        S2 = []
        D = []
        for _ in tqdm(range(horizon), disable=disable_tqdm):
            a = env.action_space.sample()
            s2, r, done, trunc, _ = env.step(a)
            #dataset.append((s,a,r,s2,done,trunc))
            S.append(s)
            A.append(a)
            R.append(r)
            S2.append(s2)
            D.append(done)
            if done or trunc:
                s, _ = env.reset()
                if done and print_done_states:
                    print("done!")
            else:
                s = s2
        S = np.array(S)
        A = np.array(A).reshape((-1,1))
        R = np.array(R)
        S2= np.array(S2)
        D = np.array(D)
        return S, A, R, S2, D
    
    def train(self):
        # Fitted Q-Iteration
        horizon = 10000
        S, A, R, S2, D = self.collect_samples(env, horizon)
        nb_actions = env.action_space.n
        gamma = 0.99
        iterations = 200
        self.Q = self.rf_fqi(S, A, R, S2, D, iterations, nb_actions, gamma)
        
        
# agent = ProjectAgent()
# agent.train()
# # evaluate the agent
# print(evaluate_HIV(agent=agent, nb_episode=1))
# agent.save("Q.pkl")

# agent2 = ProjectAgent()
# agent2.load()