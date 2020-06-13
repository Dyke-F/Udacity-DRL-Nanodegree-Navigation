from dqn_agent import DQN_Agent
import random
import torch

import torch
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import random

from qnetwork import QNetwork
from replaybuffer import ReplayBuffer

# Hyperparameters
GAMMA = 0.995
TAU = 1e-3
LR = 5e-4
UPDATE_EVERY = 4
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class DDQN_Agent(DQN_Agent):
    """ Creat a double DQN Agent inherited from DQN_Agent"""
    
    def learn(self, experiences, gamma):
        """ Update the value parameters using experience tuples sampled from ReplayBuffer
        Using Double Deep-Q-Learning should prevent overestimation of Q values.
        ==========
        PARAMETERS
        ==========
          experiences = Tuple of torch.Variable: SARS', done
          gamma (float) = discount factor to weight rewards
        """
        
        states, actions, rewards, next_states, dones = experiences
        
        self.qnetwork_target.eval()
        with torch.no_grad():
            Q_pred_local = self.qnetwork_local.forward(next_states)
            Q_pred_target = self.qnetwork_target.forward(next_states)
            topk_action = torch.max(Q_pred_local, dim=1, keepdim=True)[1]
            Q_pred_max = Q_pred_target.gather(1, topk_action)
            y = rewards + gamma * Q_pred_max * (1-dones)
        self.qnetwork_target.train()
        
        self.optimizer.zero_grad()
        y_pred = self.qnetwork_local(states).gather(1, actions)
        loss = F.mse_loss(y_pred, y)
        loss.backward()
        self.optimizer.step()
        
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)