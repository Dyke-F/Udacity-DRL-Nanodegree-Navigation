import torch
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import random

from qnetwork import QNetwork
from replaybuffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
GAMMA = 0.995
TAU = 1e-3
LR = 5e-4
UPDATE_EVERY = 4
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64


class DQN_Agent():
    """ Interacts an learns from the environment. """
    
    def __init__(self, state_size, action_size, seed, GAMMA=GAMMA, TAU=TAU, LR=LR, UPDATE_EVERY=UPDATE_EVERY, 
                 BUFFER_SIZE=BUFFER_SIZE, BATCH_SIZE=BATCH_SIZE):
        
        """ Initialize the agent.
        ==========
        PARAMETERS 
        ==========
            state_size (int) = observation dimension of the environment
            action_size (int) = dimension of each action
            seed (int) = random seed
        """
        
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        self.gamma = GAMMA
        self.tau = TAU
        self.lr = LR
        self.update_every = UPDATE_EVERY
        self.buffer_size = BUFFER_SIZE
        self.batch_size = BATCH_SIZE
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
      
        # instantiate online local and target network for weight updates
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr = self.lr)
        # create a replay buffer
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, seed, self.device)
        # time steps for updating target network every time t_step % 4 == 0
        self.t_step = 0


    def step(self, state, action, reward, next_state, done):
        ''' Append a SARS sequence to memory, then every update_every steps learn from experiences'''
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step +1) % self.update_every
        if self.t_step == 0:
            # in case enough samples are available in internal memory, sample and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)
            
    def act(self, state, eps=0.):
        """ Choose action from an epsilon-greedy policy
        ==========
        PARAMETERS
        ==========
            state (array) = current state space
            eps (float) = epsilon, for epsilon-greedy action choice """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local.forward(state)
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    
    
    def learn(self, experiences, gamma):
        """ Update the value parameters using experience tuples sampled from ReplayBuffer
        ==========
        PARAMETERS
        ==========
          experiences = Tuple of torch.Variable: SARS', done
          gamma (float) = discount factor to weight rewards
        """

        states, actions, rewards, next_states, dones = experiences
        
        # calculate max predicted Q values for the next states using target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1-dones))
        # calculate expected Q vaues from the local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        # compute MSE Loss
        loss = F.mse_loss(Q_expected, Q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
    
    
    def soft_update(self, local_model, target_model, tau):
        """ Soft update for model parameters, every update steps as defined above
        theta_target = tau * theta_local + (1-tau)*theta_target 

        ==========
        PARAMETERS 
        ==========
          local_model, target_model = PyTorch Models, weights will be copied from-to
          tau = interpolation parameter, type=float 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)