# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# Creating the architecture of the Neural Network

class Network(nn.Module):

    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        
        """
        fc = full connections (all the neurons on 1 layer will be 
        connected to all the second layer)
        """
        nb_hiddenLayer = 30
        self.fc1 = nn.Linear(input_size, nb_hiddenLayer)
        # will connect the hidden layer w/ the output layer
        self.fc2 = nn.Linear(nb_hiddenLayer, nb_action)
    
    # Forward propagation function
    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values

# Implementing Experience Replay

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        # returns the batches (action, reward, state)
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

# Implementing Deep Q Learning

class Dqn():

    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.learning_rate = 0.001
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate)
        
        '''
         Transition states (last_state, last_action, last_reward)
         last_state = vector of 5 dimensions(3 signals[straight, left, right],
                        orientation, -orientation ), which is the input_size
         in pytorch it needs to be a Tensor rather than a vector
         .unsqueeze(0) is creating a fake dimension corresponding to the batch
        '''
        
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        # action => 0 = go straight, 1 = go right, 2 = go left
        self.last_action = 0
        self.last_reward = 0
    
    '''
        Select Action function
        Here we need to find out the best action to play in order to get the 
        highest score while still exploring other actions, so we'll use SoftMax
        So we need to generate a distribution of probabilities for each of the
        Q-values. Since we have 3 possible actions, we'll have 3 Q-values. 
        The sum of these 3 Q-values is equal to 1 (100%).
        
        SoftMax will atribute the large probability to the highest Q-value
        
        We can configure the temperature parameter to set how will the algorithm
        explore the possible actions.
    '''
    def select_action(self, state):
        temperature_param = 100
        probs = F.softmax(self.model(Variable(state, volatile = True))*temperature_param)
        
        # Now we take a random draw of the distribution
        action = probs.multinomial()
        return action.data[0,0]
    
    '''
        The Learn function to train the Neural Network
        This method is the transition of the Markov decision process that is the base of 
        Deep Q-Learning
        This method will get these batches from the ReplayMemory
        We need the batch_state and the batch_next_state to compute the loss function
    '''
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        '''
         We want to get the neuron output of the input state

         self.model(batch_state) returns the output of all the possible actions so
         we need to get only the one that was decided by the network to play, 
         so we use the gather function
        
         outputs = what our neural network predicts
        '''
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        
        # target is our goal, we want that outputs be equals to target
        target = self.gamma*next_outputs + batch_reward
        
        '''
         computing the loss (error of the prediction), calculates the difference between
         our prediction and the target
         
         td = temporal difference
         smooth_l1_loss is one of the best loss function in deep learning
        '''
        td_loss = F.smooth_l1_loss(outputs, target)
        
        # Back propagation to calculate the stochastic gradient descent updating the weights
        self.optimizer.zero_grad()
        td_loss.backward(retain_variables = True)
        self.optimizer.step()
    
    """"
    Update function - updates all the elements in the transisition as soon as the AI
    reaches a new state (action, state and reward)
    
    This method is called when the car moves (enter in sand, change direction, hits the target)
    and updates the state

    This method makes the connection between the AI and the environment
    """  
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, 
                          new_state, 
                          torch.LongTensor([int(self.last_action)]), 
                          torch.Tensor([self.last_reward])))
        
        # Performs the action (take  random samples on memory)
        action = self.select_action(new_state)
        
        # Now we need to train and AI learn from the information 
        # contained in this samples of 100 random transitions
        nb_transitions = 100
        if len(self.memory.memory) > nb_transitions:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(nb_transitions)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
         # Reward_window has a fixed size
        reward_window_size = 1000
        if len(self.reward_window) > reward_window_size:
            del self.reward_window[0]
        return action
    
    # Calculates the mean of our rewards
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)
    
    # Saving our neural network and the optimizer into a file to be able to use later
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
    
    def load(self):
        # look for the file
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            # We update our existing model/optimizer to the file that is being loaded
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")
