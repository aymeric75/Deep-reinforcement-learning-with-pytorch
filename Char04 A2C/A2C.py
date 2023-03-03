import math
import random
random.seed(1)
import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import matplotlib.pyplot as plt
import pandas as pd


use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

from multiprocessing_env import SubprocVecEnv
from environment import GridWorldEnv, GridWorldEnvND

from MooreMachine import MooreMachine
import pygame
from pygame.locals import *

from Classifier import CNN

import seaborn as sns

#torch.autograd.set_detect_anomaly(True)


num_envs = 1
env_name = "CartPole-v0"


transition_function = {0:{0:2, 1:0, 2:0, 3:1, 4:0}, 1:{0:1, 1:1, 2:1, 3:1, 4:1}, 2:{0:2, 1:3, 2:2, 3:1, 4:2}, 3:{0:3, 1:3, 2:4, 3:1, 4:3}, 4:{0:4, 1:4, 2:4, 3:1, 4:4}}
output_function = [3,4,2,1,0]

minecraft_machine = MooreMachine(transition_function, output_function)
env = GridWorldEnv(minecraft_machine, "human", False)

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"




# print(obs)
# print(type(obs))
# print(obs.shape) (512, 512, 3)

termination = False




def make_env():
    def _thunk():
        #env = gym.make(env_name)
        return env
    return _thunk

plt.ion()

envs = [env]

# RNN MODEL 
#
class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Model, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        #self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers)   
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):
        
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        
        # input_size | hidden_size | num_layers

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        #hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        #hidden = torch.zeros(2, batch_size, self.hidden_dim)
        hidden = torch.zeros(1, 1, 50)
        return hidden








# ACTOR CRITIC MODEL
#
class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Softmax(),
            #nn.Softmax(),
        )
        
    def forward(self, x):

        value = self.critic(x)
        
        probs = self.actor(x)

        dist  = Categorical(probs)

        return dist, value


cnn = CNN()

# CONVERT TO THE VECTOR STATE (thanks to the cnn)
def from_obs_to_state(obss):

    obss = obss.reshape((3, 512, 512))
    obss = np.expand_dims(obss, axis=0)
    obss = np.expand_dims(obss, axis=0)
    obss = torch.tensor(obss).type(torch.double).to(torch.float32).to(device)
    
    state = cnn(obss.view(-1, 3, 512, 512))
    return state


# USED very TT episodes (to make the plot)
def test_env(vis=False):


    obs, reward, info, done = env.reset()


    state = from_obs_to_state(obs)

    done = False
    total_reward = 0

    traj_so_far_tmp = []
    
    #h_0 = torch.randn(1*1, 1, 5)
    h_0 = torch.randn(1*1, 1, 50)

    for ii in range(num_steps):
        #print("ii {}".format(ii))
        



        # we reformat the state
        state = torch.squeeze(state)
        state = torch.squeeze(state.view(5,1))
        
        
        # we append the current state to the trajectory vector
        state = torch.argmax(state).item()

        state = torch.tensor(state).to(torch.float)
        state = torch.unsqueeze(torch.unsqueeze(state, dim=0), dim=0)
        
        state = torch.unsqueeze(state, dim=0)
        # we construct the input for the RNN from the trajectory

        state.to(device)

        # 
        output, hn = rnn.rnn(state, h_0)

        #output, hn = rnn(rnn_input)

        h_0 = hn

        # 
        dist, value = model(torch.squeeze(h_0))
        #dist, value = model(state)

        action = dist.sample()

        next_state, reward, done, _ = env.step(action.item())

        state = next_state
        state = from_obs_to_state(state)
        #if vis: env.render()
        total_reward += reward

    return total_reward


# Compute the returns (of the rewards) for one episode
# 
def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        #R = rewards[step] + gamma * R * masks[step]
        A = rewards[step]
        B = gamma * R * masks[step]
        R = A + B
        returns.insert(0, R)
    return returns

def plot(frame_idx, rewards):
    plt.plot(rewards,'b-')
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.savefig("squares.png")
    plt.pause(0.0001)


# size of the state vector
num_inputs=50

# number of actions
num_outputs=4


#Hyper params:
#hidden_size = 5 #256 hidden size for the Actor Critic
hidden_size = 50
lr          = 1e-3

# number of steps per episode
num_steps   = 30

# Initializing the Actor critic model
model = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)



# number of episodes (frames)
max_frames   = 1000
frame_idx    = 0
test_rewards = []

# we test the model every TT episodes
TT=100


x_axis = []

# we reset the env


obs, reward, info, done = env.reset()


# first state to start with
state = from_obs_to_state(obs)


dim_input = state[0].size(dim=0)



# initializing the RNN model
#rnn = Model(1, 5, 5, 1)
rnn = Model(1, 5, 50, 1)
rnn.to(device)
lr=0.01
criterion = nn.CrossEntropyLoss()


optimizer = optim.Adam(list(model.parameters()) + list(rnn.parameters()) + list(cnn.parameters()))
#rnn = nn.RNN(1, 5, 1)

h_0 = torch.randn(1*1, 1, 50)


returns_cat = torch.tensor([])
values_cat = torch.tensor([])
log_probs_cat = torch.tensor([])

while frame_idx < max_frames:

    os.system("echo frame : "+str(frame_idx))

    optimizer.zero_grad()
    log_probs = []
    values    = []
    rewards   = []
    masks     = []
    entropy = 0

    # trajectory so far
    traj_so_far = []

    # initialization of the hidden state of the RNN
    

    # rollout trajectory
    for ii in range(num_steps):

        os.system("echo step : "+str(ii))

        # we reformat the state
        state = torch.squeeze(state)
        state = torch.squeeze(state.view(5,1))
        
        
        # we append the current state to the trajectory vector
        state = torch.argmax(state).item()

        state = torch.tensor(state).to(torch.float)
        state = torch.unsqueeze(torch.unsqueeze(state, dim=0), dim=0)
        
        state = torch.unsqueeze(state, dim=0)
        # we construct the input for the RNN from the trajectory

        state.to(device)


        # !!!! feed rnn with last state of the trajectory

        # 1 | 50 | 1

        # we input rnn_input into the rnn and retrieve the next hidden state and the output
        #output, hn = rnn(rnn_input)
        
        output, hn = rnn.rnn(state, h_0)
        
        h_0 = hn

        # we input the hidden state into the Actor Critic model
        dist, value = model(torch.squeeze(h_0))

        action = dist.sample()

        # we retrieve the next state, the reward and the state (done) from the env
        next_state, reward, done, _ = env.step(action.item())
        
        # this code was already here (see source in README)
        log_prob = dist.log_prob(action)
        entropy += dist.entropy().mean()
        log_prob = torch.unsqueeze(log_prob, 0)
        log_probs.append(log_prob)

        # storing the value retrieved from the Critic
        values.append(value)

        # storing the reward retrived from the enbv
        reward = float(reward)
        reward = np.expand_dims(reward, axis=0)
        reward = np.expand_dims(reward, axis=0)
        reward = torch.tensor(reward)
        rewards.append(reward)

        ## re init h_0 to 0s when "done" !!!!

        # reformatting done (must be an int)
        

        # we store the "done"
        formask = 1 if done is True else 0
        formask = np.expand_dims(formask, axis=0)
        formask = np.expand_dims(formask, axis=0)
        formask = torch.tensor(formask)

        masks.append(formask)
        
        # we reformat the next state (from an image to a state vector)
        state = next_state
        state = from_obs_to_state(state)


        
    # every TT episodes we run tests for the graph
    if frame_idx % TT == 0:
        test_rewards.append([test_env() for _ in range(2)])
        x_axis.append([frame_idx for _ in range(2)])

    
    frame_idx += 1

    # # next state, used in the computing of the loss of the episode (see source 1)
    # next_state = from_obs_to_state(next_state)

    # reformatting the state
    state = torch.squeeze(state)
    state = torch.squeeze(state.view(5,1))

    # we append the current state to the trajectory vector
    state = torch.argmax(state).item()
    state = torch.tensor(state).to(torch.float)
    state = torch.unsqueeze(torch.unsqueeze(state, dim=0), dim=0)
    state = torch.unsqueeze(state, dim=0)

    output, hn = rnn.rnn(state, h_0)

    # same as before
    dist, next_value = model(torch.squeeze(hn))
    
    action = dist.sample()
    # we retrieve the next state, the reward and the state (done) from the env
    next_state, reward, done, _ = env.step(action.item())

    state = next_state
    state = from_obs_to_state(state)
    ####
    # computing the "returns"
    returns = compute_returns(next_value, rewards, masks)
    
    # see source 2 (we use the same loss for updating the weights of both the actor critic and the rnn)
    log_probs = torch.cat(log_probs)
    returns   = torch.cat(returns).detach()
    values    = torch.cat(values)
    
    print("KKKKKKKKKK")
    print(log_probs.size())
    print(returns.size())
    print(values.size())

    
    returns_cat = torch.cat((returns_cat, returns), 0)
    values_cat = torch.cat((values_cat, values), 0)
    log_probs_cat = torch.cat((log_probs_cat, log_probs), 0)

    # torch.Size([30])
    # torch.Size([30, 1])
    # torch.Size([30])

    # log_probs_cat
    # advantage_cat
    

    if frame_idx % 2 == 0:

        returns_cat = torch.unsqueeze(returns_cat, dim=1)

        advantage = returns_cat - values_cat


        actor_loss  = -(log_probs_cat * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy
        loss.backward(retain_graph=True)
        optimizer.step()


        returns_cat = torch.tensor([])
        values_cat = torch.tensor([])
        log_probs_cat = torch.tensor([])


# plt.plot(rewards,'b-')
# plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
# plt.savefig("squares.png")
# plt.pause(0.0001)



# PLOTTING OF THE RESULTS

fig, axs = plt.subplots(1, 1, figsize=(11, 11))


final_x_axis = []
for xx in x_axis:
    for x in xx:
        final_x_axis.append(x)

final_test_rewards = []
for tt in test_rewards:
    for t in tt:
        final_test_rewards.append(t)

# print("final_x_axis")
# print(final_x_axis)
# print("final_test_rewards")
# print(final_test_rewards)

d = {'frame': final_x_axis, 'rewards': final_test_rewards }
df = pd.DataFrame(data=d)

sns.lineplot(data=df, x="frame", y="rewards", ax = axs)

axs.set_title("Test reward (during training) of RNN + Actor Critic (GridWorld)")
axs.set(xlabel='episode', ylabel="sum of rewards")

plt.tight_layout()
plt.savefig("title_plot_file"+".png")


#test_env(True)
