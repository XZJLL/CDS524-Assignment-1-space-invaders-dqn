# library
import numpy as np  
import random  
from collections import namedtuple, deque 
from dqn import DQN  

import torch  #PyTorch
import torch.optim as optim 

# use GPU (or cpu if no gpu)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# buffer for storing info
class ReplayBuffer:
  def __init__(self, action_size, buffer_size, batch_size, seed):
    self.action_size = action_size  
    self.memory = deque(maxlen=buffer_size)  
    self.batch_size = batch_size  
    
    self.experiences = namedtuple(
        "Experience", [
            "state", "action", "reward", "next_state", "done"])
    self.seed = random.seed(seed)  
# add subfunction to buffer
  def add(self, state, action, reward, next_state, done):
    
    e = self.experiences(state, action, reward, next_state, done)
    self.memory.append(e)

  def sample(self):
    # get info radomly from buffer
    experiences = random.sample(self.memory, k=self.batch_size)
    
    states = torch.cat([e.state for e in experiences if e is not None])
    actions = torch.from_numpy(
        np.vstack([e.action for e in experiences if e is not None])).long().to(DEVICE)
    rewards = torch.from_numpy(
        np.vstack([e.reward for e in experiences if e is not None])).float().to(DEVICE)
    next_states = torch.cat(
        [e.next_state for e in experiences if e is not None])
    dones = torch.from_numpy(np.vstack(
        [e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(DEVICE)
    return (states, actions, rewards, next_states, dones)

  def __len__(self):
    
    return len(self.memory)


def rgb2gray(rgb):
      # rgb to gray
      return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140]
                    )[..., np.newaxis] / 255

class DQNAgent():
  def __init__(self, state_size, action_size, seed, lr=1e-3, gamma=0.99,
               tau=1e-3, buffer_size=int(1e5), batch_size=64, update_every=100):
    # INITIALIZE PARAM
    self.state_size = state_size  
    self.action_size = action_size  
    self.seed = random.seed(seed) 
    self.batch_size = batch_size 
    self.update_every = update_every  
    self.gamma = gamma  
    self.tau = tau  

    # init local & target qnetwork
    self.qnetwork_local = DQN(state_size, action_size, seed).to(DEVICE)
    self.qnetwork_target = DQN(state_size, action_size, seed).to(DEVICE)
    # set opt
    self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
    # store to buffer
    self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)
    self.t_step = 0  

  def preprocess_state(self, s):
    
    state = rgb2gray(s.copy())  
    
    state = torch.from_numpy(state[15:200, 30:125, :].transpose(
        2, 0, 1)).float().unsqueeze(0).to(DEVICE)
    return state

  def step(self, state_, action, reward, next_state_, done):
    
    state = self.preprocess_state(state_.copy())  
    next_state = self.preprocess_state(next_state_.copy())  
    self.memory.add(state, action, reward, next_state, done)  
    self.t_step = (self.t_step + 1) % self.update_every  
    if self.t_step == 0:  
      if len(self.memory) > self.batch_size:  
        experience = self.memory.sample()  
        self.learn(experience) 

  def act(self, state, eps=0):
    
    state = rgb2gray(state)  
    state = torch.from_numpy(state[15:200, 30:125, :].transpose(
        2, 0, 1)).float().unsqueeze(0).to(DEVICE)  
    self.qnetwork_local.eval()  
    with torch.no_grad():  
      action_values = self.qnetwork_local(state)  
    self.qnetwork_local.train()  
    # e-greedy
    if random.random() > eps:
      return np.argmax(action_values.cpu().data.numpy())  # select highest
    else:
      return random.choice(np.arange(self.action_size))  

  def learn(self, experiences):
    states, actions, rewards, next_states, dones = experiences  
    criterion = torch.nn.MSELoss()  # MSE
    self.qnetwork_local.train()  
    self.qnetwork_target.eval()  
    predicted_targets = self.qnetwork_local(
        states).gather(1, actions)  # CAL predicted Q
    with torch.no_grad():  
      labels_next = self.qnetwork_target(next_states).detach().max(1)[
          0].unsqueeze(1)  # cal local q
    labels = rewards + (self.gamma * labels_next * (1 - dones))  
    loss = criterion(predicted_targets, labels).to(DEVICE)  
    self.optimizer.zero_grad()  
    loss.backward()  # backward
    self.optimizer.step()  
    

  def hard_update(self):
    self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
