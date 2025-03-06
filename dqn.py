import torch 
import torch.nn as nn  
import torch.nn.functional as F  


class DQN(nn.Module):  # def DQN from nn.Module
  def __init__(self, state_size, action_size, seed, fc1_unit=64, fc2_unit=64):
    super(DQN, self).__init__()  
    self.seed = torch.manual_seed(seed)
    self.conv1 = nn.Conv2d(1, 32, 8, stride=4, padding=1)    # def layer
    self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
    self.conv3 = nn.Conv2d(64, 128, 3)
    self.fc1 = nn.Linear(1 * 128 * 19 * 8, 512)
    self.fc2 = nn.Linear(512, action_size)

  def forward(self, state):
    x = state.clone()
    x = x.view(-1, 1, 185, 95)
    # ReLU
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = x.view(-1, 128 * 19 * 8) 
    x = F.relu(self.fc1(x))

    return self.fc2(x)
