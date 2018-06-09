import torch
import torch.nn as nn
import torch.nn.functional as F

class Voice2Vec(nn.Module):
    def __init__(self, sound_shape, dim):
        """
            sound_shape: tuple (time, frequency).
            dim: размерность выходного вектора.
        """
        super(Voice2Vec, self).__init__()
        self.sound_shape = sound_shape[::-1] # (frequency, time)
        self.dim = dim
        
        self.conv1 = nn.Conv1d(in_channels=self.sound_shape[0], out_channels=32, kernel_size=5) # 96
        self.max_pool1 = nn.MaxPool1d(2) # 48
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5) # 44 
        self.max_pool2 = nn.MaxPool1d(2) # 22
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5) # 18
        self.max_pool3 = nn.MaxPool1d(2) # 9
        self.fc1 = nn.Linear(9 * 128, 300)
        self.fc2 = nn.Linear(300, self.dim)
        
    def forward(self, x):
        """
        Args:
            x : torch.Variable, shape = (batch_size, 3, frequency, time) конкатенация из трёх речей:
                1. речь с голосом 1
                2. другая речь с голосом 1
                3. другой голос
        Return:
            y : torch.Variable, shape = (batch_size, 3, dim)
        """
        x = x.view((-1,) + self.sound_shape)
        x = F.leaky_relu(self.conv1(x))
        x = self.max_pool1(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.max_pool2(x)
        x = F.leaky_relu(self.conv3(x))
        x = self.max_pool3(x)
        x = x.view((-1, 9 * 128))
        x = F.relu(self.fc1(x))
        x = F.tanh(self.fc2(x))
        norm = torch.norm(x, dim=-1)
        x /= norm[:, None]
        x = x.view((-1, 3, self.dim))
        return x
    
    def get_vector(self, x):
        """
        Args:
            x : torch.Variable, shape = (3, frequency, time) конкатенация из трёх речей:
                1. речь с голосом 1
                2. другая речь с голосом 1
                3. другой голос
        Return:
            y : torch.Variable()
        """
        y = x.view((1, ) + x.shape)
        y = self.forward(y)
        return y[0, 0].data.numpy()