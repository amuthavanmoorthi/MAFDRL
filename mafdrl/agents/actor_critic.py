# defines Actor and Critic networks
import torch as th
import torch.nn as nn

def mlp(sizes, act=nn.ReLU, last_act=None):
    layers=[]; 
    for i in range(len(sizes)-1):
        layers+=[nn.Linear(sizes[i], sizes[i+1])]
        if i < len(sizes)-2:
            layers+=[act()]
        elif last_act:
            layers+=[last_act()]
    return nn.Sequential(*layers)

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.body = mlp([obs_dim, 128, 128], act=nn.ReLU)
        self.head = nn.Linear(128, act_dim)  # Y=WX+b affine head (your professor’s form)
    def forward(self, obs):
        h = self.body(obs)
        y = self.head(h)                     # linear head = affine transform
        return y

class Critic(nn.Module):
    # centralized critic Q(s, a_1,...,a_U) -> scalar
    def __init__(self, in_dim):
        super().__init__()
        self.net = mlp([in_dim, 256, 256, 1], act=nn.ReLU)
    def forward(self, x):
        return self.net(x)
