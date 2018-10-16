import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None):
        super().__init__()

        self.loss=nn.NLLLoss2d(weight)
       
    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs),targets)

class FocalLoss2d(nn.Module):
    def __init__(self,gamma=2,weight=None):
        super().__init__()
        self.gamma=gamma
        self.loss=nn.NLLLoss2d(weight)
    
    def forward(self,outputs,targets):
        return self.loss((1-F.softmax(outputs)) * (1-F.softmax(outputs)) * F.log_softmax(outputs),targets)

