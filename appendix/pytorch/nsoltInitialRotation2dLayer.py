import torch
import torch.nn as nn

class NsoltInitialRotation2dLayer(nn.Module):

    def __init__(self):
        super(NsoltInitialRotation2dLayer, self).__init__()

    def forward(self,x):
        return x