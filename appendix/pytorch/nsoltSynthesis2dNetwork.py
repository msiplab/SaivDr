import torch
import torch.nn as nn
from nsoltBlockIdct2dLayer import NsoltBlockIdct2dLayer 
from nsoltFinalRotation2dLayer import NsoltFinalRotation2dLayer 

class NsoltSynthesis2dNetwork(nn.Module):

    def __init__(self,
        number_of_channels=[],
        decimation_factor=[]):
        super(NsoltSynthesis2dNetwork, self).__init__()
        self.number_of_channels = number_of_channels
        self.decimation_factor = decimation_factor

    def forward(self,x):
        return x
