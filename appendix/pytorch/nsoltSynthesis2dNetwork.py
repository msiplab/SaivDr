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
        
        # Instantiation of layers
        self.layerV0T = NsoltFinalRotation2dLayer(
            number_of_channels=number_of_channels,
            decimation_factor=decimation_factor,
            name='V0~'
        )
        self.layerE0T = NsoltBlockIdct2dLayer(
            decimation_factor=decimation_factor,
            name='E0~'
        )

    def forward(self,x):
        u = self.layerV0T.forward(x)
        y = self.layerE0T.forward(u) 
        return y
