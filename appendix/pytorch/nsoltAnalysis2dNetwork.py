import torch
import torch.nn as nn
from nsoltBlockDct2dLayer import NsoltBlockDct2dLayer 
from nsoltInitialRotation2dLayer import NsoltInitialRotation2dLayer 

class NsoltAnalysis2dNetwork(nn.Module):
    """
    NSOLTANALYSIS2DNETWORK
    
    Requirements: Python 3.7.x, PyTorch 1.7.x
    
    Copyright (c) 2021, Yasas Dulanjaya, Shogo MURAMATSU
    
    All rights reserved.
    
    Contact address: Shogo MURAMATSU,
        Faculty of Engineering, Niigata University,
        8050 2-no-cho Ikarashi, Nishi-ku,
        Niigata, 950-2181, JAPAN
    
        http://msiplab.eng.niigata-u.ac.jp/ 
    """
    def __init__(self,
        number_of_channels=[],
        decimation_factor=[]):
        super(NsoltAnalysis2dNetwork, self).__init__()
        self.number_of_channels = number_of_channels
        self.decimation_factor = decimation_factor

        # Instantiation of layers
        self.layerE0 = NsoltBlockDct2dLayer(
            decimation_factor=decimation_factor,
            name='E0'
        )
        self.layerV0 = NsoltInitialRotation2dLayer(
            number_of_channels=number_of_channels,
            decimation_factor=decimation_factor,
            name='V0'
        )

    def forward(self,x):
        u = self.layerE0.forward(x)
        y = self.layerV0.forward(u)
        return y
