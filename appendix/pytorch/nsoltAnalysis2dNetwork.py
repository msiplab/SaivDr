import torch
import torch.nn as nn

class NsoltAnalysis2dNetwork(nn.Module):
    """
    NSOLTANALYSIS2DNETWORK
    
    Requirements: Python 3.7.x, PyTorch 1.7.x
    
    Copyright (c) 2021, Yasas Dulanjaya and Shogo MURAMATSU
    
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

    def forward(self,x):
        return x
