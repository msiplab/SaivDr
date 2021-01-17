import torch
import torch.nn as nn
import math
from nsoltUtility import Direction
from orthonormalTransform import OrthonormalTransform

class NsoltInitialRotation2dLayer(nn.Module):
    """
    NSOLTINITIALROTATION2DLAYER 
    
       コンポーネント別に入力(nComponents):
          nSamples x nRows x nCols x nDecs
    
       コンポーネント別に出力(nComponents):
          nSamples x nRows x nCols x nChs
    
    Requirements: Python 3.7.x, PyTorch 1.7.x
    
    Copyright (c) 2020-2021, Shogo MURAMATSU
    
    All rights reserved.
    
    Contact address: Shogo MURAMATSU,
        Faculty of Engineering, Niigata University,
        8050 2-no-cho Ikarashi, Nishi-ku,
        Niigata, 950-2181, JAPAN
    
        http://msiplab.eng.niigata-u.ac.jp/ 
    """

    def __init__(self,
        number_of_channels=[],
        decimation_factor=[],
        no_dc_leakage=False,
        name=''):
        super(NsoltInitialRotation2dLayer, self).__init__()
        self.name = name
        self.number_of_channels = number_of_channels
        self.decimation_factor = decimation_factor
        self.description = self.description = "NSOLT initial rotation " \
                + "(ps,pa) = (" \
                + str(self.number_of_channels[0]) + "," \
                + str(self.number_of_channels[1]) + "), "  \
                + "(mv,mh) = (" \
                + str(self.decimation_factor[Direction.VERTICAL]) + "," \
                + str(self.decimation_factor[Direction.HORIZONTAL]) + ")"

    def forward(self,x):
        return x