#import torch
import torch.nn as nn
#import math 
from nsoltUtility import Direction
from orthonormalTransform import OrthonormalTransform

class NsoltIntermediateRotation2dLayer(nn.Module):
    """
    NSOLTINTERMEDIATEROTATION2DLAYER 
    
       コンポーネント別に入力(nComponents):
          nSamples x nRows x nCols x nChs
    
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
        mode='Synthesis',
        name='',
        mus=1):
        super(NsoltIntermediateRotation2dLayer, self).__init__()
        self.name = name
        self.number_of_channels = number_of_channels
        self.description = mode \
                + " NSOLT intermediate rotation " \
                + "(ps,pa) = (" \
                + str(self.number_of_channels[0]) + "," \
                + str(self.number_of_channels[1]) + ")"

        # Instantiation of orthormal transforms
        ps,pa = self.number_of_channels                
        self.orthTransUn = OrthonormalTransform(n=pa,mode=mode)
        self.orthTransUn.angles = nn.init.zeros_(self.orthTransUn.angles)
        self.orthTransUn.mus = mus

    def forward(self,X):
        nSamples = X.size(dim=0)
        nrows = X.size(dim=1)
        ncols = X.size(dim=2)
        ps,pa = self.number_of_channels

        # Process
        Z = X.clone()
        Ya = X[:,:,:,ps:].view(-1,pa).T 
        Za = self.orthTransUn.forward(Ya)
        Z[:,:,:,ps:] = Za.T.view(nSamples,nrows,ncols,pa)
        return Z

    @property
    def mode(self):
        return self.orthTransUn.mode