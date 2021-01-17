import torch
import torch.nn as nn
import math
from nsoltUtility import Direction
from orthonormalTransform import OrthonormalTransform

class NsoltFinalRotation2dLayer(nn.Module):
    """
    NSOLTFINALROTATION2DLAYER 
    
       コンポーネント別に入力(nComponents):
          nSamples x nRows x nCols x nChs
    
       コンポーネント別に出力(nComponents):
          nSamples x nRows x nCols x nDecs
    
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
        super(NsoltFinalRotation2dLayer, self).__init__()
        self.name = name
        self.number_of_channels = number_of_channels
        self.decimation_factor = decimation_factor
        self.description = "NSOLT final rotation " \
                + "(ps,pa) = (" \
                + str(self.number_of_channels[0]) + "," \
                + str(self.number_of_channels[1]) + "), "  \
                + "(mv,mh) = (" \
                + str(self.decimation_factor[Direction.VERTICAL]) + "," \
                + str(self.decimation_factor[Direction.HORIZONTAL]) + ")"

        # Instantiation of orthormal transforms
        ps,pa = self.number_of_channels
        self.orthTransW0T = OrthonormalTransform(n=ps,mode='Synthesis')
        self.orthTransW0T.angles = nn.init.zeros_(self.orthTransW0T.angles)        
        self.orthTransU0T = OrthonormalTransform(n=pa,mode='Synthesis')
        self.orthTransU0T.angles = nn.init.zeros_(self.orthTransU0T.angles)                

        # No DC leakage
        self.no_dc_leakage = no_dc_leakage

    def forward(self,X):
        nSamples = X.size(dim=0)
        nrows = X.size(dim=1)
        ncols = X.size(dim=2)
        ps, pa = self.number_of_channels
        stride = self.decimation_factor
        nDecs = stride[0]*stride[1] # math.prod(stride)

        # No DC leackage
        if self.no_dc_leakage:
            self.orthTransW0T.mus[0] = 1
            self.orthTransW0T.angles.data[:ps-1] = \
                torch.zeros(ps-1,dtype=self.orthTransW0T.angles.data.dtype)

        # Process
        Ys = X[:,:,:,:ps].view(-1,ps).T
        Ya = X[:,:,:,ps:].view(-1,pa).T 
        ms = int(math.ceil(nDecs/2.))
        ma = int(math.floor(nDecs/2.))
        Zsa = torch.cat( 
            ( self.orthTransW0T.forward(Ys)[:ms,:],
              self.orthTransU0T.forward(Ya)[:ma,:]),
             dim=0 )
        return Zsa.T.view(nSamples,nrows,ncols,nDecs)