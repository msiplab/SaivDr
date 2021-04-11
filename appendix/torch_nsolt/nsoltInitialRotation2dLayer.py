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

        # Instantiation of orthormal transforms
        ps,pa = self.number_of_channels
        self.orthTransW0 = OrthonormalTransform(n=ps,mode='Analysis')
        self.orthTransW0.angles = nn.init.zeros_(self.orthTransW0.angles)        
        self.orthTransU0 = OrthonormalTransform(n=pa,mode='Analysis')
        self.orthTransU0.angles = nn.init.zeros_(self.orthTransU0.angles)                

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
            self.orthTransW0.mus[0] = 1
            self.orthTransW0.angles.data[:ps-1] = \
                torch.zeros(ps-1,dtype=self.orthTransW0.angles.data.dtype,device=self.orthTransW0.angles.data.device)
        
        # Process
        ms = int(math.ceil(nDecs/2.))
        ma = int(math.floor(nDecs/2.)) 
        Ys = torch.zeros(ps,nrows*ncols*nSamples,dtype=X.dtype,device=X.device)                       
        Ya = torch.zeros(pa,nrows*ncols*nSamples,dtype=X.dtype,device=X.device)                       
        Ys[:ms,:] = X[:,:,:,:ms].view(-1,ms).T
        if ma > 0:
            Ya[:ma,:] = X[:,:,:,ms:].view(-1,ma).T 
        Zsa = torch.cat(
            ( self.orthTransW0.forward(Ys),
              self.orthTransU0.forward(Ya)),dim=0)
        return Zsa.T.view(nSamples,nrows,ncols,ps+pa)