import torch
import torch.nn as nn
#import torch_dct as dct
import math
from nsoltUtility import Direction, idct_2d

class NsoltBlockIdct2dLayer(nn.Module):
    """
    NSOLTBLOCKIDCT2DLAYER
    
       コンポーネント別に入力(nComponents):
          nSamples x nRows x nCols x nDecs
    
       ベクトル配列をブロック配列にして出力:
          nSamples x nComponents x (Stride(1)xnRows) x (Stride(2)xnCols) 
        
    Requirements: Python 3.7.x, PyTorch 1.7.x/1.8.x
    
    Copyright (c) 2020-2021, Shogo MURAMATSU
    
    All rights reserved.
    
    Contact address: Shogo MURAMATSU,
        Faculty of Engineering, Niigata University,
        8050 2-no-cho Ikarashi, Nishi-ku,
        Niigata, 950-2181, JAPAN
    
        http://msiplab.eng.niigata-u.ac.jp/
    """
    def __init__(self,
        name='',
        decimation_factor=[],
        number_of_components=1
        ):
        super(NsoltBlockIdct2dLayer, self).__init__()
        self.decimation_factor = decimation_factor 
        self.name = name 
        self.description = "Block IDCT of size " \
            + str(self.decimation_factor[Direction.VERTICAL]) + "x" \
            + str(self.decimation_factor[Direction.HORIZONTAL])
        #self.type = ''
        self.num_inputs = number_of_components

    def forward(self,*args):
        block_size = self.decimation_factor
        for iComponent in range(self.num_inputs):
            X = args[iComponent]
            nsamples = X.size(0)
            nrows = X.size(1)
            ncols = X.size(2)
            # Permute IDCT coefficients
            V = permuteIdctCoefs_(X,block_size)
            # 2D IDCT
            Y = idct_2d(V)
            # Reshape and return
            height = nrows * block_size[Direction.VERTICAL] 
            width = ncols * block_size[Direction.HORIZONTAL] 
            if iComponent<1:
                Z = Y.reshape(nsamples,1,height,width)
            else:
                Z = torch.cat((Z,Y.reshape(nsamples,1,height,width)),dim=1)
        return Z

def permuteIdctCoefs_(x,block_size):
    coefs = x.view(-1,block_size[Direction.VERTICAL]*block_size[Direction.HORIZONTAL]) # math.prod(block_size)
    decY_ = block_size[Direction.VERTICAL]
    decX_ = block_size[Direction.HORIZONTAL]
    chDecY = int(math.ceil(decY_/2.))
    chDecX = int(math.ceil(decX_/2.))
    fhDecY = int(math.floor(decY_/2.))
    fhDecX = int(math.floor(decX_/2.))
    nQDecsee = chDecY*chDecX
    nQDecsoo = fhDecY*fhDecX
    nQDecsoe = fhDecY*chDecX
    cee = coefs[:,:nQDecsee]
    coo = coefs[:,nQDecsee:nQDecsee+nQDecsoo]
    coe = coefs[:,nQDecsee+nQDecsoo:nQDecsee+nQDecsoo+nQDecsoe]
    ceo = coefs[:,nQDecsee+nQDecsoo+nQDecsoe:]
    nBlocks = coefs.size(0)
    value = torch.empty(nBlocks,decY_,decX_,dtype=x.dtype,device=x.device)
    value[:,0::2,0::2] = cee.view(nBlocks,chDecY,chDecX)
    value[:,1::2,1::2] = coo.view(nBlocks,fhDecY,fhDecX)
    value[:,1::2,0::2] = coe.view(nBlocks,fhDecY,chDecX)
    value[:,0::2,1::2] = ceo.view(nBlocks,chDecY,fhDecX)
    return value
