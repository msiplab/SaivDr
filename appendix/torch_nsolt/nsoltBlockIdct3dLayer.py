import torch
import torch.nn as nn
import torch_dct as dct
import math
from nsoltUtility import Direction

class NsoltBlockIdct3dLayer(nn.Module):
    """
    NSOLTBLOCKIDCT3DLAYER
    
       コンポーネント別に入力(nComponents):
          nSamples x nRows x nCols x nLays x nDecs 
    
       ベクトル配列をブロック配列にして出力:
          nSamples x nComponents x (Stride[0]xnRows) x (Stride[1]xnCols) x (Stride[2]xnLays)
        
    Requirements: Python 3.7.x, PyTorch 1.7.x
    
    Copyright (c) 2020-2021, Yuya Kodama, Shogo MURAMATSU
    
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
        super(NsoltBlockIdct3dLayer, self).__init__()
        self.decimation_factor = decimation_factor 
        self.name = name 
        self.description = "Block IDCT of size " \
            + str(self.decimation_factor[Direction.VERTICAL]) + "x" \
            + str(self.decimation_factor[Direction.HORIZONTAL]) + "x" \
            + str(self.decimation_factor[Direction.DEPTH])
        #self.type = ''
        self.num_inputs = number_of_components

    def forward(self,*args):
        block_size = self.decimation_factor
        for iComponent in range(self.num_inputs):
            X = args[iComponent]
            nsamples = X.size(0)
            nrows = X.size(1)
            ncols = X.size(2)
            nlays = X.size(3)
            
            # Permute IDCT coefficients
            V = permuteIdctCoefs_(X,block_size)
            # 3D IDCT
            Y = dct.idct_3d(V,norm='ortho')
            # Reshape and return
            height = nrows * block_size[Direction.VERTICAL] 
            width = ncols * block_size[Direction.HORIZONTAL] 
            depth = nlays * block_size[Direction.DEPTH]
            if iComponent<1:
                Z = Y.reshape(nsamples,1,height,width,depth)
            else:
                Z = torch.cat((Z,Y.reshape(nsamples,1,height,width,depth)),dim=1)
        return Z

def permuteIdctCoefs_(x,block_size):
    coefs = x.view(-1,block_size[Direction.VERTICAL]*block_size[Direction.HORIZONTAL]*block_size[Direction.DEPTH]) # x.view(-1,math.prod(block_size)) 
    decY_ = block_size[Direction.VERTICAL]
    decX_ = block_size[Direction.HORIZONTAL]
    decZ_ = block_size[Direction.DEPTH]
    chDecY = int(math.ceil(decY_/2.)) #.astype(int)
    chDecX = int(math.ceil(decX_/2.)) #.astype(int)
    chDecZ = int(math.ceil(decZ_/2.)) #.astype(int)
    fhDecY = int(math.floor(decY_/2.)) #.astype(int)
    fhDecX = int(math.floor(decX_/2.)) #.astype(int)
    fhDecZ = int(math.floor(decZ_/2.)) #.astype(int)

    nQDecseee = chDecY*chDecX*chDecZ
    nQDecseeo = chDecY*chDecX*fhDecZ
    nQDecseoe = chDecY*fhDecX*chDecZ
    nQDecseoo = chDecY*fhDecX*fhDecZ
    nQDecsoee = fhDecY*chDecX*chDecZ
    nQDecsoeo = fhDecY*chDecX*fhDecZ
    nQDecsooe = fhDecY*fhDecX*chDecZ
    nQDecsooo = fhDecY*fhDecX*fhDecZ

    start_idx = 0
    ceee, start_idx = coefs_align(coefs,start_idx,start_idx+nQDecseee)
    ceeo, start_idx = coefs_align(coefs,start_idx,start_idx+nQDecseeo)
    ceoe, start_idx = coefs_align(coefs,start_idx,start_idx+nQDecseoe)
    ceoo, start_idx = coefs_align(coefs,start_idx,start_idx+nQDecseoo)
    coee, start_idx = coefs_align(coefs,start_idx,start_idx+nQDecsoee)
    coeo, start_idx = coefs_align(coefs,start_idx,start_idx+nQDecsoeo)
    cooe, start_idx = coefs_align(coefs,start_idx,start_idx+nQDecsooe)
    cooo, start_idx = coefs_align(coefs,start_idx,start_idx+nQDecsooo)

    nBlocks = coefs.size(0)

    value = torch.zeros(nBlocks,decY_,decX_,decZ_,dtype=x.dtype)

    value[:,0::2,0::2,0::2] = ceee.view(nBlocks,chDecY,chDecX,chDecZ)
    value[:,0::2,0::2,1::2] = ceeo.view(nBlocks,chDecY,chDecX,fhDecZ)
    value[:,0::2,1::2,0::2] = ceoe.view(nBlocks,chDecY,fhDecX,chDecZ)
    value[:,0::2,1::2,1::2] = ceoo.view(nBlocks,chDecY,fhDecX,fhDecZ)
    value[:,1::2,0::2,0::2] = coee.view(nBlocks,fhDecY,chDecX,chDecZ)
    value[:,1::2,0::2,1::2] = coeo.view(nBlocks,fhDecY,chDecX,fhDecZ)
    value[:,1::2,1::2,0::2] = cooe.view(nBlocks,fhDecY,fhDecX,chDecZ)
    value[:,1::2,1::2,1::2] = cooo.view(nBlocks,fhDecY,fhDecX,fhDecZ)
    
    return value

def coefs_align(coefs,start_idx,end_idx):
    output = coefs[:,start_idx:end_idx]
    return output, end_idx