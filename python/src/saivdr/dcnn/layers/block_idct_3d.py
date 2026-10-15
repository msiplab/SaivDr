import torch
import torch.nn as nn
import math
from nsoltUtility import Direction, block_dct_matrix_3d

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
        decV = block_size[Direction.VERTICAL]
        decH = block_size[Direction.HORIZONTAL]
        decD = block_size[Direction.DEPTH]
        for iComponent in range(self.num_inputs):
            X = args[iComponent]
            nsamples = X.size(0)
            nrows = X.size(1)
            ncols = X.size(2)
            nlays = X.size(3)
            # Block IDCT matrix (the transpose of Cvhd in MATLAB)
            Cvhd = block_dct_matrix_3d(block_size,dtype=X.dtype,device=X.device)
            # nsamples x nrows x ncols x nlays x (decD x decH x decV)
            arrayY = X @ Cvhd
            # Place the blocks: nsamples x 1 x height x width x depth
            height = nrows * decV
            width = ncols * decH
            depth = nlays * decD
            Y = arrayY.reshape(nsamples,nrows,ncols,nlays,decD,decH,decV)\
                .permute(0,1,6,2,5,3,4)\
                .reshape(nsamples,1,height,width,depth)
            if iComponent<1:
                Z = Y
            else:
                Z = torch.cat((Z,Y),dim=1)
        return Z
