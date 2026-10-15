import torch
import torch.nn as nn
import math
from nsoltUtility import Direction, block_dct_matrix_3d

class NsoltBlockDct3dLayer(nn.Module):
    """
    NSOLTBLOCKCDT3DLAYER
    
       ベクトル配列をブロック配列を入力:
          nSamples x nComponents x (Stride[0]xnRows) x (Stride[1]xnCols) x (Stride[2]xnLays)
    
       コンポーネント別に出力(nComponents):
          nSamples x nDecs x nRows x nCols x nLays
        
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
        super(NsoltBlockDct3dLayer, self).__init__()
        self.decimation_factor = decimation_factor
        self.name = name
        self.description = "Block DCT of size " \
            + str(self.decimation_factor[Direction.VERTICAL]) + "x" \
            + str(self.decimation_factor[Direction.HORIZONTAL]) + "x" \
            + str(self.decimation_factor[Direction.DEPTH])
        #self.type = ''
        self.num_outputs = number_of_components
        #self.num_inputs = 1

    def forward(self,X):
        nComponents = self.num_outputs
        nSamples = X.size(0)
        height = X.size(2)
        width = X.size(3)
        depth = X.size(4)
        stride = self.decimation_factor
        decV = stride[Direction.VERTICAL]
        decH = stride[Direction.HORIZONTAL]
        decD = stride[Direction.DEPTH]
        nrows = int(math.ceil(height/decV))
        ncols = int(math.ceil(width/decH))
        nlays = int(math.ceil(depth/decD))
        ndecs = decV*decH*decD

        # Block DCT matrix (the same as Cvhd in MATLAB nsoltBlockDct3dLayer)
        Cvhd = block_dct_matrix_3d(stride,dtype=X.dtype,device=X.device)
        # Split into decV x decH x decD blocks, whose voxels are arranged in
        # column-major order as in MATLAB:
        # (nSamples x nComponents x nrows x ncols x nlays) x (decD x decH x decV)
        arrayX = X.reshape(nSamples,nComponents,nrows,decV,ncols,decH,nlays,decD)\
            .permute(0,1,2,4,6,7,5,3)\
            .reshape(nSamples,nComponents,nrows,ncols,nlays,ndecs)
        # Apply the DCT: nSamples x nComponents x nrows x ncols x nlays x ndecs
        Z = arrayX @ Cvhd.T

        if nComponents<2:
            return torch.squeeze(Z,dim=1)
        else:
            return map(lambda x: torch.squeeze(x,dim=1),torch.chunk(Z,nComponents,dim=1))
