import torch
import torch.nn as nn
from nsoltUtility import Direction

class NsoltBlockDct2dLayer(nn.Module):
    """
    NSOLTBLOCKDCT2DLAYER
    
       ベクトル配列をブロック配列を入力:
          (Stride(1)xnRows) x (Stride(2)xnCols) x nComponents x nSamples
    
       コンポーネント別に出力(nComponents):
          nDecs x nRows x nCols x nSamples
        
    Requirements: Python 3.7.x, PyTorch 1.7.x
    
    Copyright (c) 2020, Shogo MURAMATSU
    
    All rights reserved.
    
    Contact address: Shogo MURAMATSU,
        Faculty of Engineering, Niigata University,
        8050 2-no-cho Ikarashi, Nishi-ku,
        Niigata, 950-2181, JAPAN
    
    http://msiplab.eng.niigata-u.ac.jp/
    """

    def __init__(self,
        name='',
        decimation_factor=[]
        ):
        super(NsoltBlockDct2dLayer, self).__init__()
        self.decimation_factor = decimation_factor
        self.name = name
        self.description = "Block DCT of size " \
            + str(self.decimation_factor[Direction.VERTICAL]) + "x" \
            + str(self.decimation_factor[Direction.HORIZONTAL])

    def forward(self,x):
        return x
