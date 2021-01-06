import torch
import torch.nn as nn
import numpy as np

class OrthonormalTransform(nn.Module):
    """
    ORTHONORMALTRANSFORMTESTCASE
    
    Requirements: Python 3.7.x, PyTorch 1.7.x
    
    Copyright (c) 2021, Shogo MURAMATSU
    
    All rights reserved.
    
    Contact address: Shogo MURAMATSU,
        Faculty of Engineering, Niigata University,
        8050 2-no-cho Ikarashi, Nishi-ku,
        Niigata, 950-2181, JAPAN
    
        http://msiplab.eng.niigata-u.ac.jp/    
    """
    def __init__(self,
        dimension=2,
        in_features=None,
        out_features=None):
        super(OrthonormalTransform, self).__init__()
        nAngs = int(dimension*(dimension-1)/2)
        self.angles = nn.Parameter(torch.zeros(nAngs))
        self.mus = torch.ones(dimension)

    def forward(self,X):
        c = torch.cos(self.angles[0])
        s = torch.sin(self.angles[0])
        R = torch.tensor([
            [ c, -s ],
            [ s,  c ] ],
            dtype=X.dtype)
        return R @ X
