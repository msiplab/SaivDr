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
        n=2):
        super(OrthonormalTransform, self).__init__()
        nAngs = int(n*(n-1)/2)
        self.nPoints = n
        self.angles = nn.Parameter(torch.zeros(nAngs))
        self.mus = torch.ones(self.nPoints)

    def forward(self,X):
        nPoints = self.nPoints
        angles = self.angles
        if X.ndim == 1:
            Y = X.unsqueeze(1)
        else:
            Y = X
        iAng = 0
        for iTop in range(nPoints-1):
            vt = Y[iTop,:]
            for iBtm in range(iTop+1,nPoints):
                angle = angles[iAng]
                c = torch.cos(angle)
                s = torch.sin(angle)
                vb = Y[iBtm,:]
                #
                u = s*(vt + vb)
                vt = (c + s)*vt
                vb = (c - s)*vb
                vt -= u
                Y[iBtm,:] = vb + u
                iAng += 1
            Y[iTop,:] = vt
        for irow in range(Y.size(0)):
            Y[irow,:] *= self.mus[irow]
            
        return Y
