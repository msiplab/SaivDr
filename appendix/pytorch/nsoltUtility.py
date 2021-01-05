import torch
import numpy as np

class Direction:
    VERTICAL = 0
    HORIZONTAL = 1
    DEPTH = 2

class OrthonormalMatrixGenerationSystem:
    """
    ORTHONORMALMATRIXGENERATIONSYSTEM
    
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
        dtype=torch.get_default_dtype(),
        partial_difference=False):
        super(OrthonormalMatrixGenerationSystem, self).__init__()
        self.dtype = dtype
        self.partial_difference = partial_difference

    def __call__(self,
        angles=0,
        mus=1,
        index_pd_angle=None):
        
        # Number of angles
        if np.isscalar(angles):
            angles = np.array([angles])
        nAngles = len(angles)

        # Number of dimensions
        nDims = ((1+np.sqrt(1+8*nAngles))/2).astype(int)

        # Setup of mus
        if np.isscalar(mus):
            mus = mus * np.eye(nDims)
        else:
            mus = np.diag(mus)

        matrix = np.eye(nDims)
        iAng = 0
        for iTop in range(nDims-1):
            vt = matrix[iTop,:]
            for iBtm in range(iTop+1,nDims):
                angle = angles[iAng]
                if iAng == index_pd_angle:
                    angle = angle + np.pi/2.
                c = np.cos(angle)
                s = np.sin(angle)
                vb = matrix[iBtm,:]
                #
                u  = s*(vt + vb)
                vt = (c + s)*vt
                vb = (c - s)*vb
                vt = vt - u
                if iAng == index_pd_angle:
                    matrix = np.zeros_like(matrix)
                matrix[iBtm,:] = vb + u
                iAng = iAng + 1
            matrix[iTop,:] = vt
        matrix = mus.dot(matrix)

        return torch.tensor(matrix,dtype=self.dtype)

