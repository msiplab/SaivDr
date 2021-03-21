import torch
import math

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
        if isinstance(angles, int) or isinstance(angles, float):
            angles = torch.tensor([angles],dtype=self.dtype)
        elif not torch.is_tensor(angles):
            angles = torch.tensor(angles,dtype=self.dtype)
        else:
            angles = angles.to(dtype=self.dtype)
        nAngles = len(angles)

        # Number of dimensions
        nDims = int((1+math.sqrt(1+8*nAngles))/2)

        # Setup of mus
        if isinstance(mus, int) or isinstance(mus, float):
            mus = mus * torch.ones(nDims,dtype=self.dtype)
        elif not torch.is_tensor(mus): #isinstance(mus, list):
            mus = torch.tensor(mus,dtype=self.dtype)
        else:
            mus = mus.to(dtype=self.dtype)

        matrix = torch.eye(nDims,dtype=self.dtype)
        iAng = 0
        for iTop in range(nDims-1):
            vt = matrix[iTop,:]
            for iBtm in range(iTop+1,nDims):
                angle = angles[iAng]
                if self.partial_difference and iAng == index_pd_angle:
                    angle = angle + math.pi/2.
                c = torch.cos(angle)
                s = torch.sin(angle)
                vb = matrix[iBtm,:]
                #
                u  = s*(vt + vb)
                vt = (c + s)*vt
                vb = (c - s)*vb
                vt = vt - u
                if self.partial_difference and iAng == index_pd_angle:
                    matrix = torch.zeros_like(matrix,dtype=self.dtype)
                matrix[iBtm,:] = vb + u
                iAng = iAng + 1
            matrix[iTop,:] = vt
        matrix = mus.view(-1,1) * matrix

        return matrix.clone()

