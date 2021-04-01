import torch
import math

def dct(x): 
    """ 
        Revised torch_dct.dct for PyTorch 1.8.x 
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    if torch.__version__[:3] == '1.7': 
        Vc = torch.rfft(v, 1, onesided=False)
    else:
        Vc = torch.fft.fft(v,dim=1)
        
    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * math.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    if torch.__version__[:3] == '1.7':
        V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i
    else:
        V = Vc.real * W_r - Vc.imag * W_i

    # Normalization
    V[:, 0] /= math.sqrt(N) * 2
    V[:, 1:] /= math.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V

def idct(X):
    """ 
        Revised torch_dct.idct for PyTorch 1.8.x
    """
    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    # Normalization
    X_v[:, 0] *= math.sqrt(N) * 2
    X_v[:, 1:] *= math.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * math.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
    if torch.__version__[:3] == '1.7':
        v = torch.irfft(V, 1, onesided=False)
    else:
        v = torch.fft.ifft(torch.view_as_complex(V), dim=1).real

    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape) 

def dct_2d(x):
    X1 = dct(x)
    X2 = dct(X1.transpose(-1, -2))
    return X2.transpose(-1, -2)

def idct_2d(X):
    x1 = idct(X)
    x2 = idct(x1.transpose(-1, -2))
    return x2.transpose(-1, -2)

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
        """
        The output is set on the same device with the angles
        """

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

        # Setup of mus, which is send to the same device with angles
        if isinstance(mus, int) or isinstance(mus, float):
            mus = mus * torch.ones(nDims,dtype=self.dtype,device=angles.device)
        elif not torch.is_tensor(mus): #isinstance(mus, list):
            mus = torch.tensor(mus,dtype=self.dtype,device=angles.device)
        else:
            mus = mus.to(dtype=self.dtype,device=angles.device)

        matrix = torch.eye(nDims,dtype=self.dtype,device=angles.device)
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

