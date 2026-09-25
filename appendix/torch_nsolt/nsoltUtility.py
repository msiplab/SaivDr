import torch
import math
from nsoltLayerExceptions import InvalidIndex

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

def dctmtx(n,dtype=None,device=None):
    """
    Orthonormal DCT-II matrix, equivalent to MATLAB dctmtx(n)
    """
    k = torch.arange(n,dtype=dtype,device=device).view(-1,1)
    i = torch.arange(n,dtype=dtype,device=device).view(1,-1)
    C = math.sqrt(2./n)*torch.cos(math.pi*(2*i+1)*k/(2*n))
    C[0,:] = C[0,:]/math.sqrt(2.)
    return C

def _dctmtx_even_odd(n,dtype=None,device=None):
    """
    Even- and odd-indexed rows of dctmtx(n)
    """
    C = dctmtx(n,dtype=dtype,device=device)
    return C[0::2,:], C[1::2,:]

def block_dct_matrix_2d(decimation_factor,dtype=None,device=None):
    """
    2-D block DCT matrix, equivalent to Cvh in MATLAB nsoltBlockDct2dLayer

    Rows are ordered as [ Cee; Coo; Coe; Ceo ] and columns correspond to
    the pixels of a decV x decH block in column-major order (v + decV*h).
    """
    decV = decimation_factor[Direction.VERTICAL]
    decH = decimation_factor[Direction.HORIZONTAL]
    Cve, Cvo = _dctmtx_even_odd(decV,dtype=dtype,device=device)
    Che, Cho = _dctmtx_even_odd(decH,dtype=dtype,device=device)
    Cee = torch.kron(Che,Cve)
    Coo = torch.kron(Cho,Cvo)
    Coe = torch.kron(Che,Cvo)
    Ceo = torch.kron(Cho,Cve)
    return torch.cat((Cee,Coo,Coe,Ceo),dim=0)

def block_dct_matrix_3d(decimation_factor,dtype=None,device=None):
    """
    3-D block DCT matrix, equivalent to Cvhd in MATLAB nsoltBlockDct3dLayer

    Rows are ordered as [ Ceee; Ceoo; Cooe; Coeo; Ceeo; Ceoe; Cooo; Coee ]
    (Cyxz), where the coefficients in each group are ordered as in
    saivdr.dictionary.nsoltx (depth fastest, vertical slowest). Columns
    correspond to the voxels of a decV x decH x decD block in column-major
    order (v + decV*h + decV*decH*d).
    """
    decV = decimation_factor[Direction.VERTICAL]
    decH = decimation_factor[Direction.HORIZONTAL]
    decD = decimation_factor[Direction.DEPTH]
    Cve, Cvo = _dctmtx_even_odd(decV,dtype=dtype,device=device)
    Che, Cho = _dctmtx_even_odd(decH,dtype=dtype,device=device)
    Cde, Cdo = _dctmtx_even_odd(decD,dtype=dtype,device=device)
    groups = (
        (Cve,Che,Cde), # Ceee
        (Cve,Cho,Cdo), # Ceoo
        (Cvo,Cho,Cde), # Cooe
        (Cvo,Che,Cdo), # Coeo
        (Cve,Che,Cdo), # Ceeo
        (Cve,Cho,Cde), # Ceoe
        (Cvo,Cho,Cdo), # Cooo
        (Cvo,Che,Cde)  # Coee
        )
    rows = []
    for Cy, Cx, Cz in groups:
        # kron rows are in the order (z,x,y) with y fastest
        G = torch.kron(Cz,torch.kron(Cx,Cy))
        ny, nx, nz = Cy.size(0), Cx.size(0), Cz.size(0)
        # Reorder to y slowest and z fastest as in saivdr.dictionary.nsoltx
        ncols = decV*decH*decD
        G = G.reshape(nz,nx,ny,ncols).permute(2,1,0,3).reshape(ny*nx*nz,ncols)
        rows.append(G)
    return torch.cat(rows,dim=0)

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
        self.nextangle = -1

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
        self.nDims = int((1+math.sqrt(1+8*nAngles))/2)

        # Setup of mus, which is send to the same device with angles
        if isinstance(mus, int) or isinstance(mus, float):
            mus = mus * torch.ones(self.nDims,dtype=self.dtype,device=angles.device)
        elif not torch.is_tensor(mus): #isinstance(mus, list):
            mus = torch.tensor(mus,dtype=self.dtype,device=angles.device)
        else:
            mus = mus.to(dtype=self.dtype,device=angles.device)

        # Generation process
        if self.partial_difference == 'sequential':
            matrix = self.step_sequential_(angles,mus,index_pd_angle)
        else:
            matrix = self.step_normal_(angles,mus,index_pd_angle)
        matrix = mus.view(-1,1) * matrix

        return matrix.clone()
    
    def reset(self):
        self.nextangle = -1

    def step_normal_(self,angles,mus,index_pd_angle):
        #
        iAng = 0
        matrix = torch.eye(self.nDims,dtype=self.dtype,device=angles.device)
        for iTop in range(self.nDims-1):
            vt = matrix[iTop,:]
            for iBtm in range(iTop+1,self.nDims):
                angle = angles[iAng]
                if self.partial_difference and iAng == index_pd_angle:
                    angle = angle + math.pi/2.
                vb = matrix[iBtm,:]
                vt, vb = rot_(vt, vb, angle)
                if self.partial_difference and iAng == index_pd_angle:
                    matrix = torch.zeros_like(matrix,dtype=self.dtype)
                matrix[iBtm,:] = vb
                iAng = iAng + 1
            matrix[iTop,:] = vt
        return matrix

    def step_sequential_(self,angles,mus,index_pd_angle):
        # Check index_pd_angle
        if index_pd_angle >= 0 and index_pd_angle != self.nextangle:
            raise InvalidIndex(
                'Unable to proceed sequential differentiation. Index = %d is expected, but %d was given.'\
                %(self.nextangle, index_pd_angle))
        #
        if index_pd_angle < 0: # Initialization
            self.matrixpst = torch.eye(self.nDims,dtype=self.dtype,device=angles.device)            
            self.matrixpre = torch.eye(self.nDims,dtype=self.dtype,device=angles.device)            
            #
            iAng = 0
            for iTop in range(self.nDims-1):
                vt = self.matrixpst[iTop,:]
                for iBtm in range(iTop+1,self.nDims):
                    angle = angles[iAng]
                    vb = self.matrixpst[iBtm,:]
                    vt, vb = rot_(vt, vb, angle)
                    self.matrixpst[iBtm,:] = vb
                    iAng = iAng + 1
                self.matrixpst[iTop,:] = vt
            self.nextangle = 0
            return self.matrixpst
        else: # Sequential differentiation
            matrix = torch.eye(self.nDims,dtype=self.dtype,device=angles.device)
            matrixrev = matrix.clone()
            matrixdif = torch.zeros_like(matrix,dtype=self.dtype)       
            #
            iAng = 0
            for iTop in range(self.nDims-1):
                rt = matrixrev[iTop,:]
                dt = torch.zeros(1,self.nDims,dtype=self.dtype,device=angles.device)
                dt[0,iTop] = 1
                for iBtm in range(iTop+1,self.nDims):                    
                    if iAng == index_pd_angle:
                        angle = angles[iAng]
                        #
                        rb = matrixrev[iBtm,:]
                        rt,rb = rot_(rt,rb,-angle)
                        matrixrev[iTop,:] = rt
                        matrixrev[iBtm,:] = rb
                        #
                        db = torch.zeros(1,self.nDims,dtype=self.dtype,device=angles.device)
                        db[0,iBtm] = 1
                        dangle = angle + math.pi/2.
                        dt,db = rot_(dt,db,dangle)
                        matrixdif[iTop,:] = dt
                        matrixdif[iBtm,:] = db
                        #
                        self.matrixpst = self.matrixpst @ matrixrev
                        matrix = self.matrixpst @ matrixdif @ self.matrixpre
                        self.matrixpre = matrixrev.T @ self.matrixpre
                    iAng = iAng + 1
            self.nextangle += 1
            return matrix

def rot_(vt,vb,angle):
    c = torch.cos(angle)
    s = torch.sin(angle)
    u  = s*(vt + vb)
    vt = (c + s)*vt
    vb = (c - s)*vb
    vt = vt - u
    vb = vb + u
    return vt, vb
