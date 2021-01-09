import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
from nsoltUtility import OrthonormalMatrixGenerationSystem
from nsoltLayerExceptions import InvalidMode, InvalidMus

class OrthonormalTransform(nn.Module):
    """
    ORTHONORMALTRANSFORM
    
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
        n=2,
        mus=1,
        mode='Analysis',
        dtype=torch.get_default_dtype()):

        super(OrthonormalTransform, self).__init__()
        nAngs = int(n*(n-1)/2)
        self.dtype = dtype
        self.nPoints = n
        if mode in {'Analysis','Synthesis'}:
            self.mode = mode
        else:
            raise InvalidMode(
                '%s : Mode should be either of Analysis or Synthesis'\
                % str(mode)
            )
        self.angles = nn.Parameter(torch.zeros(nAngs,dtype=self.dtype))
        if torch.is_tensor(mus):
            self.mus = mus
        elif mus == 1:
            self.mus = torch.ones(self.nPoints,dtype=self.dtype)
        elif mus == -1:
            self.mus = -torch.ones(self.nPoints,dtype=self.dtype)
        else:
            raise InvalidMus(
                '%s : Mus should be either of 1 or -1'\
                % str(mus)
            )
        # TODO: Check if mus in {-1,1}^n

    def forward(self,X):
        angles = self.angles
        mus = self.mus
        if self.mode=='Analysis':
            givensrots = GivensRotations4Analyzer.apply
        elif self.mode=='Synthesis':
            givensrots = GivensRotations4Synthesizer.apply
        else:
            raise InvalidMode(
                '%s : Mode should be either of Analysis or Synthesis'\
                % str(mode)
            )
        return givensrots(X,angles,mus)            

class GivensRotations4Analyzer(autograd.Function):
    """
    GIVENSROTATIONS4ANALYZER
    
    Requirements: Python 3.7.x, PyTorch 1.7.x
    
    Copyright (c) 2021, Shogo MURAMATSU
    
    All rights reserved.
    
    Contact address: Shogo MURAMATSU,
        Faculty of Engineering, Niigata University,
        8050 2-no-cho Ikarashi, Nishi-ku,
        Niigata, 950-2181, JAPAN
    
        http://msiplab.eng.niigata-u.ac.jp/    
    """ 

    @staticmethod
    def forward(ctx, input, angles, mus):
        ctx.mark_non_differentiable(mus)
        ctx.save_for_backward(input,angles,mus)
        omgs = OrthonormalMatrixGenerationSystem(dtype=input.dtype)
        R = omgs(angles,mus)
        return R @ input
    
    @staticmethod
    def backward(ctx, grad_output):
        input, angles, mus = ctx.saved_tensors
        omgs = OrthonormalMatrixGenerationSystem(dtype=input.dtype)
        R = omgs(angles,mus)
        grad_input = R.T @ grad_output # dLdX = dZdX @ dLdZ
        if ctx.needs_input_grad[1]:
            omgs.partial_difference=True
            grad_angles = torch.empty_like(angles,dtype=input.dtype)
            for iAngle in range(len(grad_angles)):
                dRi = omgs(angles,mus,index_pd_angle=iAngle)
                grad_angles[iAngle] = torch.sum(grad_input * (dRi @ input))
        grad_mus = torch.zeros_like(mus,dtype=input.dtype)                
        return grad_input, grad_angles, grad_mus

class GivensRotations4Synthesizer(autograd.Function):
    """
    GIVENSROTATIONS4SYNTHESIZER
    
    Requirements: Python 3.7.x, PyTorch 1.7.x
    
    Copyright (c) 2021, Shogo MURAMATSU
    
    All rights reserved.
    
    Contact address: Shogo MURAMATSU,
        Faculty of Engineering, Niigata University,
        8050 2-no-cho Ikarashi, Nishi-ku,
        Niigata, 950-2181, JAPAN
    
        http://msiplab.eng.niigata-u.ac.jp/    
    """ 

    @staticmethod
    def forward(ctx, input, angles, mus):
        ctx.mark_non_differentiable(mus)        
        ctx.save_for_backward(input,angles,mus)
        omgs = OrthonormalMatrixGenerationSystem(dtype=input.dtype)
        R = omgs(angles,mus)
        return R.T @ input
    
    @staticmethod
    def backward(ctx, grad_output):
        input, angles, mus = ctx.saved_tensors
        omgs = OrthonormalMatrixGenerationSystem(dtype=input.dtype)
        R = omgs(angles,mus)
        grad_input = R @ grad_output # dLdX = dZdX @ dLdZ
        if ctx.needs_input_grad[1]:
            omgs.partial_difference=True
            grad_angles = torch.empty_like(angles,dtype=input.dtype)
            for iAngle in range(len(grad_angles)):
                dRi = omgs(angles,mus,index_pd_angle=iAngle)
                grad_angles[iAngle] = torch.sum(grad_input * (dRi.T @ input))
        grad_mus = torch.zeros_like(mus,dtype=input.dtype)
        return grad_input, grad_angles, grad_mus
