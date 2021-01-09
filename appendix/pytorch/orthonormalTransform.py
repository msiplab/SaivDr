import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
from nsoltUtility import OrthonormalMatrixGenerationSystem
from nsoltLayerExceptions import InvalidMode

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
        self.mus = torch.ones(self.nPoints)

    def forward(self,X):
        angles = self.angles
        if self.mode=='Analysis':
            givensrots = GivensRotations4Analyzer.apply
            return self.mus.view(-1,1) * givensrots(X,angles)            
        elif self.mode=='Synthesis':
            givensrots = GivensRotations4Synthesizer.apply
            return givensrots(self.mus.view(-1,1) * X,angles)
        else:
            raise InvalidMode(
                '%s : Mode should be either of Analysis or Synthesis'\
                % str(mode)
            )

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
    def forward(ctx, input, angles):
        ctx.save_for_backward(input, angles)
        omgs = OrthonormalMatrixGenerationSystem(dtype=input.dtype)
        R = omgs(angles)
        return R @ input
    
    @staticmethod
    def backward(ctx, grad_output):
        input, angles = ctx.saved_tensors
        omgs = OrthonormalMatrixGenerationSystem(dtype=input.dtype)
        R = omgs(angles)
        grad_input = R.T @ grad_output # dLdX = dZdX @ dLdZ
        if ctx.needs_input_grad[1]:
            omgs.partial_difference=True
            grad_angles = torch.empty_like(angles,dtype=input.dtype)
            for iAngle in range(len(grad_angles)):
                dRi = omgs(angles,index_pd_angle=iAngle)
                grad_angles[iAngle] = torch.sum(grad_input * (dRi @ input))
        return grad_input, grad_angles

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
    def forward(ctx, input, angles):
        ctx.save_for_backward(input, angles)
        omgs = OrthonormalMatrixGenerationSystem(dtype=input.dtype)
        R = omgs(angles)
        return R.T @ input
    
    @staticmethod
    def backward(ctx, grad_output):
        input, angles = ctx.saved_tensors
        omgs = OrthonormalMatrixGenerationSystem(dtype=input.dtype)
        R = omgs(angles)
        grad_input = R @ grad_output # dLdX = dZdX @ dLdZ
        if ctx.needs_input_grad[1]:
            omgs.partial_difference=True
            grad_angles = torch.empty_like(angles,dtype=input.dtype)
            for iAngle in range(len(grad_angles)):
                dRi = omgs(angles,index_pd_angle=iAngle)
                grad_angles[iAngle] = torch.sum(grad_input * (dRi.T @ input))
        return grad_input, grad_angles
