import torch
import torch.nn as nn
import torch.autograd as autograd
#import numpy as np
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
        dtype=torch.get_default_dtype(),
        device=torch.device("cpu")):

        super(OrthonormalTransform, self).__init__()
        self.dtype = dtype
        self.nPoints = n
        self.device = device

        # Mode
        if mode in {'Analysis','Synthesis'}:
            self.__mode = mode
        else:
            raise InvalidMode(
                '%s : Mode should be either of Analysis or Synthesis'\
                % str(mode)
            )

        # Angles
        nAngs = int(n*(n-1)/2)
        self.angles = nn.Parameter(torch.zeros(nAngs,dtype=self.dtype,device=self.device))

        # Mus
        if torch.is_tensor(mus):
            self.__mus = mus.to(dtype=self.dtype,device=self.device)
        elif mus == 1:
            self.__mus = torch.ones(self.nPoints,dtype=self.dtype,device=self.device)
        elif mus == -1:
            self.__mus = -torch.ones(self.nPoints,dtype=self.dtype,device=self.device)
        else:
            self.__mus = torch.tensor(mus,dtype=self.dtype,device=self.device)
        self.checkMus()

    def forward(self,X):
        angles = self.angles
        mus = self.__mus
        mode = self.__mode
        if mode=='Analysis':
            givensrots = GivensRotations4Analyzer.apply
        else:
            givensrots = GivensRotations4Synthesizer.apply
        return givensrots(X,angles,mus)           

    @property
    def mode(self):
        return self.__mode 

    @mode.setter
    def mode(self,mode):
        if mode in {'Analysis','Synthesis'}:
            self.__mode = mode
        else:
            raise InvalidMode(
                '%s : Mode should be either of Analysis or Synthesis'\
                % str(mode)
            )

    @property 
    def mus(self):
        return self.__mus
    
    @mus.setter
    def mus(self,mus):
        if torch.is_tensor(mus):
            self.__mus = mus.to(dtype=self.dtype,device=self.device)
        elif mus == 1:
            self.__mus = torch.ones(self.nPoints,dtype=self.dtype,device=self.device)
        elif mus == -1:
            self.__mus = -torch.ones(self.nPoints,dtype=self.dtype,device=self.device)
        else:
            self.__mus = torch.tensor(mus,dtype=self.dtype,device=self.device)
        self.checkMus()

    def checkMus(self):
        if torch.not_equal(torch.abs(self.__mus),torch.ones(self.nPoints,device=self.device)).any():
            raise InvalidMus(
                '%s : Elements in mus should be either of 1 or -1'\
                % str(self.__mus)
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
    def forward(ctx, input, angles, mus):
        ctx.mark_non_differentiable(mus)
        ctx.save_for_backward(input,angles,mus)
        omgs = OrthonormalMatrixGenerationSystem(dtype=input.dtype,partial_difference=False)
        R = omgs(angles,mus) #.to(input.device)
        return R @ input
    
    @staticmethod
    def backward(ctx, grad_output):
        input, angles, mus = ctx.saved_tensors
        grad_input = grad_angles = grad_mus = None
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:        
            omgs = OrthonormalMatrixGenerationSystem(dtype=input.dtype,partial_difference=False)
            R = omgs(angles,mus) #.to(input.device)
            dLdX = R.T @ grad_output # dLdX = dZdX @ dLdZ
        # 
        if ctx.needs_input_grad[0]:
            grad_input = dLdX
        if ctx.needs_input_grad[1]:
            omgs.partial_difference=True
            grad_angles = torch.zeros_like(angles,dtype=input.dtype)
            for iAngle in range(len(grad_angles)):
                dRi = omgs(angles,mus,index_pd_angle=iAngle).to(input.device)
                #grad_angles[iAngle] = torch.sum(dLdX * (dRi @ input))
                grad_angles[iAngle] = torch.sum(grad_output * (dRi @ input))
        if ctx.needs_input_grad[2]:
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
        omgs = OrthonormalMatrixGenerationSystem(dtype=input.dtype,partial_difference=False)
        R = omgs(angles,mus) #.to(input.device)
        return R.T @ input
    
    @staticmethod
    def backward(ctx, grad_output):
        input, angles, mus = ctx.saved_tensors
        grad_input = grad_angles = grad_mus = None
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            omgs = OrthonormalMatrixGenerationSystem(dtype=input.dtype,partial_difference=False)
            R = omgs(angles,mus) #.to(input.device)
            dLdX = R @ grad_output # dLdX = dZdX @ dLdZ
        #            
        if ctx.needs_input_grad[0]:
            grad_input = dLdX
        if ctx.needs_input_grad[1]:
            omgs.partial_difference=True
            grad_angles = torch.zeros_like(angles,dtype=input.dtype)
            for iAngle in range(len(grad_angles)):
                dRi = omgs(angles,mus,index_pd_angle=iAngle) #.to(input.device)
                #grad_angles[iAngle] = torch.sum(dLdX * (dRi.T @ input))
                grad_angles[iAngle] = torch.sum(grad_output * (dRi.T @ input))
        if ctx.needs_input_grad[2]:
            grad_mus = torch.zeros_like(mus,dtype=input.dtype)
        return grad_input, grad_angles, grad_mus
