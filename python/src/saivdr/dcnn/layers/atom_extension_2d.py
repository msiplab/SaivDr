import torch
import torch.nn as nn
import torch.autograd as autograd
from nsoltLayerExceptions import InvalidDirection, InvalidTargetChannels

class NsoltAtomExtension2dLayer(nn.Module):
    """
    NSOLTATOMEXTENSION2DLAYER
        コンポーネント別に入力(nComponents=1のみサポート):
            nSamples x nRows x nCols x nChsTotal
    
        コンポーネント別に出力(nComponents=1のみサポート):
            nSamples x nRows x nCols x nChsTotal
    
    Requirements: Python 3.7.x, PyTorch 1.7.x
    
    Copyright (c) 2020-2021, Shogo MURAMATSU
    
    All rights reserved.
    
    Contact address: Shogo MURAMATSU,
                    Faculty of Engineering, Niigata University,
                    8050 2-no-cho Ikarashi, Nishi-ku,
                    Niigata, 950-2181, JAPAN
    
    http://msiplab.eng.niigata-u.ac.jp/
    """
    def __init__(self,
            name='',
            number_of_channels=[],
            direction='',
            target_channels=''):
        super(NsoltAtomExtension2dLayer, self).__init__()
        self.number_of_channels = number_of_channels
        self.name = name

        # Target channels
        if target_channels in { 'Sum', 'Difference' }:
            self.target_channels = target_channels
        else:
            raise InvalidTargetChannels(
                '%s : Target should be either of Sum or Difference'\
                % self.direction
            )

        # Shift direction
        if direction in { 'Right', 'Left', 'Down', 'Up' }:
            self.direction = direction        
        else:
           raise InvalidDirection(
                '%s : Direction should be either of Right, Left, Down or Up'\
                % self.direction
            )

        # Description
        self.description = direction \
            + " shift the " \
            + target_channels.lower() \
            + "-channel Coefs. " \
            + "(ps,pa) = (" \
            + str(number_of_channels[0]) + "," \
            + str(number_of_channels[1]) + ")"
        self.type = ''        

    def forward(self,X):
        # Number of channels
        nchs = torch.tensor(self.number_of_channels,dtype=torch.int)

        # Target channels
        if self.target_channels == 'Difference':
            target = torch.tensor((0,))
        else:
            target = torch.tensor((1,))
        # Shift direction
        if self.direction == 'Right':
            shift = torch.tensor(( 0, 0, 1, 0 ))
        elif self.direction == 'Left':
            shift = torch.tensor(( 0, 0, -1, 0 ))
        elif self.direction == 'Down':
            shift = torch.tensor(( 0, 1, 0, 0 ))
        else:
            shift = torch.tensor(( 0, -1, 0, 0 ))
        # Atom extension function
        atomext = AtomExtension2d.apply

        return atomext(X,nchs,target,shift)

class AtomExtension2d(autograd.Function):
    @staticmethod
    def forward(ctx, input, nchs, target, shift):
        ctx.mark_non_differentiable(nchs,target,shift)
        ctx.save_for_backward(nchs,target,shift)
        # Block butterfly 
        X = block_butterfly(input,nchs)
        # Block shift
        X = block_shift(X,nchs,target,shift)        
        # Block butterfly 
        return block_butterfly(X,nchs)/2.

    @staticmethod
    def backward(ctx, grad_output):
        nchs,target,shift = ctx.saved_tensors
        grad_input = grad_nchs = grad_target = grad_shift = None
        if ctx.needs_input_grad[0]:
            # Block butterfly 
            X = block_butterfly(grad_output,nchs)
            # Block shift
            X = block_shift(X,nchs,target,-shift)
            # Block butterfly 
            grad_input = block_butterfly(X,nchs)/2.
        if ctx.needs_input_grad[1]:
            grad_nchs = torch.zeros_like(nchs)
        if ctx.needs_input_grad[2]:
            grad_target = torch.zeros_like(target)
        if ctx.needs_input_grad[3]:
            grad_shift = torch.zeros_like(shift)
               
        return grad_input, grad_nchs, grad_target, grad_shift

def block_butterfly(X,nchs):
    """
    Block butterfly
    """
    ps = nchs[0]
    Xs = X[:,:,:,:ps]
    Xa = X[:,:,:,ps:]
    return torch.cat((Xs+Xa,Xs-Xa),dim=-1)

def block_shift(X,nchs,target,shift):
    """
    Block shift
    """
    ps = nchs[0]
    if target == 0: # Difference channel
        X[:,:,:,ps:] = torch.roll(X[:,:,:,ps:],shifts=tuple(shift.tolist()),dims=(0,1,2,3))
    else: # Sum channel
        X[:,:,:,:ps] = torch.roll(X[:,:,:,:ps],shifts=tuple(shift.tolist()),dims=(0,1,2,3))
    return X