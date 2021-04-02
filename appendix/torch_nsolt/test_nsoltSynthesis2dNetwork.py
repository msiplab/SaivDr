import itertools
import unittest
from parameterized import parameterized
import math
import random
import torch
import torch.nn as nn
#import torch_dct as dct
from orthonormalTransform import OrthonormalTransform
from nsoltUtility import OrthonormalMatrixGenerationSystem
from nsoltSynthesis2dNetwork import NsoltSynthesis2dNetwork
from nsoltLayerExceptions import InvalidNumberOfChannels, InvalidPolyPhaseOrder, InvalidNumberOfVanishingMoments, InvalidNumberOfLevels
from nsoltUtility import Direction, idct_2d

nchs = [ [2, 2], [3, 3], [4, 4] ]
stride = [ [1, 1], [1, 2], [2, 1], [2, 2] ]
ppord = [ [0, 0], [0, 2], [2, 0], [2, 2], [4, 4] ]
datatype = [ torch.float, torch.double ]
height = [ 8, 16, 32 ]
width = [ 8, 16, 32 ]
nvm = [ 0, 1 ]
nlevels = [ 1, 2, 3 ]
isdevicetest = True

class NsoltSynthesis2dNetworkTestCase(unittest.TestCase):
    """
    NSOLTSYNHESIS2DNETWORKTESTCASE Test cases for NsoltSynthesis2dNetwork
    
    Requirements: Python 3.7.x, PyTorch 1.7.x
    
    Copyright (c) 2021, Yasas Dulanjaya and Shogo MURAMATSU
    
    All rights reserved.
    
    Contact address: Shogo MURAMATSU,
        Faculty of Engineering, Niigata University,
        8050 2-no-cho Ikarashi, Nishi-ku,
        Niigata, 950-2181, JAPAN
    
        http://msiplab.eng.niigata-u.ac.jp/
    """
    @parameterized.expand(
        list(itertools.product(nchs,stride))
    )
    def testConstructor(self,
        nchs, stride):

        # Expcted values
        expctdNchs = nchs
        expctdStride = stride
        expctdPpord = [0,0]        
        expctdNvms = 1

        # Instantiation of target class
        network = NsoltSynthesis2dNetwork(
            number_of_channels=nchs,
            decimation_factor=stride
        )

        # Actual values
        actualNchs = network.number_of_channels
        actualStride = network.decimation_factor
        actualPpord = network.polyphase_order        
        actualNvms = network.number_of_vanishing_moments

        # Evaluation
        self.assertTrue(isinstance(network, nn.Module))
        self.assertEqual(actualNchs,expctdNchs)
        self.assertEqual(actualStride,expctdStride)
        self.assertEqual(actualPpord,expctdPpord)        
        self.assertEqual(actualNvms,expctdNvms)        

    @parameterized.expand(
        list(itertools.product(nchs,stride,height,width,datatype))
    )
    def testForwardGrayScale(self,
            nchs,stride, height, width, datatype):
        rtol,atol = 1e-5,1e-8
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")       

        # Parameters
        nSamples = 8
        nrows = int(math.ceil(height/stride[Direction.VERTICAL]))
        ncols = int(math.ceil(width/stride[Direction.HORIZONTAL]))
        nComponents = 1
        nDecs = stride[0]*stride[1] #math.prod(stride)
        nChsTotal = sum(nchs)

        # nSamples x nRows x nCols x nChsTotal
        X = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype)
        X = X.to(device)
        
        # Expected values        
        # nSamples x nRows x nCols x nDecs
        ps,pa = nchs
        W0T = torch.eye(ps,dtype=datatype).to(device)
        U0T = torch.eye(pa,dtype=datatype).to(device)
        Ys = X[:,:,:,:ps].view(-1,ps).T
        Ya = X[:,:,:,ps:].view(-1,pa).T
        ms,ma = int(math.ceil(nDecs/2.)),int(math.floor(nDecs/2.))        
        Zsa = torch.cat(
                ( W0T[:ms,:] @ Ys, 
                  U0T[:ma,:] @ Ya ),dim=0)
        V = Zsa.T.view(nSamples,nrows,ncols,nDecs)
        A = permuteIdctCoefs_(V,stride)
        Y = idct_2d(A)
        expctdZ = Y.reshape(nSamples,nComponents,height,width)
        
        # Instantiation of target class
        network = NsoltSynthesis2dNetwork(
            number_of_channels=nchs,
            decimation_factor=stride
        )
        network = network.to(device)

        # Actual values
        with torch.no_grad():
            actualZ = network.forward(X)

        # Evaluation
        self.assertEqual(actualZ.dtype,datatype)
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))
        self.assertFalse(actualZ.requires_grad)

    @parameterized.expand(  
        list(itertools.product(nchs,stride))
    )
    def testNumberOfChannelsException(self,
        nchs,stride):
        ps,pa = nchs
        with self.assertRaises(InvalidNumberOfChannels):
            NsoltSynthesis2dNetwork(
                number_of_channels = [ps,ps+1],
                decimation_factor = stride
            )

        with self.assertRaises(InvalidNumberOfChannels):
            NsoltSynthesis2dNetwork(
                number_of_channels = [pa+1,pa],
                decimation_factor = stride
            )

    @parameterized.expand(
        list(itertools.product(nchs,stride,ppord))
    )
    def testNumberOfPolyPhaseOrderException(self,
        nchs,stride,ppord):
        with self.assertRaises(InvalidPolyPhaseOrder):
            NsoltSynthesis2dNetwork(
                polyphase_order = [ ppord[0]+1, ppord[1] ],
                number_of_channels = nchs,
                decimation_factor = stride
            )

        with self.assertRaises(InvalidPolyPhaseOrder):
            NsoltSynthesis2dNetwork(
                polyphase_order = [ ppord[0], ppord[1]+1 ],
                number_of_channels = nchs,
                decimation_factor = stride
            )

        with self.assertRaises(InvalidPolyPhaseOrder):
            NsoltSynthesis2dNetwork(
                polyphase_order = [ ppord[0]+1, ppord[1]+1 ],
                number_of_channels = nchs,
                decimation_factor = stride
            )   

    @parameterized.expand(
        list(itertools.product(nchs,stride,ppord))
    )
    def testNumberOfVanishingMomentsException(self,
        nchs,stride,ppord):
        nVm = -1
        with self.assertRaises(InvalidNumberOfVanishingMoments):
            NsoltSynthesis2dNetwork(
                number_of_channels = nchs,
                decimation_factor = stride,
                polyphase_order = ppord,
                number_of_vanishing_moments = nVm
            )

        nVm = 2
        with self.assertRaises(InvalidNumberOfVanishingMoments):
            NsoltSynthesis2dNetwork(
                number_of_channels = nchs,
                decimation_factor = stride,
                polyphase_order = ppord,
                number_of_vanishing_moments = nVm
            )

    @parameterized.expand(
        list(itertools.product(nchs,stride,ppord))
    )
    def testNumberOfLevelsException(self,
        nchs,stride,ppord):
        nlevels = -1
        with self.assertRaises(InvalidNumberOfLevels):
            NsoltSynthesis2dNetwork(
                number_of_channels = nchs,
                decimation_factor = stride,
                polyphase_order = ppord,
                number_of_levels = nlevels
            )

        nlevels = 0.5
        with self.assertRaises(InvalidNumberOfLevels):
            NsoltSynthesis2dNetwork(
                number_of_channels = nchs,
                decimation_factor = stride,
                polyphase_order = ppord,
                number_of_levels = nlevels
            )

    @parameterized.expand(
        list(itertools.product(nchs,stride,height,width,datatype))
    )
    def testForwardGrayScaleWithInitialization(self,
            nchs,stride, height, width, datatype):
        rtol,atol = 1e-3,1e-6
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")                 
        gen = OrthonormalMatrixGenerationSystem(dtype=datatype)

        # Initialization function of angle parameters
        angle0 = 2.0*math.pi*random.random()
        def init_angles(m):
            if type(m) == OrthonormalTransform:
                torch.nn.init.constant_(m.angles,angle0)

        # Parameters
        nVm = 0
        nSamples = 8
        nrows = int(math.ceil(height/stride[Direction.VERTICAL]))
        ncols = int(math.ceil(width/stride[Direction.HORIZONTAL]))
        nComponents = 1
        nDecs = stride[0]*stride[1] #math.prod(stride)
        nChsTotal = sum(nchs)

        # nSamples x nRows x nCols x nChsTotal
        X = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype,device=device,requires_grad=True)
        
        # Expected values        
        # nSamples x nRows x nCols x nDecs
        ps,pa = nchs
        angles = angle0*torch.ones(int((nChsTotal-2)*nChsTotal/4)) #,dtype=datatype)
        nAngsW = int(len(angles)/2)
        angsW,angsU = angles[:nAngsW],angles[nAngsW:]
        W0T,U0T = gen(angsW).T.to(device),gen(angsU).T.to(device)
        Ys = X[:,:,:,:ps].view(-1,ps).T
        Ya = X[:,:,:,ps:].view(-1,pa).T
        ms,ma = int(math.ceil(nDecs/2.)),int(math.floor(nDecs/2.))        
        Zsa = torch.cat(
                ( W0T[:ms,:] @ Ys, 
                  U0T[:ma,:] @ Ya ),dim=0)
        V = Zsa.T.view(nSamples,nrows,ncols,nDecs)
        A = permuteIdctCoefs_(V,stride)
        Y = idct_2d(A)
        expctdZ = Y.reshape(nSamples,nComponents,height,width)
        
        # Instantiation of target class
        network = NsoltSynthesis2dNetwork(
            number_of_channels=nchs,
            decimation_factor=stride,
            number_of_vanishing_moments=nVm
        )
        network = network.to(device)

        # Initialization of angle parameters
        network.apply(init_angles)

        # Actual values
        with torch.no_grad():
            actualZ = network.forward(X)

        # Evaluation
        self.assertEqual(actualZ.dtype,datatype)
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))
        self.assertFalse(actualZ.requires_grad)

    @parameterized.expand(
        list(itertools.product(nchs,stride,height,width,datatype))
    )
    def testForwardGrayScaleOrd22(self,
            nchs, stride, height, width, datatype):
        rtol,atol = 1e-3,1e-6
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")          

        # Parameters
        ppOrd = [ 2, 2 ]
        nSamples = 8
        nrows = int(math.ceil(height/stride[Direction.VERTICAL]))
        ncols = int(math.ceil(width/stride[Direction.HORIZONTAL]))
        nComponents = 1
        nDecs = stride[0]*stride[1] #math.prod(stride)
        nChsTotal = sum(nchs)

        # nSamples x nRows x nCols x nChsTotal
        X = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype,device=device,requires_grad=True)

        # Expected values        
        # nSamples x nRows x nCols x nDecs
        ps,pa = nchs
        Z = X
        # Vertical atom concatenation
        Uv2T = -torch.eye(pa,dtype=datatype).to(device)
        Z = intermediate_rotation(Z,nchs,Uv2T)
        Z = block_butterfly(Z,nchs)
        Z = block_shift(Z,nchs,1,[0,1,0,0]) # target=sum, shift=down
        Z = block_butterfly(Z,nchs)/2.
        Uv1T = -torch.eye(pa,dtype=datatype).to(device)
        Z = intermediate_rotation(Z,nchs,Uv1T)
        Z = block_butterfly(Z,nchs)
        Z = block_shift(Z,nchs,0,[0,-1,0,0]) # target=diff, shift=up
        Z = block_butterfly(Z,nchs)/2.
        # Horizontal atom concatenation
        Uh2T = -torch.eye(pa,dtype=datatype).to(device)
        Z = intermediate_rotation(Z,nchs,Uh2T)
        Z = block_butterfly(Z,nchs)
        Z = block_shift(Z,nchs,1,[0,0,1,0]) # target=sum, shift=right
        Z = block_butterfly(Z,nchs)/2.
        Uh1T = -torch.eye(pa,dtype=datatype).to(device)
        Z = intermediate_rotation(Z,nchs,Uh1T)
        Z = block_butterfly(Z,nchs)
        Z = block_shift(Z,nchs,0,[0,0,-1,0]) # target=diff, shift=left
        Z = block_butterfly(Z,nchs)/2.
        # Final rotation
        W0T = torch.eye(ps,dtype=datatype).to(device)
        U0T = torch.eye(pa,dtype=datatype).to(device)
        Ys = Z[:,:,:,:ps].view(-1,ps).T
        Ya = Z[:,:,:,ps:].view(-1,pa).T
        ms,ma = int(math.ceil(nDecs/2.)),int(math.floor(nDecs/2.))        
        Zsa = torch.cat(
                ( W0T[:ms,:] @ Ys, 
                  U0T[:ma,:] @ Ya ),dim=0)
        V = Zsa.T.view(nSamples,nrows,ncols,nDecs)
        A = permuteIdctCoefs_(V,stride)
        Y = idct_2d(A)
        expctdZ = Y.reshape(nSamples,nComponents,height,width)
        
        # Instantiation of target class
        network = NsoltSynthesis2dNetwork(
            number_of_channels=nchs,
            decimation_factor=stride,
            polyphase_order=ppOrd
        )
        network = network.to(device)

        # Actual values
        with torch.no_grad():
            actualZ = network.forward(X)

        # Evaluation
        self.assertEqual(actualZ.dtype,datatype)
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))
        self.assertFalse(actualZ.requires_grad)

    @parameterized.expand(
        list(itertools.product(nchs,stride,height,width,datatype))
    )
    def testForwardGrayScaleOrd20(self,
            nchs, stride, height, width, datatype):
        rtol,atol = 1e-3,1e-6
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")          

        # Parameters
        ppOrd = [ 2, 0 ]
        nSamples = 8
        nrows = int(math.ceil(height/stride[Direction.VERTICAL]))
        ncols = int(math.ceil(width/stride[Direction.HORIZONTAL]))
        nComponents = 1
        nDecs = stride[0]*stride[1] #math.prod(stride)
        nChsTotal = sum(nchs)

        # nSamples x nRows x nCols x nChsTotal
        X = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype,device=device,requires_grad=True)

        # Expected values        
        # nSamples x nRows x nCols x nDecs
        ps,pa = nchs
        Z = X
        # Vertical atom concatenation
        Uv2T = -torch.eye(pa,dtype=datatype).to(device)
        Z = intermediate_rotation(Z,nchs,Uv2T)
        Z = block_butterfly(Z,nchs)
        Z = block_shift(Z,nchs,1,[0,1,0,0]) # target=sum, shift=down
        Z = block_butterfly(Z,nchs)/2.
        Uv1T = -torch.eye(pa,dtype=datatype).to(device)
        Z = intermediate_rotation(Z,nchs,Uv1T)
        Z = block_butterfly(Z,nchs)
        Z = block_shift(Z,nchs,0,[0,-1,0,0]) # target=diff, shift=up
        Z = block_butterfly(Z,nchs)/2.
        # Final rotation
        W0T = torch.eye(ps,dtype=datatype).to(device)
        U0T = torch.eye(pa,dtype=datatype).to(device)
        Ys = Z[:,:,:,:ps].view(-1,ps).T
        Ya = Z[:,:,:,ps:].view(-1,pa).T
        ms,ma = int(math.ceil(nDecs/2.)),int(math.floor(nDecs/2.))        
        Zsa = torch.cat(
                ( W0T[:ms,:] @ Ys, 
                  U0T[:ma,:] @ Ya ),dim=0)
        V = Zsa.T.view(nSamples,nrows,ncols,nDecs)
        A = permuteIdctCoefs_(V,stride)
        Y = idct_2d(A)
        expctdZ = Y.reshape(nSamples,nComponents,height,width)
        
        # Instantiation of target class
        network = NsoltSynthesis2dNetwork(
            number_of_channels=nchs,
            decimation_factor=stride,
            polyphase_order=ppOrd
        )
        network = network.to(device)

        # Actual values
        with torch.no_grad():
            actualZ = network.forward(X)

        # Evaluation
        self.assertEqual(actualZ.dtype,datatype)
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))
        self.assertFalse(actualZ.requires_grad)

    @parameterized.expand(
        list(itertools.product(nchs,stride,height,width,datatype))
    )
    def testForwardGrayScaleOrd02(self,
            nchs, stride, height, width, datatype):
        rtol,atol = 1e-3,1e-6
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")            

        # Parameters
        ppOrd = [ 0, 2 ]
        nSamples = 8
        nrows = int(math.ceil(height/stride[Direction.VERTICAL]))
        ncols = int(math.ceil(width/stride[Direction.HORIZONTAL]))
        nComponents = 1
        nDecs = stride[0]*stride[1] #math.prod(stride)
        nChsTotal = sum(nchs)

        # nSamples x nRows x nCols x nChsTotal
        X = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype,device=device,requires_grad=True)

        # Expected values        
        # nSamples x nRows x nCols x nDecs
        ps,pa = nchs
        Z = X
        # Horizontal atom concatenation
        Uh2T = -torch.eye(pa,dtype=datatype).to(device)
        Z = intermediate_rotation(Z,nchs,Uh2T)
        Z = block_butterfly(Z,nchs)
        Z = block_shift(Z,nchs,1,[0,0,1,0]) # target=sum, shift=right
        Z = block_butterfly(Z,nchs)/2.
        Uh1T = -torch.eye(pa,dtype=datatype).to(device)
        Z = intermediate_rotation(Z,nchs,Uh1T)
        Z = block_butterfly(Z,nchs)
        Z = block_shift(Z,nchs,0,[0,0,-1,0]) # target=diff, shift=left
        Z = block_butterfly(Z,nchs)/2.
        # Final rotation
        W0T = torch.eye(ps,dtype=datatype).to(device)
        U0T = torch.eye(pa,dtype=datatype).to(device)
        Ys = Z[:,:,:,:ps].view(-1,ps).T
        Ya = Z[:,:,:,ps:].view(-1,pa).T
        ms,ma = int(math.ceil(nDecs/2.)),int(math.floor(nDecs/2.))        
        Zsa = torch.cat(
                ( W0T[:ms,:] @ Ys, 
                  U0T[:ma,:] @ Ya ),dim=0)
        V = Zsa.T.view(nSamples,nrows,ncols,nDecs)
        A = permuteIdctCoefs_(V,stride)
        Y = idct_2d(A)
        expctdZ = Y.reshape(nSamples,nComponents,height,width)
        
        # Instantiation of target class
        network = NsoltSynthesis2dNetwork(
            number_of_channels=nchs,
            decimation_factor=stride,
            polyphase_order=ppOrd
        )
        network = network.to(device)

        # Actual values
        with torch.no_grad():
            actualZ = network.forward(X)

        # Evaluation
        self.assertEqual(actualZ.dtype,datatype)
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))
        self.assertFalse(actualZ.requires_grad)

    @parameterized.expand(
        list(itertools.product(nchs,stride,ppord,datatype))
    )
    def testForwardGrayScaleOverlapping(self,
            nchs, stride, ppord, datatype):
        rtol,atol = 1e-3,1e-6
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")       

        # Parameters
        height = 8
        width = 16
        ppOrd = ppord
        nSamples = 8
        nrows = int(math.ceil(height/stride[Direction.VERTICAL]))
        ncols = int(math.ceil(width/stride[Direction.HORIZONTAL]))
        nComponents = 1
        nDecs = stride[0]*stride[1] #math.prod(stride)
        nChsTotal = sum(nchs)

        # nSamples x nRows x nCols x nChsTotal
        X = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype,device=device,requires_grad=True)
        
        # Expected values        
        # nSamples x nRows x nCols x nDecs
        ps,pa = nchs
        Z = X
        # Vertical atom concatenation
        for ordV in range(int(ppOrd[Direction.VERTICAL]/2)):
            Uv2T = -torch.eye(pa,dtype=datatype).to(device)
            Z = intermediate_rotation(Z,nchs,Uv2T)
            Z = block_butterfly(Z,nchs)
            Z = block_shift(Z,nchs,1,[0,1,0,0]) # target=sum, shift=down
            Z = block_butterfly(Z,nchs)/2.
            Uv1T = -torch.eye(pa,dtype=datatype).to(device)
            Z = intermediate_rotation(Z,nchs,Uv1T)
            Z = block_butterfly(Z,nchs)
            Z = block_shift(Z,nchs,0,[0,-1,0,0]) # target=diff, shift=up
            Z = block_butterfly(Z,nchs)/2.
        # Horizontal atom concatenation
        for ordH in range(int(ppOrd[Direction.HORIZONTAL]/2)):
            Uh2T = -torch.eye(pa,dtype=datatype).to(device)
            Z = intermediate_rotation(Z,nchs,Uh2T)
            Z = block_butterfly(Z,nchs)
            Z = block_shift(Z,nchs,1,[0,0,1,0]) # target=sum, shift=right
            Z = block_butterfly(Z,nchs)/2.
            Uh1T = -torch.eye(pa,dtype=datatype).to(device)
            Z = intermediate_rotation(Z,nchs,Uh1T)
            Z = block_butterfly(Z,nchs)
            Z = block_shift(Z,nchs,0,[0,0,-1,0]) # target=diff, shift=left
            Z = block_butterfly(Z,nchs)/2.
        # Final rotation
        W0T = torch.eye(ps,dtype=datatype).to(device)
        U0T = torch.eye(pa,dtype=datatype).to(device)
        Ys = Z[:,:,:,:ps].view(-1,ps).T
        Ya = Z[:,:,:,ps:].view(-1,pa).T
        ms,ma = int(math.ceil(nDecs/2.)),int(math.floor(nDecs/2.))        
        Zsa = torch.cat(
                ( W0T[:ms,:] @ Ys, 
                  U0T[:ma,:] @ Ya ),dim=0)
        V = Zsa.T.view(nSamples,nrows,ncols,nDecs)
        A = permuteIdctCoefs_(V,stride)
        Y = idct_2d(A)
        expctdZ = Y.reshape(nSamples,nComponents,height,width)
        
        # Instantiation of target class
        network = NsoltSynthesis2dNetwork(
            number_of_channels=nchs,
            decimation_factor=stride,
            polyphase_order=ppOrd
        )
        network = network.to(device)

        # Actual values
        with torch.no_grad():
            actualZ = network.forward(X)

        # Evaluation
        self.assertEqual(actualZ.dtype,datatype)
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))
        self.assertFalse(actualZ.requires_grad)

    @parameterized.expand(
        list(itertools.product(nchs,stride,ppord,datatype))
    )
    def testForwardGrayScaleOverlappingWithInitalization(self,
            nchs, stride, ppord, datatype):
        rtol,atol = 1e-3,1e-6
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")           
        gen = OrthonormalMatrixGenerationSystem(dtype=datatype)

        # Initialization function of angle parameters
        angle0 = 2.0*math.pi*random.random()
        def init_angles(m):
            if type(m) == OrthonormalTransform:
                torch.nn.init.constant_(m.angles,angle0)

        # Parameters
        nVm = 0
        height = 8
        width = 16
        ppOrd = ppord
        nSamples = 8
        nrows = int(math.ceil(height/stride[Direction.VERTICAL]))
        ncols = int(math.ceil(width/stride[Direction.HORIZONTAL]))
        nComponents = 1
        nDecs = stride[0]*stride[1] #math.prod(stride)
        nChsTotal = sum(nchs)

        # nSamples x nRows x nCols x nChsTotal
        X = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype,device=device,requires_grad=True)

        # Expected values        
        # nSamples x nRows x nCols x nDecs
        ps,pa = nchs
        angles = angle0*torch.ones(int((nChsTotal-2)*nChsTotal/4)).to(device) #,dtype=datatype)
        nAngsW = int(len(angles)/2)
        angsW,angsU = angles[:nAngsW],angles[nAngsW:]
        Z = X
        # Vertical atom concatenation
        for ordV in range(int(ppOrd[Direction.VERTICAL]/2)):
            Uv2T = -gen(angsU).T
            Z = intermediate_rotation(Z,nchs,Uv2T)
            Z = block_butterfly(Z,nchs)
            Z = block_shift(Z,nchs,1,[0,1,0,0]) # target=sum, shift=down
            Z = block_butterfly(Z,nchs)/2.
            Uv1T = -gen(angsU).T
            Z = intermediate_rotation(Z,nchs,Uv1T)
            Z = block_butterfly(Z,nchs)
            Z = block_shift(Z,nchs,0,[0,-1,0,0]) # target=diff, shift=up
            Z = block_butterfly(Z,nchs)/2.
        # Horizontal atom concatenation
        for ordH in range(int(ppOrd[Direction.HORIZONTAL]/2)):
            Uh2T = -gen(angsU).T
            Z = intermediate_rotation(Z,nchs,Uh2T)
            Z = block_butterfly(Z,nchs)
            Z = block_shift(Z,nchs,1,[0,0,1,0]) # target=sum, shift=right
            Z = block_butterfly(Z,nchs)/2.
            Uh1T = -gen(angsU).T
            Z = intermediate_rotation(Z,nchs,Uh1T)
            Z = block_butterfly(Z,nchs)
            Z = block_shift(Z,nchs,0,[0,0,-1,0]) # target=diff, shift=left
            Z = block_butterfly(Z,nchs)/2.
        # Final rotation
        W0T,U0T = gen(angsW).T,gen(angsU).T        
        Ys = Z[:,:,:,:ps].view(-1,ps).T
        Ya = Z[:,:,:,ps:].view(-1,pa).T
        ms,ma = int(math.ceil(nDecs/2.)),int(math.floor(nDecs/2.))        
        Zsa = torch.cat(
                ( W0T[:ms,:] @ Ys, 
                  U0T[:ma,:] @ Ya ),dim=0)
        V = Zsa.T.view(nSamples,nrows,ncols,nDecs)
        A = permuteIdctCoefs_(V,stride)
        Y = idct_2d(A)
        expctdZ = Y.reshape(nSamples,nComponents,height,width)
        
        # Instantiation of target class
        network = NsoltSynthesis2dNetwork(
            number_of_channels=nchs,
            decimation_factor=stride,
            polyphase_order=ppOrd,
            number_of_vanishing_moments=nVm
        )
        network = network.to(device)

        # Initialization of angle parameters
        network.apply(init_angles)

        # Actual values
        with torch.no_grad():
            actualZ = network.forward(X)

        # Evaluation
        self.assertEqual(actualZ.dtype,datatype)
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))
        self.assertFalse(actualZ.requires_grad)

    @parameterized.expand(
        list(itertools.product(nchs,stride,ppord,datatype))
    )
    def testForwardGrayScaleOverlappingWithNoDcLeackage(self,
            nchs, stride, ppord, datatype):
        rtol,atol = 1e-3,1e-6
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")            
        #gen = OrthonormalMatrixGenerationSystem(dtype=datatype)

        # Initialization function of angle parameters
        def init_angles(m):
            if type(m) == OrthonormalTransform:
                torch.nn.init.normal_(m.angles)

        # Parameters
        nVm = 1
        height = 8
        width = 16
        ppOrd = ppord
        nSamples = 8
        nrows = int(math.ceil(height/stride[Direction.VERTICAL]))
        ncols = int(math.ceil(width/stride[Direction.HORIZONTAL]))
        nComponents = 1
        nDecs = stride[0]*stride[1] #math.prod(stride)
        nChsTotal = sum(nchs)

        # nSamples x nRows x nCols x nChsTotal
        X = torch.cat(
            [math.sqrt(nDecs)*torch.ones(nSamples,nrows,ncols,1,dtype=datatype,device=device,requires_grad=True),
            torch.zeros(nSamples,nrows,ncols,nChsTotal-1,dtype=datatype,device=device,requires_grad=True)],
            dim=3)

        # Expected values        
        # nSamples x nRows x nCols x nDecs
        expctdZ = torch.ones(nSamples,nComponents,height,width,dtype=datatype).to(device)
        
        # Instantiation of target class
        network = NsoltSynthesis2dNetwork(
            number_of_channels=nchs,
            decimation_factor=stride,
            polyphase_order=ppOrd,
            number_of_vanishing_moments=nVm
        )
        network = network.to(device)

        # Initialization of angle parameters
        network.apply(init_angles)

        # Actual values
        with torch.no_grad():
            actualZ = network.forward(X)

        # Evaluation
        self.assertEqual(actualZ.dtype,datatype)
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))
        self.assertFalse(actualZ.requires_grad)    
        
    @parameterized.expand(
        list(itertools.product(nchs,stride,nvm,nlevels,datatype))
    )
    def testForwardGrayScaleMultiLevels(self,
            nchs, stride, nvm, nlevels, datatype):
        rtol,atol = 1e-3,1e-6
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")      
        gen = OrthonormalMatrixGenerationSystem(dtype=datatype)

        # Initialization function of angle parameters
        angle0 = 2.0*math.pi*random.random()
        def init_angles(m):
            if type(m) == OrthonormalTransform:
                torch.nn.init.constant_(m.angles,angle0)

        # Parameters
        nVm = nvm
        height = 8
        width = 16
        ppOrd = [ 2, 2 ]
        nSamples = 8
        nrows = int(math.ceil(height/(stride[Direction.VERTICAL]**nlevels)))
        ncols = int(math.ceil(width/(stride[Direction.HORIZONTAL]**nlevels)))        
        nComponents = 1
        nDecs = stride[0]*stride[1] #math.prod(stride)
        nChsTotal = sum(nchs)

        # nSamples x nRows x nCols x nChsTotal
        nrows_ = nrows
        ncols_ = ncols
        X = []
        for iLevel in range(1,nlevels+1):
            if iLevel == 1:
                X.append(torch.randn(nSamples,nrows_,ncols_,1,dtype=datatype,device=device,requires_grad=True)) 
            X.append(torch.randn(nSamples,nrows_,ncols_,nChsTotal-1,dtype=datatype,device=device,requires_grad=True))     
            nrows_ *= stride[Direction.VERTICAL]
            ncols_ *= stride[Direction.HORIZONTAL]
        X = tuple(X)

        # Expected values        
        # nSamples x nRows x nCols x nDecs
        ps,pa = nchs
        # Multi-level reconstruction
        for iLevel in range(nlevels,0,-1):
            angles = angle0*torch.ones(int((nChsTotal-2)*nChsTotal/4)).to(device) #,dtype=datatype)
            nAngsW = int(len(angles)/2)
            angsW,angsU = angles[:nAngsW],angles[nAngsW:]
            angsW,angsU = angles[:nAngsW],angles[nAngsW:]
            if nVm > 0:
                angsW[:(ps-1)] = torch.zeros_like(angsW[:(ps-1)])
            # Extract scale channel
            if iLevel == nlevels:
                Xdc = X[0]
            Xac = X[nlevels-iLevel+1]
            Z = torch.cat((Xdc,Xac),dim=3)
            # Vertical atom concatenation
            for ordV in range(int(ppOrd[Direction.VERTICAL]/2)):
                Uv2T = -gen(angsU).T
                Z = intermediate_rotation(Z,nchs,Uv2T)
                Z = block_butterfly(Z,nchs)
                Z = block_shift(Z,nchs,1,[0,1,0,0]) # target=sum, shift=down
                Z = block_butterfly(Z,nchs)/2.
                Uv1T = -gen(angsU).T
                Z = intermediate_rotation(Z,nchs,Uv1T)
                Z = block_butterfly(Z,nchs)
                Z = block_shift(Z,nchs,0,[0,-1,0,0]) # target=diff, shift=up
                Z = block_butterfly(Z,nchs)/2.
            # Horizontal atom concatenation
            for ordH in range(int(ppOrd[Direction.HORIZONTAL]/2)):
                Uh2T = -gen(angsU).T
                Z = intermediate_rotation(Z,nchs,Uh2T)
                Z = block_butterfly(Z,nchs)
                Z = block_shift(Z,nchs,1,[0,0,1,0]) # target=sum, shift=right
                Z = block_butterfly(Z,nchs)/2.
                Uh1T = -gen(angsU).T
                Z = intermediate_rotation(Z,nchs,Uh1T)
                Z = block_butterfly(Z,nchs)
                Z = block_shift(Z,nchs,0,[0,0,-1,0]) # target=diff, shift=left
                Z = block_butterfly(Z,nchs)/2.
            # Final rotation
            W0T,U0T = gen(angsW).T,gen(angsU).T        
            Ys = Z[:,:,:,:ps].view(-1,ps).T
            Ya = Z[:,:,:,ps:].view(-1,pa).T
            ms,ma = int(math.ceil(nDecs/2.)),int(math.floor(nDecs/2.))        
            Zsa = torch.cat(
                    ( W0T[:ms,:] @ Ys, 
                      U0T[:ma,:] @ Ya ),dim=0)
            V = Zsa.T.view(nSamples,nrows,ncols,nDecs)
            A = permuteIdctCoefs_(V,stride)
            Y = idct_2d(A)
            # Update
            nrows *= stride[Direction.VERTICAL]
            ncols *= stride[Direction.HORIZONTAL]            
            Xdc = Y.reshape(nSamples,nrows,ncols,1)
        expctdZ = Xdc.view(nSamples,nComponents,height,width)
        
        # Instantiation of target class
        network = NsoltSynthesis2dNetwork(
            number_of_channels=nchs,
            decimation_factor=stride,
            polyphase_order=ppOrd,
            number_of_vanishing_moments=nVm,
            number_of_levels=nlevels
        )
        network = network.to(device)

        # Initialization of angle parameters
        network.apply(init_angles)

        # Actual values
        with torch.no_grad():
            actualZ = network.forward(X)

        # Evaluation
        self.assertEqual(actualZ.dtype,datatype)
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))
        self.assertFalse(actualZ.requires_grad)

    @parameterized.expand(
        list(itertools.product(nchs,stride,nvm,nlevels,datatype))        
    )
    def testBackwardGrayScale(self,
        nchs,stride,nvm,nlevels,datatype):
        rtol,atol = 1e-3,1e-6
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")              

        # Initialization function of angle parameters
        def init_angles(m):
            if type(m) == OrthonormalTransform:
                torch.nn.init.zeros_(m.angles)             

        # Parameters
        nVm = nvm
        height = 8 
        width = 16
        ppOrd = [ 2, 2 ]
        nSamples = 8
        nrows = int(math.ceil(height/(stride[Direction.VERTICAL]**nlevels)))
        ncols = int(math.ceil(width/(stride[Direction.HORIZONTAL]**nlevels)))        
        nComponents = 1
        nDecs = stride[0]*stride[1] #math.prod(stride)
        nChsTotal = sum(nchs)

        # Coefficients nSamples x nRows x nCols x nChsTotal
        nrows_ = nrows
        ncols_ = ncols
        X = []
        for iLevel in range(1,nlevels+1):
            if iLevel == 1:
                X.append(torch.randn(nSamples,nrows_,ncols_,dtype=datatype,device=device,requires_grad=True)) 
            X.append(torch.randn(nSamples,nrows_,ncols_,nChsTotal-1,dtype=datatype,device=device,requires_grad=True))     
            nrows_ *= stride[Direction.VERTICAL]
            ncols_ *= stride[Direction.HORIZONTAL]
        X = tuple(X)

        # Source (nSamples x nComponents x ((Stride[0]**nlevels) x nRows) x ((Stride[1]**nlevels) x nCols))
        dLdZ = torch.randn(nSamples,nComponents,height,width,dtype=datatype,device=device)

        # Instantiation of target class
        network = NsoltSynthesis2dNetwork(
                number_of_channels=nchs,
                decimation_factor=stride,
                polyphase_order=ppOrd,
                number_of_vanishing_moments=nVm,
                number_of_levels=nlevels
            ).to(device)

        # Initialization of angle parameters
        network.apply(init_angles)

        # Expected values
        adjoint = network.T
        expctddLdX = adjoint(dLdZ)
        
        # Actual values
        Z = network(X)
        Z.backward(dLdZ,retain_graph=True)
        actualdLdX = []
        for iCh in range(len(X)):
            actualdLdX.append(X[iCh].grad)

        # Evaluation
        for iCh in range(len(X)):
            self.assertEqual(actualdLdX[iCh].dtype,datatype)
            self.assertTrue(torch.allclose(actualdLdX[iCh],expctddLdX[iCh],rtol=rtol,atol=atol))
        self.assertTrue(Z.requires_grad)

"""
        
        %Dec11Ch4Ord00Level2
        function testStepDec11Ch4Ord00Level2(testCase)
            
            dec = 1;
            ch = 4;
            ord = 0;
            height = 32;
            width = 32;
            nLevels = 2;
            subCoefs = cell(nLevels*(ch-1)+1,1);
            subCoefs{1} = rand(height/(dec^2),width/(dec^2));
            subCoefs{2} = rand(height/(dec^2),width/(dec^2));
            subCoefs{3} = rand(height/(dec^2),width/(dec^2));
            subCoefs{4} = rand(height/(dec^2),width/(dec^2));
            subCoefs{5} = rand(height/(dec),width/(dec));
            subCoefs{6} = rand(height/(dec),width/(dec));
            subCoefs{7} = rand(height/(dec),width/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');        
            phase = 0; % for phase adjustment required experimentaly
            subsubCoefs = cell(ch,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                imgExpctd = imfilter(...
                    upsample(...
                    upsample(subsubCoefs{1}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],1),'cir');
                for iSubSub = 2:ch
                    iSubband = (iLevel-1)*(ch-1)+iSubSub;
                    subbandImg = imfilter(...
                        upsample(...
                        upsample(subCoefs{iSubband}.',dec,phase).',...
                        dec,phase),step(lppufb,[],[],iSubSub),'cir');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');            
            testCase.synthesizer = NsoltSynthesis2dSystem(...
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch4Ord00Level2
        function testStepDec22Ch4Ord00Level2(testCase)
            
            dec = 2;
            ch = 4;
            ord = 0;
            height = 32;
            width = 32;
            nLevels = 2;
            subCoefs = cell(nLevels*(ch-1)+1,1);
            subCoefs{1} = rand(height/(dec^2),width/(dec^2));
            subCoefs{2} = rand(height/(dec^2),width/(dec^2));
            subCoefs{3} = rand(height/(dec^2),width/(dec^2));
            subCoefs{4} = rand(height/(dec^2),width/(dec^2));
            subCoefs{5} = rand(height/(dec),width/(dec));
            subCoefs{6} = rand(height/(dec),width/(dec));
            subCoefs{7} = rand(height/(dec),width/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');        
            phase = 1; % for phase adjustment required experimentaly
            subsubCoefs = cell(ch,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                imgExpctd = imfilter(...
                    upsample(...
                    upsample(subsubCoefs{1}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],1),'cir');
                for iSubSub = 2:ch
                    iSubband = (iLevel-1)*(ch-1)+iSubSub;
                    subbandImg = imfilter(...
                        upsample(...
                        upsample(subCoefs{iSubband}.',dec,phase).',...
                        dec,phase),step(lppufb,[],[],iSubSub),'cir');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');            
            testCase.synthesizer = NsoltSynthesis2dSystem(...
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch6Ord00Level2
        function testStepDec22Ch6Ord00Level2(testCase)
            
            dec = 2;
            ch = 6;
            ord = 0;
            height = 32;
            width = 32;
            nLevels = 2;
            subCoefs = cell(nLevels*(ch-1)+1,1);
            subCoefs{1} = rand(height/(dec^2),width/(dec^2));
            subCoefs{2} = rand(height/(dec^2),width/(dec^2));
            subCoefs{3} = rand(height/(dec^2),width/(dec^2));
            subCoefs{4} = rand(height/(dec^2),width/(dec^2));
            subCoefs{5} = rand(height/(dec^2),width/(dec^2));
            subCoefs{6} = rand(height/(dec^2),width/(dec^2));
            subCoefs{7} = rand(height/(dec),width/(dec));
            subCoefs{8} = rand(height/(dec),width/(dec));
            subCoefs{9} = rand(height/(dec),width/(dec));
            subCoefs{10} = rand(height/(dec),width/(dec));
            subCoefs{11} = rand(height/(dec),width/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');        
            phase = 1; % for phase adjustment required experimentaly
            subsubCoefs = cell(ch,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                imgExpctd = imfilter(...
                    upsample(...
                    upsample(subsubCoefs{1}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],1),'cir');
                for iSubSub = 2:ch
                    iSubband = (iLevel-1)*(ch-1)+iSubSub;
                    subbandImg = imfilter(...
                        upsample(...
                        upsample(subCoefs{iSubband}.',dec,phase).',...
                        dec,phase),step(lppufb,[],[],iSubSub),'cir');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');            
            testCase.synthesizer = NsoltSynthesis2dSystem(...
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch8Ord00Level2
        function testStepDec22Ch8Ord00Level2(testCase)
            
            dec = 2;
            ch = 8;
            ord = 0;
            height = 32;
            width = 32;
            nLevels = 2;
            subCoefs = cell(nLevels*(ch-1)+1,1);
            subCoefs{1} = rand(height/(dec^2),width/(dec^2));
            subCoefs{2} = rand(height/(dec^2),width/(dec^2));
            subCoefs{3} = rand(height/(dec^2),width/(dec^2));
            subCoefs{4} = rand(height/(dec^2),width/(dec^2));
            subCoefs{5} = rand(height/(dec^2),width/(dec^2));
            subCoefs{6} = rand(height/(dec^2),width/(dec^2));
            subCoefs{7} = rand(height/(dec^2),width/(dec^2));
            subCoefs{8} = rand(height/(dec^2),width/(dec^2));
            subCoefs{9} = rand(height/(dec),width/(dec));
            subCoefs{10} = rand(height/(dec),width/(dec));
            subCoefs{11} = rand(height/(dec),width/(dec));
            subCoefs{12} = rand(height/(dec),width/(dec));
            subCoefs{13} = rand(height/(dec),width/(dec));
            subCoefs{14} = rand(height/(dec),width/(dec));
            subCoefs{15} = rand(height/(dec),width/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');        
            phase = 1; % for phase adjustment required experimentaly
            subsubCoefs = cell(ch,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                imgExpctd = imfilter(...
                    upsample(...
                    upsample(subsubCoefs{1}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],1),'cir');
                for iSubSub = 2:ch
                    iSubband = (iLevel-1)*(ch-1)+iSubSub;
                    subbandImg = imfilter(...
                        upsample(...
                        upsample(subCoefs{iSubband}.',dec,phase).',...
                        dec,phase),step(lppufb,[],[],iSubSub),'cir');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');            
            testCase.synthesizer = NsoltSynthesis2dSystem(...
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec11Ch4Ord22Level1
        function testStepDec11Ch4Ord22Level1(testCase)
            
            dec = 1;
            ch = 4;
            ord = 2;
            height = 32;
            width = 32;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(ch,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end   
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');        
            imgExpctd = zeros(height,width);
            phase = 0; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');            
            testCase.synthesizer = NsoltSynthesis2dSystem(...
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = dec;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch4Ord22Level1
        function testStepDec22Ch4Ord22Level1(testCase)
            
            dec = 2;
            ch = 4;
            ord = 2;
            height = 32;
            width = 32;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(ch,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end   
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');        
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');            
            testCase.synthesizer = NsoltSynthesis2dSystem(...
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = dec;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch6Ord22Level1
        function testStepDec22Ch6Ord22Level1(testCase)
            
            dec = 2;
            ch = 6;
            ord = 2;
            height = 32;
            width = 32;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(ch,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end   
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');        
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');            
            testCase.synthesizer = NsoltSynthesis2dSystem(...
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = dec;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch8Ord22Level1
        function testStepDec22Ch8Ord22Level1(testCase)
            
            dec = 2;
            ch = 8;
            ord = 2;
            height = 32;
            width = 32;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(ch,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end   
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');        
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');            
            testCase.synthesizer = NsoltSynthesis2dSystem(...
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = dec;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec11Ch4Ord22Level2PeriodicExt
        function testStepDec11Ch4Ord22Level2PeriodicExt(testCase)
            
            dec = 1;
            ch = 4;
            ord = 2;
            height = 32;
            width = 32;
            nLevels = 2;
            subCoefs = cell(nLevels*(ch-1)+1,1);
            subCoefs{1} = rand(height/(dec^2),width/(dec^2));
            subCoefs{2} = rand(height/(dec^2),width/(dec^2));
            subCoefs{3} = rand(height/(dec^2),width/(dec^2));
            subCoefs{4} = rand(height/(dec^2),width/(dec^2));
            subCoefs{5} = rand(height/(dec),width/(dec));
            subCoefs{6} = rand(height/(dec),width/(dec));
            subCoefs{7} = rand(height/(dec),width/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');        
            phase = 0; % for phase adjustment required experimentaly
            subsubCoefs = cell(ch,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                imgExpctd = imfilter(...
                    upsample(...
                    upsample(subsubCoefs{1}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],1),'cir');
                for iSubSub = 2:ch
                    iSubband = (iLevel-1)*(ch-1)+iSubSub;
                    subbandImg = imfilter(...
                        upsample(...
                        upsample(subCoefs{iSubband}.',dec,phase).',...
                        dec,phase),step(lppufb,[],[],iSubSub),'cir');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');            
            testCase.synthesizer = NsoltSynthesis2dSystem(...
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch4Ord22Level2PeriodicExt
        function testStepDec22Ch4Ord22Level2PeriodicExt(testCase)
            
            dec = 2;
            ch = 4;
            ord = 2;
            height = 32;
            width = 32;
            nLevels = 2;
            subCoefs = cell(nLevels*(ch-1)+1,1);
            subCoefs{1} = rand(height/(dec^2),width/(dec^2));
            subCoefs{2} = rand(height/(dec^2),width/(dec^2));
            subCoefs{3} = rand(height/(dec^2),width/(dec^2));
            subCoefs{4} = rand(height/(dec^2),width/(dec^2));
            subCoefs{5} = rand(height/(dec),width/(dec));
            subCoefs{6} = rand(height/(dec),width/(dec));
            subCoefs{7} = rand(height/(dec),width/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');        
            phase = 1; % for phase adjustment required experimentaly
            subsubCoefs = cell(ch,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                imgExpctd = imfilter(...
                    upsample(...
                    upsample(subsubCoefs{1}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],1),'cir');
                for iSubSub = 2:ch
                    iSubband = (iLevel-1)*(ch-1)+iSubSub;
                    subbandImg = imfilter(...
                        upsample(...
                        upsample(subCoefs{iSubband}.',dec,phase).',...
                        dec,phase),step(lppufb,[],[],iSubSub),'cir');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');            
            testCase.synthesizer = NsoltSynthesis2dSystem(...
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch6Ord22Level2PeriodicExt
        function testStepDec22Ch6Ord22Level2PeriodicExt(testCase)
            
            dec = 2;
            ch = 6;
            ord = 2;
            height = 32;
            width = 32;
            nLevels = 2;
            subCoefs = cell(nLevels*(ch-1)+1,1);
            subCoefs{1} = rand(height/(dec^2),width/(dec^2));
            subCoefs{2} = rand(height/(dec^2),width/(dec^2));
            subCoefs{3} = rand(height/(dec^2),width/(dec^2));
            subCoefs{4} = rand(height/(dec^2),width/(dec^2));
            subCoefs{5} = rand(height/(dec^2),width/(dec^2));
            subCoefs{6} = rand(height/(dec^2),width/(dec^2));
            subCoefs{7} = rand(height/(dec),width/(dec));
            subCoefs{8} = rand(height/(dec),width/(dec));
            subCoefs{9} = rand(height/(dec),width/(dec));
            subCoefs{10} = rand(height/(dec),width/(dec));
            subCoefs{11} = rand(height/(dec),width/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');        
            phase = 1; % for phase adjustment required experimentaly
            subsubCoefs = cell(ch,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                imgExpctd = imfilter(...
                    upsample(...
                    upsample(subsubCoefs{1}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],1),'cir');
                for iSubSub = 2:ch
                    iSubband = (iLevel-1)*(ch-1)+iSubSub;
                    subbandImg = imfilter(...
                        upsample(...
                        upsample(subCoefs{iSubband}.',dec,phase).',...
                        dec,phase),step(lppufb,[],[],iSubSub),'cir');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');            
            testCase.synthesizer = NsoltSynthesis2dSystem(...
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch44Ord22Level2PeriodicExt
        function testStepDec22Ch44Ord22Level2PeriodicExt(testCase)
            
            dec = 2;
            decch = [ dec dec 4 4 ];
            ch = sum(decch(3:4));
            ord = 2;
            height = 32;
            width = 32;
            nLevels = 2;
            subCoefs = cell(nLevels*(ch-1)+1,1);
            subCoefs{1} = rand(height/(dec^2),width/(dec^2));
            subCoefs{2} = rand(height/(dec^2),width/(dec^2));
            subCoefs{3} = rand(height/(dec^2),width/(dec^2));
            subCoefs{4} = rand(height/(dec^2),width/(dec^2));
            subCoefs{5} = rand(height/(dec^2),width/(dec^2));
            subCoefs{6} = rand(height/(dec^2),width/(dec^2));
            subCoefs{7} = rand(height/(dec^2),width/(dec^2));
            subCoefs{8} = rand(height/(dec^2),width/(dec^2));
            subCoefs{9} = rand(height/(dec),width/(dec));
            subCoefs{10} = rand(height/(dec),width/(dec));
            subCoefs{11} = rand(height/(dec),width/(dec));
            subCoefs{12} = rand(height/(dec),width/(dec));
            subCoefs{13} = rand(height/(dec),width/(dec));
            subCoefs{14} = rand(height/(dec),width/(dec));
            subCoefs{15} = rand(height/(dec),width/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');        
            phase = 1; % for phase adjustment required experimentaly
            subsubCoefs = cell(ch,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                imgExpctd = imfilter(...
                    upsample(...
                    upsample(subsubCoefs{1}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],1),'cir');
                for iSubSub = 2:ch
                    iSubband = (iLevel-1)*(ch-1)+iSubSub;
                    subbandImg = imfilter(...
                        upsample(...
                        upsample(subCoefs{iSubband}.',dec,phase).',...
                        dec,phase),step(lppufb,[],[],iSubSub),'cir');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');            
            testCase.synthesizer = NsoltSynthesis2dSystem(...
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch44Ord44Level3PeriodicExt
        function testStepDec22Ch44Ord44Level3PeriodicExt(testCase)
            
            dec = 2;
            decch = [ dec dec 4 4 ];
            ch = sum(decch(3:4));
            ord = 4;
            height = 64;
            width = 64;
            nLevels = 3;
            subCoefs = cell(nLevels*(ch-1)+1,1);
            subCoefs{1} = rand(height/(dec^3),width/(dec^3));
            subCoefs{2} = rand(height/(dec^3),width/(dec^3));
            subCoefs{3} = rand(height/(dec^3),width/(dec^3));
            subCoefs{4} = rand(height/(dec^3),width/(dec^3));
            subCoefs{5} = rand(height/(dec^3),width/(dec^3));
            subCoefs{6} = rand(height/(dec^3),width/(dec^3));
            subCoefs{7} = rand(height/(dec^3),width/(dec^3));
            subCoefs{8} = rand(height/(dec^3),width/(dec^3));
            subCoefs{9} = rand(height/(dec^2),width/(dec^2));
            subCoefs{10} = rand(height/(dec^2),width/(dec^2));
            subCoefs{11} = rand(height/(dec^2),width/(dec^2));
            subCoefs{12} = rand(height/(dec^2),width/(dec^2));
            subCoefs{13} = rand(height/(dec^2),width/(dec^2));
            subCoefs{14} = rand(height/(dec^2),width/(dec^2));
            subCoefs{15} = rand(height/(dec^2),width/(dec^2));
            subCoefs{16} = rand(height/(dec),width/(dec));
            subCoefs{17} = rand(height/(dec),width/(dec));
            subCoefs{18} = rand(height/(dec),width/(dec));
            subCoefs{19} = rand(height/(dec),width/(dec));
            subCoefs{20} = rand(height/(dec),width/(dec));
            subCoefs{21} = rand(height/(dec),width/(dec));
            subCoefs{22} = rand(height/(dec),width/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');        
            phase = 1; % for phase adjustment required experimentaly
            subsubCoefs = cell(ch,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                imgExpctd = imfilter(...
                    upsample(...
                    upsample(subsubCoefs{1}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],1),'cir');
                for iSubSub = 2:ch
                    iSubband = (iLevel-1)*(ch-1)+iSubSub;
                    subbandImg = imfilter(...
                        upsample(...
                        upsample(subCoefs{iSubband}.',dec,phase).',...
                        dec,phase),step(lppufb,[],[],iSubSub),'cir');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');            
            testCase.synthesizer = NsoltSynthesis2dSystem(...
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        function testSetLpPuFb2dDec22Ch44Ord44(testCase)
            
            dec = 2;
            decch = [ dec dec 4 4 ];
            ord = 4;
            height = 32;
            width = 32;
            subCoefs{1} = rand(height/(dec),width/(dec));
            subCoefs{2} = rand(height/(dec),width/(dec));
            subCoefs{3} = rand(height/(dec),width/(dec));
            subCoefs{4} = rand(height/(dec),width/(dec));
            subCoefs{5} = rand(height/(dec),width/(dec));
            subCoefs{6} = rand(height/(dec),width/(dec));
            subCoefs{7} = rand(height/(dec),width/(dec));
            subCoefs{8} = rand(height/(dec),width/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord]);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');            
            testCase.synthesizer = NsoltSynthesis2dSystem(...
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Termination');
            imgPre = step(testCase.synthesizer,coefs,scales);
            
            % Update lppufb
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            imgPst = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan;
            diff = norm(imgPst(:)-imgPre(:));
            testCase.verifyTrue(diff<1e-15);
                        
            % ReInstantiation of target class
            testCase.synthesizer = NsoltSynthesis2dSystem(...
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Termination');
            imgPst = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan;
            diff = norm(imgPst(:)-imgPre(:));
            testCase.verifyThat(diff,IsGreaterThan(0));
        end
        
        function testIsCloneLpPuFbFalse(testCase)
            
            dec = 2;
            ch = [ 4 4 ];
            ord = 4;
            height = 32;
            width = 32;
            subCoefs{1} = rand(height/(dec),width/(dec));
            subCoefs{2} = rand(height/(dec),width/(dec));
            subCoefs{3} = rand(height/(dec),width/(dec));
            subCoefs{4} = rand(height/(dec),width/(dec));
            subCoefs{5} = rand(height/(dec),width/(dec));
            subCoefs{6} = rand(height/(dec),width/(dec));
            subCoefs{7} = rand(height/(dec),width/(dec));
            subCoefs{8} = rand(height/(dec),width/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord],...
                'OutputMode','ParameterMatrixSet');
            
            % Instantiation of target class
            testCase.synthesizer = NsoltSynthesis2dSystem(...
                'LpPuFb2d',lppufb,...
                'IsCloneLpPuFb',true);
            
            % Pre
            imgPre = step(testCase.synthesizer,coefs,scales);
            
            % Update lppufb
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Pst
            imgPst = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan;
            diff = norm(imgPst(:)-imgPre(:));
            testCase.verifyEqual(diff,0,'AbsTol',1e-13,sprintf('%g',diff));
            
            % ReInstantiation of target class
            testCase.synthesizer = NsoltSynthesis2dSystem(...
                'LpPuFb2d',lppufb,...
                'IsCloneLpPuFb',false);
            
            % Pre
            imgPre = step(testCase.synthesizer,coefs,scales);
            
            % Update lppufb
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Pst
            imgPst = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan;
            diff = norm(imgPst(:)-imgPre(:));
            testCase.verifyThat(diff,IsGreaterThan(0),sprintf('%g',diff));
        end
        
        % Test
        function testClone(testCase)
            
            dec = [ 2 2 ];
            ch =  [ 4 4 ];
            ord = [ 4 4 ];
            height = 64;
            width  = 64;
            coefs = rand(sum(ch)/prod(dec)*height*width,1);
            scales = repmat([height/dec(1) width/dec(2)],[sum(ch) 1]);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord,...
                'OutputMode','ParameterMatrixSet');
            
            % Instantiation of target class
            testCase.synthesizer = NsoltSynthesis2dSystem(...
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Termination');
            
            % Clone
            cloneSynthesizer = clone(testCase.synthesizer);
            
            % Evaluation
            testCase.verifyEqual(cloneSynthesizer,testCase.synthesizer);
            testCase.verifyFalse(cloneSynthesizer == testCase.synthesizer);
            prpOrg = get(testCase.synthesizer,'LpPuFb2d');
            prpCln = get(cloneSynthesizer,'LpPuFb2d');
            testCase.verifyEqual(prpCln,prpOrg);
            testCase.verifyFalse(prpCln == prpOrg);
            %
            recImgExpctd = step(testCase.synthesizer,coefs,scales);
            recImgActual = step(cloneSynthesizer,coefs,scales);
            testCase.assertEqual(recImgActual,recImgExpctd);
            
        end
        
        % Test
        function testConstructionTypeII(testCase)
            
            % Expected values
            import saivdr.dictionary.nsoltx.*
            lppufbExpctd = OvsdLpPuFb2dTypeIIVm1System(...
                'OutputMode','ParameterMatrixSet');
            
            % Instantiation
            testCase.synthesizer = NsoltSynthesis2dSystem(...
                'LpPuFb2d',lppufbExpctd);
            
            % Actual value
            lppufbActual = get(testCase.synthesizer,'LpPuFb2d');
            
            % Evaluation
            testCase.assertEqual(lppufbActual,lppufbExpctd);
        end
        
        % Test
        function testDefaultConstruction6plus2(testCase)
            
            % Preperation
            nChs = [6 2];
           
            % Expected values
            import saivdr.dictionary.nsoltx.*
            lppufbExpctd = OvsdLpPuFb2dTypeIIVm1System(...
                'NumberOfChannels',nChs,...
                'OutputMode','ParameterMatrixSet');
            
            % Instantiation
            import saivdr.dictionary.nsoltx.ChannelGroup
            testCase.synthesizer = NsoltAnalysis2dSystem(...
                'NumberOfSymmetricChannels',nChs(ChannelGroup.UPPER),...
                'NumberOfAntisymmetricChannels',nChs(ChannelGroup.LOWER));
            
            % Actual value
            lppufbActual = get(testCase.synthesizer,'LpPuFb2d');
            
            % Evaluation
            testCase.assertEqual(lppufbActual,lppufbExpctd);
        end
        % Test for default construction
        function testInverseBlockDctDec33(testCase)
            
            dec = 3;
            height = 24;
            width = 24;
            subCoefs  = rand(height*dec,width/dec);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iCh = 1:dec*dec
                subImg = subCoefs(iCh:dec*dec:end,:);
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iCh,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Expected values
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec]);
            E0 = step(lppufb,[],[]);
            %fun = @(x) reshape(flipud(E0.'*x.data(:)),dec,dec);
            %imgExpctd = blockproc(subCoefs,[dec*dec 1],fun);
            fun = @(x) reshape(flipud(E0.'*x.data(:)),dec,dec);
            imgExpctd = blockproc(subCoefs,[dec*dec 1],fun);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
            
        end
        
        % Test
        function testInverseBlockDctDec55(testCase)
            
            dec = 5;
            height = 40;
            width = 40;
            subCoefs = rand(height*dec,width/dec);
            coefs = zeros(1,height*width);
            scales = zeros(dec*dec,2);
            sIdx = 1;
            for iCh = 1:dec*dec
                subImg = subCoefs(iCh:dec*dec:end,:);
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iCh,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Expected values
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec]);
            E0 = step(lppufb,[],[]);
            fun = @(x) reshape(flipud(E0.'*x.data(:)),dec,dec);
            imgExpctd = blockproc(subCoefs,[dec*dec 1],fun);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec22Ch5Ord00(testCase)
            
            dec = 2;
            nChs = 5;
            height = 16;
            width = 16;
            subCoefs = rand(height*nChs/dec,width/dec);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iCh = 1:nChs
                subImg = subCoefs(iCh:nChs:end,:);
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iCh,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',nChs,...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            E = step(lppufb,[],[]);
            
            % Expected values
            fun = @(x) reshape(flipud(E.'*x.data(:)),dec,dec);
            imgExpctd = blockproc(subCoefs,[nChs 1],fun);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec44Cg16Ord00(testCase)
            
            dec = 4;
            nChs = 17;
            height = 32;
            width = 32;
            subCoefs = rand(height*nChs,width/dec);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iCh = 1:nChs
                subImg = subCoefs(iCh:nChs:end,:);
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iCh,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', nChs,...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            E = step(lppufb,[],[]);
            
            % Expected values
            fun = @(x) reshape(flipud(E.'*x.data(:)),dec,dec);
            imgExpctd = blockproc(subCoefs,[nChs 1],fun);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec22Ch5Ord22Vm0(testCase)
            
            dec = 2;
            nChs = 5;
            ord = 2;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            import saivdr.dictionary.nsoltx.*
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(dec+1:end-dec,dec+1:end-dec); % ignore border
            imgActual = imgActual(dec+1:end-dec,dec+1:end-dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec22Ch5Ord22Vm1(testCase)
            
            dec = 2;
            nChs = 5;
            ord = 2;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            import saivdr.dictionary.nsoltx.*
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(dec+1:end-dec,dec+1:end-dec); % ignore border
            imgActual = imgActual(dec+1:end-dec,dec+1:end-dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec22Ch5Ord22PeriodicExt(testCase)
            
            dec = 2;
            nChs = 5;
            ord = 2;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec44Ch17Ord22(testCase)
            
            dec = 4;
            nChs = 17;
            ord = 2;
            height = 32;
            width = 32;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 2; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(dec+1:end-dec,dec+1:end-dec); % ignore border
            imgActual = imgActual(dec+1:end-dec,dec+1:end-dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec44Ch17Ord22PeriodicExt(testCase)
            
            dec = 4;
            nChs = 17;
            ord = 2;
            height = 32;
            width = 32;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 2; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test for boundary operation
        function testInverseBlockDctDec33Ord22(testCase)
            
            dec = 3;
            ord = 2;
            height = 24;
            width = 24;
            nChs = dec*dec;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'PolyPhaseOrder',[ord ord]);
            
            
            % Expected values
            coefsExpctd = zeros(height,width);
            for iCol = 1:dec
                for iRow = 1:dec
                    iSubband = (iCol-1)*dec + iRow;
                    coefsExpctd = coefsExpctd + upsample(...
                        upsample(...
                        subCoefs{iSubband}.',dec,iCol-1).',dec,iRow-1);
                end
            end
            E0 = step(NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec]),[],[]);
            fun = @(x) reshape(flipud(E0.'*x.data(:)),dec,dec);
            imgExpctd = blockproc(coefsExpctd,...
                [dec dec],fun);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test for boundary operation
        function testInverseBlockDctDec55Ord22(testCase)
            
            dec = 5;
            ord = 2;
            height = 40;
            width = 40;
            nChs = dec*dec;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'PolyPhaseOrder',[ord ord]);
            
            
            % Expected values
            coefsExpctd = zeros(height,width);
            for iCol = 1:dec
                for iRow = 1:dec
                    iSubband = (iCol-1)*dec + iRow;
                    coefsExpctd = coefsExpctd + upsample(...
                        upsample(...
                        subCoefs{iSubband}.',dec,iCol-1).',dec,iRow-1);
                end
            end
            E0 = step(NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec]),[],[]);
            fun = @(x) reshape(flipud(E0.'*x.data(:)),dec,dec);
            imgExpctd = blockproc(coefsExpctd,...
                [dec dec],fun);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec22Ch5Ord44(testCase)
            
            dec = 2;
            nChs = 5;
            ord = 4;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(2*dec+1:end-2*dec,2*dec+1:end-2*dec); % ignore border
            imgActual = imgActual(2*dec+1:end-2*dec,2*dec+1:end-2*dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec22Ch5Ord44PeriodicExt(testCase)
            
            dec = 2;
            nChs = 5;
            ord = 4;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec4Ch17Ord44(testCase)
            
            dec = 4;
            nChs = 17;
            ord = 4;
            height = 32;
            width = 32;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 2; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(2*dec+1:end-2*dec,2*dec+1:end-2*dec); % ignore border
            imgActual = imgActual(2*dec+1:end-2*dec,2*dec+1:end-2*dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec44Ch17Ord44PeriodicExt(testCase)
            
            dec = 4;
            nChs = 17;
            ord = 4;
            height = 32;
            width = 32;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 2; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testInverseBlockDctDec33Ord44(testCase)
            
            dec = 3;
            ord = 4;
            height = 24;
            width = 24;
            nChs = dec*dec;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'PolyPhaseOrder',[ord ord]);
            
            
            % Expected values
            coefsExpctd = zeros(height,width);
            for iCol = 1:dec
                for iRow = 1:dec
                    iSubband = (iCol-1)*dec + iRow;
                    coefsExpctd = coefsExpctd + upsample(...
                        upsample(...
                        subCoefs{iSubband}.',dec,iCol-1).',dec,iRow-1);
                end
            end
            E0 = step(NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec]),[],[]);
            fun = @(x) reshape(flipud(E0.'*x.data(:)),dec,dec);
            imgExpctd = blockproc(coefsExpctd,...
                [dec dec],fun);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test for boundary operation
        function testInverseBlockDctDec55Ord44(testCase)
            
            dec = 5;
            ord = 4;
            height = 40;
            width = 40;
            nChs = dec*dec;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'PolyPhaseOrder',[ord ord]);
            
            
            % Expected values
            coefsExpctd = zeros(height,width);
            for iCol = 1:dec
                for iRow = 1:dec
                    iSubband = (iCol-1)*dec + iRow;
                    coefsExpctd = coefsExpctd + upsample(...
                        upsample(...
                        subCoefs{iSubband}.',dec,iCol-1).',dec,iRow-1);
                end
            end
            E0 = step(NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec]),[],[]);
            fun = @(x) reshape(flipud(E0.'*x.data(:)),dec,dec);
            imgExpctd = blockproc(coefsExpctd,...
                [dec dec],fun);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec22Ch5Ord66(testCase)
            
            dec = 2;
            nChs = 5;
            ord = 6;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(3*dec+1:end-3*dec,3*dec+1:end-3*dec); % ignore border
            imgActual = imgActual(3*dec+1:end-3*dec,3*dec+1:end-3*dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec22Ch5Ord66PeriodicExt(testCase)
            
            dec = 2;
            nChs = 5;
            ord = 6;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb,...'
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec44Ch17Ord66(testCase)
            
            dec = 4;
            nChs = 17;
            ord = 6;
            height = 32;
            width = 32;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 2; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(3*dec+1:end-3*dec,3*dec+1:end-3*dec); % ignore border
            imgActual = imgActual(3*dec+1:end-3*dec,3*dec+1:end-3*dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec44Ch17Ord66PeriodicExt(testCase)
            
            dec = 4;
            nChs = 17;
            ord = 6;
            height = 32;
            width = 32;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 2; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testInverseBlockDctDec33Ord66(testCase)
            
            dec = 3;
            ord = 6;
            height = 24;
            width = 24;
            nChs = dec*dec;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',[ord ord]);
            
            
            % Expected values
            coefsExpctd = zeros(height,width);
            for iCol = 1:dec
                for iRow = 1:dec
                    iSubband = (iCol-1)*dec + iRow;
                    coefsExpctd = coefsExpctd + upsample(...
                        upsample(...
                        subCoefs{iSubband}.',dec,iCol-1).',dec,iRow-1);
                end
            end
            E0 = step(NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec]),[],[]);
            fun = @(x) reshape(flipud(E0.'*x.data(:)),dec,dec);
            imgExpctd = blockproc(coefsExpctd,...
                [dec dec],fun);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test for boundary operation
        function testInverseBlockDctDec55Ord66(testCase)
            
            dec = 5;
            ord = 6;
            height = 40;
            width = 40;
            nChs = dec*dec;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'PolyPhaseOrder',[ord ord]);
            
            
            % Expected values
            coefsExpctd = zeros(height,width);
            for iCol = 1:dec
                for iRow = 1:dec
                    iSubband = (iCol-1)*dec + iRow;
                    coefsExpctd = coefsExpctd + upsample(...
                        upsample(...
                        subCoefs{iSubband}.',dec,iCol-1).',dec,iRow-1);
                end
            end
            E0 = step(NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec]),[],[]);
            fun = @(x) reshape(flipud(E0.'*x.data(:)),dec,dec);
            imgExpctd = blockproc(coefsExpctd,...
                [dec dec],fun);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec22Ch5Ord02(testCase)
            
            dec = 2;
            nChs = 5;
            ord = 2;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder',[0 ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(:,dec+1:end-dec); % ignore border
            imgActual = imgActual(:,dec+1:end-dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec44Ch17Ord02(testCase)
            
            dec = 4;
            nChs = 17;
            ord = 2;
            height = 32;
            width = 32;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder',[0 ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 2; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(:,dec+1:end-dec); % ignore border
            imgActual = imgActual(:,dec+1:end-dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec22Ch5Ord04(testCase)
            
            dec = 2;
            nChs = 5;
            ord = 4;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder',[0 ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(:,2*dec+1:end-2*dec); % ignore border
            imgActual = imgActual(:,2*dec+1:end-2*dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec44Ch17Ord04(testCase)
            
            dec = 4;
            nChs = 17;
            ord = 4;
            height = 32;
            width = 32;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder',[0 ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 2; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(:,2*dec+1:end-2*dec); % ignore border
            imgActual = imgActual(:,2*dec+1:end-2*dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec22Ch5Ord20(testCase)
            
            dec = 2;
            nChs = 5;
            ord = 2;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder',[ord 0]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(dec+1:end-dec,:); % ignore border
            imgActual = imgActual(dec+1:end-dec,:); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec44Ch17Ord20(testCase)
            
            dec = 4;
            nChs = 17;
            ord = 2;
            height = 32;
            width = 32;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder',[ord 0]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 2; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(dec+1:end-dec,:); % ignore border
            imgActual = imgActual(dec+1:end-dec,:); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec22Ch5Ord40(testCase)
            
            dec = 2;
            nChs = 5;
            ord = 4;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder',[ord 0]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(2*dec+1:end-2*dec,:); % ignore border
            imgActual = imgActual(2*dec+1:end-2*dec,:); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec44Ch17Ord40(testCase)
            
            dec = 4;
            nChs = 17;
            ord = 4;
            height = 32;
            width = 32;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder',[ord 0]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 2; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(2*dec+1:end-2*dec,:); % ignore border
            imgActual = imgActual(2*dec+1:end-2*dec,:); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec22Ch5Ord02PeriodicExt(testCase)
            
            dec = 2;
            nChs = 5;
            ord = 2;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder',[0 ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec44Ch17Ord02PeriodicExt(testCase)
            
            dec = 4;
            nChs = 17;
            ord = 2;
            height = 32;
            width = 32;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder',[0 ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 2; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec22Ch5Ord04PeriodicExt(testCase)
            
            dec = 2;
            nChs = 5;
            ord = 4;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder',[0 ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec44Ch17Ord04PeriodicExt(testCase)
            
            dec = 4;
            nChs = 17;
            ord = 4;
            height = 32;
            width = 32;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder',[0 ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 2; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec22Ch5Ord20PeriodicExt(testCase)
            
            dec = 2;
            nChs = 5;
            ord = 2;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder',[ord 0]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec44Ch17Ord20PeriodicExt(testCase)
            
            dec = 4;
            nChs = 17;
            ord = 2;
            height = 32;
            width = 32;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder',[ord 0]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 2; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        
        %Test
        function testStepDec22Ch5Ord40PeriodicExt(testCase)
            
            dec = 2;
            nChs = 5;
            ord = 4;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder',[ord 0]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec44Ch17Ord40PeriodicExt(testCase)
            
            dec = 4;
            nChs = 17;
            ord = 4;
            height = 32;
            width = 32;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder',[ord 0]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 2; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec22Ch5Ord24(testCase)
            
            dec = 2;
            nChs = 5;
            ord = [2 4];
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(dec+1:end-dec,2*dec+1:end-2*dec); % ignore border
            imgActual = imgActual(dec+1:end-dec,2*dec+1:end-2*dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec44Ch17Ord24(testCase)
            
            dec = 4;
            nChs = 17;
            ord = [2 4];
            height = 32;
            width = 32;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 2; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(dec+1:end-dec,2*dec+1:end-2*dec); % ignore border
            imgActual = imgActual(dec+1:end-dec,2*dec+1:end-2*dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec22Ch5Ord24PeriodicExt(testCase)
            
            dec = 2;
            nChs = 5;
            ord = [2 4];
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec44Ch17Ord24PeriodicExt(testCase)
            
            dec = 4;
            nChs = 17;
            ord = [2 4];
            height = 32;
            width = 32;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 2; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec22Ch5Ord42(testCase)
            
            dec = 2;
            nChs = 5;
            ord = [4 2];
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(2*dec+1:end-2*dec,dec+1:end-dec); % ignore border
            imgActual = imgActual(2*dec+1:end-2*dec,dec+1:end-dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec44Ch17Ord42(testCase)
            
            dec = 4;
            nChs = 17;
            ord = [4 2];
            height = 32;
            width = 32;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 2; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(2*dec+1:end-2*dec,dec+1:end-dec); % ignore border
            imgActual = imgActual(2*dec+1:end-2*dec,dec+1:end-dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec22Ch5Ord42PeriodicExt(testCase)
            
            dec = 2;
            nChs = 5;
            ord = [4 2];
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec44Ch17Ord42PeriodicExt(testCase)
            
            dec = 4;
            nChs = 17;
            ord = [4 2];
            height = 32;
            width = 32;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 2; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec11Ch5Ord00
        function testStepDec11Ch5Ord00(testCase)
            
            dec = 1;
            ch = 5;
            ord = 0;
            height = 16;
            width = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(ch,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 0; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch7Ord00
        function testStepDec22Ch7Ord00(testCase)
            
            dec = 2;
            ch = 7;
            ord = 0;
            height = 16;
            width = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(ch,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
            
        end
        
        %Dec22Ch9Ord00
        function testStepDec22Ch9Ord00(testCase)
            
            dec = 2;
            ch = 9;
            ord = 0;
            height = 16;
            width = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(ch,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
            
        end
        
        %Dec22Ch11Ord00
        function testStepDec22Ch11Ord00(testCase)
            
            dec = 2;
            ch = 11;
            ord = 0;
            height = 16;
            width = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(ch,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
            
        end
        
        %Dec11Ch4Ord22
        function testStepDec11Ch5Ord22(testCase)
            
            dec = 1;
            ch = 5;
            ord = 2;
            height = 16;
            width = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(ch,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 0; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(dec+1:end-dec,dec+1:end-dec); % ignore border
            imgActual = imgActual(dec+1:end-dec,dec+1:end-dec); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch4Ord22
        function testStepDec22Ch7Ord22(testCase)
            
            dec = 2;
            ch = 7;
            ord = 2;
            height = 16;
            width = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(ch,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = dec;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch6Ord22
        function testStepDec22Ch9Ord22(testCase)
            
            dec = 2;
            ch  = 9;
            ord = 2;
            height = 16;
            width = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(ch,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = dec;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch8Ord22
        function testStepDec22Ch11Ord22(testCase)
            
            dec = 2;
            ch  = 11;
            ord = 2;
            height = 16;
            width = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(ch,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = dec;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec11Ch4Ord44
        function testStepDec11Ch5Ord44(testCase)
            
            dec = 1;
            ch  = 5;
            ord = 4;
            height = 16;
            width = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(ch,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 0; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = ord*dec/2;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch4Ord44
        function testStepDec22Ch7Ord44(testCase)
            
            dec = 2;
            ch  = 7;
            ord = 4;
            height = 16;
            width = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(ch,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = ord*dec/2;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch6Ord44
        function testStepDec22Ch9Ord44(testCase)
            
            dec = 2;
            ch  = 9;
            ord = 4;
            height = 16;
            width = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(ch,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = ord*dec/2;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch8Ord44
        function testStepDec22Ch11Ord44(testCase)
            
            dec = 2;
            ch  = 11;
            ord = 4;
            height = 16;
            width = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(ch,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = ord*dec/2;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec11Ch4Ord66
        function testStepDec11Ch5Ord66(testCase)
            
            dec = 1;
            ch  = 5;
            ord = 6;
            height = 16;
            width = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(ch,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 0; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = ord*dec/2;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch4Ord66
        function testStepDec22Ch7Ord66(testCase)
            
            dec = 2;
            ch  = 7;
            ord = 6;
            height = 16;
            width = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(ch,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = ord*dec/2;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch6Ord66
        function testStepDec22Ch9Ord66(testCase)
            
            dec = 2;
            ch  = 9;
            ord = 6;
            height = 16;
            width = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(ch,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = ord*dec/2;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch8Ord66
        function testStepDec22Ch11Ord66(testCase)
            
            dec = 2;
            ch  = 11;
            ord = 6;
            height = 16;
            width = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(ch,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = ord*dec/2;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec11Ch4Ord66PeriodicExt
        function testStepDec11Ch5Ord66PeriodicExt(testCase)
            
            dec = 1;
            ch  = 5;
            ord = 6;
            height = 16;
            width = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(ch,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 0; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = 0;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border);
            imgActual = imgActual(border+1:end-border,border+1:end-border);
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
              
        %Dec22Ch6Ord66PeriodicExt
        function testStepDec22Ch7Ord66PeriodicExt(testCase)
            
            dec = 2;
            ch  = 7;
            ord = 6;
            height = 16;
            width = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(ch,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = 0;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border);
            imgActual = imgActual(border+1:end-border,border+1:end-border);
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch8Ord66PeriodicExt
        function testStepDec22Ch9Ord66PeriodicExt(testCase)
            
            dec = 2;
            ch  = 9;
            ord = 6;
            height = 16;
            width = 16;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(ch,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = 0;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border);
            imgActual = imgActual(border+1:end-border,border+1:end-border);
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec11Ch4Ord00Level1
        function testStepDec11Ch5Ord00Level1(testCase)
            
            dec = 1;
            ch = 5;
            ord = 0;
            height = 32;
            width = 32;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(ch,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end  ;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 0; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(...
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch5Ord00Level1
        function testStepDec22Ch5Ord00Level1(testCase)
            
            dec = 2;
            ch = 5;
            ord = 0;
            height = 32;
            width = 32;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(ch,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(...
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch6Ord00Level1
        function testStepDec22Ch7Ord00Level1(testCase)
            
            dec = 2;
            nChs= 7;
            ord = 0;
            height = 32;
            width = 32;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(...
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
            
        end
        
        %Dec22Ch8Ord00Level1
        function testStepDec22Ch9Ord00Level1(testCase)
            
            dec = 2;
            ch = 9;
            ord = 0;
            height = 32;
            width = 32;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(ch,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(...
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec11Ch4Ord00Level2
        function testStepDec11Ch5Ord00Level2(testCase)
            
            dec = 1;
            ch = 5;
            ord = 0;
            height = 32;
            width = 32;
            nLevels = 2;
            subCoefs = cell(nLevels*(ch-1)+1,1);
            subCoefs{1} = rand(height/(dec^2),width/(dec^2));
            subCoefs{2} = rand(height/(dec^2),width/(dec^2));
            subCoefs{3} = rand(height/(dec^2),width/(dec^2));
            subCoefs{4} = rand(height/(dec^2),width/(dec^2));
            subCoefs{5} = rand(height/(dec^2),width/(dec^2));
            subCoefs{6} = rand(height/(dec),width/(dec));
            subCoefs{7} = rand(height/(dec),width/(dec));
            subCoefs{8} = rand(height/(dec),width/(dec));
            subCoefs{9} = rand(height/(dec),width/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            phase = 0; % for phase adjustment required experimentaly
            subsubCoefs = cell(ch,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                imgExpctd = imfilter(...
                    upsample(...
                    upsample(subsubCoefs{1}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],1),'cir');
                for iSubSub = 2:ch
                    iSubband = (iLevel-1)*(ch-1)+iSubSub;
                    subbandImg = imfilter(...
                        upsample(...
                        upsample(subCoefs{iSubband}.',dec,phase).',...
                        dec,phase),step(lppufb,[],[],iSubSub),'cir');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(...
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch4Ord00Level2
        function testStepDec22Ch5Ord00Level2(testCase)
            
            dec = 2;
            ch = 5;
            ord = 0;
            height = 32;
            width = 32;
            nLevels = 2;
            subCoefs = cell(nLevels*(ch-1)+1,1);
            subCoefs{1} = rand(height/(dec^2),width/(dec^2));
            subCoefs{2} = rand(height/(dec^2),width/(dec^2));
            subCoefs{3} = rand(height/(dec^2),width/(dec^2));
            subCoefs{4} = rand(height/(dec^2),width/(dec^2));
            subCoefs{5} = rand(height/(dec^2),width/(dec^2));
            subCoefs{6} = rand(height/(dec),width/(dec));
            subCoefs{7} = rand(height/(dec),width/(dec));
            subCoefs{8} = rand(height/(dec),width/(dec));
            subCoefs{9} = rand(height/(dec),width/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            phase = 1; % for phase adjustment required experimentaly
            subsubCoefs = cell(ch,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                imgExpctd = imfilter(...
                    upsample(...
                    upsample(subsubCoefs{1}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],1),'cir');
                for iSubSub = 2:ch
                    iSubband = (iLevel-1)*(ch-1)+iSubSub;
                    subbandImg = imfilter(...
                        upsample(...
                        upsample(subCoefs{iSubband}.',dec,phase).',...
                        dec,phase),step(lppufb,[],[],iSubSub),'cir');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(...
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch6Ord00Level2
        function testStepDec22Ch7Ord00Level2(testCase)
            
            dec = 2;
            ch = 7;
            ord = 0;
            height = 32;
            width = 32;
            nLevels = 2;
            subCoefs = cell(nLevels*(ch-1)+1,1);
            subCoefs{1} = rand(height/(dec^2),width/(dec^2));
            subCoefs{2} = rand(height/(dec^2),width/(dec^2));
            subCoefs{3} = rand(height/(dec^2),width/(dec^2));
            subCoefs{4} = rand(height/(dec^2),width/(dec^2));
            subCoefs{5} = rand(height/(dec^2),width/(dec^2));
            subCoefs{6} = rand(height/(dec^2),width/(dec^2));
            subCoefs{7} = rand(height/(dec^2),width/(dec^2));
            subCoefs{8} = rand(height/(dec),width/(dec));
            subCoefs{9} = rand(height/(dec),width/(dec));
            subCoefs{10} = rand(height/(dec),width/(dec));
            subCoefs{11} = rand(height/(dec),width/(dec));
            subCoefs{12} = rand(height/(dec),width/(dec));
            subCoefs{13} = rand(height/(dec),width/(dec));
            subCoefs{14} = rand(height/(dec),width/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            phase = 1; % for phase adjustment required experimentaly
            subsubCoefs = cell(ch,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                imgExpctd = imfilter(...
                    upsample(...
                    upsample(subsubCoefs{1}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],1),'cir');
                for iSubSub = 2:ch
                    iSubband = (iLevel-1)*(ch-1)+iSubSub;
                    subbandImg = imfilter(...
                        upsample(...
                        upsample(subCoefs{iSubband}.',dec,phase).',...
                        dec,phase),step(lppufb,[],[],iSubSub),'cir');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(...
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch8Ord00Level2
        function testStepDec22Ch9Ord00Level2(testCase)
            
            dec = 2;
            ch = 9;
            ord = 0;
            height = 32;
            width = 32;
            nLevels = 2;
            subCoefs = cell(nLevels*(ch-1)+1,1);
            subCoefs{1} = rand(height/(dec^2),width/(dec^2));
            subCoefs{2} = rand(height/(dec^2),width/(dec^2));
            subCoefs{3} = rand(height/(dec^2),width/(dec^2));
            subCoefs{4} = rand(height/(dec^2),width/(dec^2));
            subCoefs{5} = rand(height/(dec^2),width/(dec^2));
            subCoefs{6} = rand(height/(dec^2),width/(dec^2));
            subCoefs{7} = rand(height/(dec^2),width/(dec^2));
            subCoefs{8} = rand(height/(dec^2),width/(dec^2));
            subCoefs{9} = rand(height/(dec^2),width/(dec^2));
            subCoefs{10} = rand(height/(dec),width/(dec));
            subCoefs{11} = rand(height/(dec),width/(dec));
            subCoefs{12} = rand(height/(dec),width/(dec));
            subCoefs{13} = rand(height/(dec),width/(dec));
            subCoefs{14} = rand(height/(dec),width/(dec));
            subCoefs{15} = rand(height/(dec),width/(dec));
            subCoefs{16} = rand(height/(dec),width/(dec));
            subCoefs{17} = rand(height/(dec),width/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            phase = 1; % for phase adjustment required experimentaly
            subsubCoefs = cell(ch,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                imgExpctd = imfilter(...
                    upsample(...
                    upsample(subsubCoefs{1}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],1),'cir');
                for iSubSub = 2:ch
                    iSubband = (iLevel-1)*(ch-1)+iSubSub;
                    subbandImg = imfilter(...
                        upsample(...
                        upsample(subCoefs{iSubband}.',dec,phase).',...
                        dec,phase),step(lppufb,[],[],iSubSub),'cir');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(...
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec11Ch4Ord22Level1
        function testStepDec11Ch5Ord22Level1(testCase)
            
            dec = 1;
            ch = 5;
            ord = 2;
            height = 32;
            width = 32;
            %nLevels = 1;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(ch,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 0; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(...
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = dec;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch4Ord22Level1
        function testStepDec22Ch5Ord22Level1(testCase)
            
            dec = 2;
            ch = 5;
            ord = 2;
            height = 32;
            width = 32;
            %nLevels = 1;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(ch,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(...
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = dec;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch6Ord22Level1
        function testStepDec22Ch7Ord22Level1(testCase)
            
            dec = 2;
            ch = 7;
            ord = 2;
            height = 32;
            width = 32;
            % nLevels = 1;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(ch,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(...
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = dec;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch8Ord22Level1
        function testStepDec22Ch9Ord22Level1(testCase)
            
            dec = 2;
            ch = 9;
            ord = 2;
            height = 32;
            width = 32;
            %nLevels = 1;
            subCoefs = cell(ch,1);
            coefs = zeros(1,height*width);
            scales = zeros(ch,2);
            sIdx = 1;
            for iSubband = 1:ch
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:ch
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(...
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border = dec;
            imgExpctd = imgExpctd(border+1:end-border,border+1:end-border); % ignore border
            imgActual = imgActual(border+1:end-border,border+1:end-border); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec11Ch4Ord22Level2PeriodicExt
        function testStepDec11Ch5Ord22Level2PeriodicExt(testCase)
            
            dec = 1;
            ch = 5;
            ord = 2;
            height = 32;
            width = 32;
            nLevels = 2;
            subCoefs = cell(nLevels*(ch-1)+1,1);
            subCoefs{1} = rand(height/(dec^2),width/(dec^2));
            subCoefs{2} = rand(height/(dec^2),width/(dec^2));
            subCoefs{3} = rand(height/(dec^2),width/(dec^2));
            subCoefs{4} = rand(height/(dec^2),width/(dec^2));
            subCoefs{5} = rand(height/(dec^2),width/(dec^2));
            subCoefs{6} = rand(height/(dec),width/(dec));
            subCoefs{7} = rand(height/(dec),width/(dec));
            subCoefs{8} = rand(height/(dec),width/(dec));
            subCoefs{9} = rand(height/(dec),width/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            phase = 0; % for phase adjustment required experimentaly
            subsubCoefs = cell(ch,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                imgExpctd = imfilter(...
                    upsample(...
                    upsample(subsubCoefs{1}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],1),'cir');
                for iSubSub = 2:ch
                    iSubband = (iLevel-1)*(ch-1)+iSubSub;
                    subbandImg = imfilter(...
                        upsample(...
                        upsample(subCoefs{iSubband}.',dec,phase).',...
                        dec,phase),step(lppufb,[],[],iSubSub),'cir');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(...
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch4Ord22Level2PeriodicExt
        function testStepDec22Ch5Ord22Level2PeriodicExt(testCase)
            
            dec = 2;
            ch = 5;
            ord = 2;
            height = 32;
            width = 32;
            nLevels = 2;
            subCoefs = cell(nLevels*(ch-1)+1,1);
            subCoefs{1} = rand(height/(dec^2),width/(dec^2));
            subCoefs{2} = rand(height/(dec^2),width/(dec^2));
            subCoefs{3} = rand(height/(dec^2),width/(dec^2));
            subCoefs{4} = rand(height/(dec^2),width/(dec^2));
            subCoefs{5} = rand(height/(dec^2),width/(dec^2));
            subCoefs{6} = rand(height/(dec),width/(dec));
            subCoefs{7} = rand(height/(dec),width/(dec));
            subCoefs{8} = rand(height/(dec),width/(dec));
            subCoefs{9} = rand(height/(dec),width/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            phase = 1; % for phase adjustment required experimentaly
            subsubCoefs = cell(ch,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                imgExpctd = imfilter(...
                    upsample(...
                    upsample(subsubCoefs{1}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],1),'cir');
                for iSubSub = 2:ch
                    iSubband = (iLevel-1)*(ch-1)+iSubSub;
                    subbandImg = imfilter(...
                        upsample(...
                        upsample(subCoefs{iSubband}.',dec,phase).',...
                        dec,phase),step(lppufb,[],[],iSubSub),'cir');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(...
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch6Ord22Level2PeriodicExt
        function testStepDec22Ch7Ord22Level2PeriodicExt(testCase)
            
            dec = 2;
            ch = 7;
            ord = 2;
            height = 32;
            width = 32;
            nLevels = 2;
            subCoefs = cell(nLevels*(ch-1)+1,1);
            subCoefs{1} = rand(height/(dec^2),width/(dec^2));
            subCoefs{2} = rand(height/(dec^2),width/(dec^2));
            subCoefs{3} = rand(height/(dec^2),width/(dec^2));
            subCoefs{4} = rand(height/(dec^2),width/(dec^2));
            subCoefs{5} = rand(height/(dec^2),width/(dec^2));
            subCoefs{6} = rand(height/(dec^2),width/(dec^2));
            subCoefs{7} = rand(height/(dec^2),width/(dec^2));
            subCoefs{8} = rand(height/(dec),width/(dec));
            subCoefs{9} = rand(height/(dec),width/(dec));
            subCoefs{10} = rand(height/(dec),width/(dec));
            subCoefs{11} = rand(height/(dec),width/(dec));
            subCoefs{12} = rand(height/(dec),width/(dec));
            subCoefs{13} = rand(height/(dec),width/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            phase = 1; % for phase adjustment required experimentaly
            subsubCoefs = cell(ch,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                imgExpctd = imfilter(...
                    upsample(...
                    upsample(subsubCoefs{1}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],1),'cir');
                for iSubSub = 2:ch
                    iSubband = (iLevel-1)*(ch-1)+iSubSub;
                    subbandImg = imfilter(...
                        upsample(...
                        upsample(subCoefs{iSubband}.',dec,phase).',...
                        dec,phase),step(lppufb,[],[],iSubSub),'cir');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(...
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch8Ord22Level2PeriodicExt
        function testStepDec22Ch9Ord22Level2PeriodicExt(testCase)
            
            dec = 2;
            ch = 9;
            ord = 2;
            height = 32;
            width = 32;
            nLevels = 2;
            subCoefs = cell(nLevels*(ch-1)+1,1);
            subCoefs{1} = rand(height/(dec^2),width/(dec^2));
            subCoefs{2} = rand(height/(dec^2),width/(dec^2));
            subCoefs{3} = rand(height/(dec^2),width/(dec^2));
            subCoefs{4} = rand(height/(dec^2),width/(dec^2));
            subCoefs{5} = rand(height/(dec^2),width/(dec^2));
            subCoefs{6} = rand(height/(dec^2),width/(dec^2));
            subCoefs{7} = rand(height/(dec^2),width/(dec^2));
            subCoefs{8} = rand(height/(dec^2),width/(dec^2));
            subCoefs{9} = rand(height/(dec^2),width/(dec^2));
            subCoefs{10} = rand(height/(dec),width/(dec));
            subCoefs{11} = rand(height/(dec),width/(dec));
            subCoefs{12} = rand(height/(dec),width/(dec));
            subCoefs{13} = rand(height/(dec),width/(dec));
            subCoefs{14} = rand(height/(dec),width/(dec));
            subCoefs{15} = rand(height/(dec),width/(dec));
            subCoefs{16} = rand(height/(dec),width/(dec));
            subCoefs{17} = rand(height/(dec),width/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            phase = 1; % for phase adjustment required experimentaly
            subsubCoefs = cell(ch,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                imgExpctd = imfilter(...
                    upsample(...
                    upsample(subsubCoefs{1}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],1),'cir');
                for iSubSub = 2:ch
                    iSubband = (iLevel-1)*(ch-1)+iSubSub;
                    subbandImg = imfilter(...
                        upsample(...
                        upsample(subCoefs{iSubband}.',dec,phase).',...
                        dec,phase),step(lppufb,[],[],iSubSub),'cir');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(...
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch9Ord44Level3PeriodicExt
        function testStepDec22Ch9Ord44Level3PeriodicExt(testCase)
            
            dec = 2;
            ch = 9;
            ord = 4;
            height = 64;
            width = 64;
            nLevels = 3;
            subCoefs = cell(nLevels*(ch-1)+1,1);
            subCoefs{1} = rand(height/(dec^3),width/(dec^3));
            subCoefs{2} = rand(height/(dec^3),width/(dec^3));
            subCoefs{3} = rand(height/(dec^3),width/(dec^3));
            subCoefs{4} = rand(height/(dec^3),width/(dec^3));
            subCoefs{5} = rand(height/(dec^3),width/(dec^3));
            subCoefs{6} = rand(height/(dec^3),width/(dec^3));
            subCoefs{7} = rand(height/(dec^3),width/(dec^3));
            subCoefs{8} = rand(height/(dec^3),width/(dec^3));
            subCoefs{9} = rand(height/(dec^3),width/(dec^3));
            subCoefs{10} = rand(height/(dec^2),width/(dec^2));
            subCoefs{11} = rand(height/(dec^2),width/(dec^2));
            subCoefs{12} = rand(height/(dec^2),width/(dec^2));
            subCoefs{13} = rand(height/(dec^2),width/(dec^2));
            subCoefs{14} = rand(height/(dec^2),width/(dec^2));
            subCoefs{15} = rand(height/(dec^2),width/(dec^2));
            subCoefs{16} = rand(height/(dec^2),width/(dec^2));
            subCoefs{17} = rand(height/(dec^2),width/(dec^2));
            subCoefs{18} = rand(height/(dec),width/(dec));
            subCoefs{19} = rand(height/(dec),width/(dec));
            subCoefs{20} = rand(height/(dec),width/(dec));
            subCoefs{21} = rand(height/(dec),width/(dec));
            subCoefs{22} = rand(height/(dec),width/(dec));
            subCoefs{23} = rand(height/(dec),width/(dec));
            subCoefs{24} = rand(height/(dec),width/(dec));
            subCoefs{25} = rand(height/(dec),width/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            phase = 1; % for phase adjustment required experimentaly
            subsubCoefs = cell(ch,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                imgExpctd = imfilter(...
                    upsample(...
                    upsample(subsubCoefs{1}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],1),'cir');
                for iSubSub = 2:ch
                    iSubband = (iLevel-1)*(ch-1)+iSubSub;
                    subbandImg = imfilter(...
                        upsample(...
                        upsample(subCoefs{iSubband}.',dec,phase).',...
                        dec,phase),step(lppufb,[],[],iSubSub),'cir');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(...
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec11Ch7Ord88Level3PeriodicExt
        function testStepDec11Ch9Ord88Level3PeriodicExt(testCase)
            
            dec = 1;
            ch = 9;
            ord = 8;
            height = 64;
            width = 64;
            nLevels = 3;
            subCoefs = cell(nLevels*(ch-1)+1,1);
            subCoefs{1} = rand(height/(dec^3),width/(dec^3));
            subCoefs{2} = rand(height/(dec^3),width/(dec^3));
            subCoefs{3} = rand(height/(dec^3),width/(dec^3));
            subCoefs{4} = rand(height/(dec^3),width/(dec^3));
            subCoefs{5} = rand(height/(dec^3),width/(dec^3));
            subCoefs{6} = rand(height/(dec^3),width/(dec^3));
            subCoefs{7} = rand(height/(dec^3),width/(dec^3));
            subCoefs{8} = rand(height/(dec^3),width/(dec^3));
            subCoefs{9} = rand(height/(dec^3),width/(dec^3));
            subCoefs{10} = rand(height/(dec^2),width/(dec^2));
            subCoefs{11} = rand(height/(dec^2),width/(dec^2));
            subCoefs{12} = rand(height/(dec^2),width/(dec^2));
            subCoefs{13} = rand(height/(dec^2),width/(dec^2));
            subCoefs{14} = rand(height/(dec^2),width/(dec^2));
            subCoefs{15} = rand(height/(dec^2),width/(dec^2));
            subCoefs{16} = rand(height/(dec^2),width/(dec^2));
            subCoefs{17} = rand(height/(dec^2),width/(dec^2));
            subCoefs{18} = rand(height/(dec),width/(dec));
            subCoefs{19} = rand(height/(dec),width/(dec));
            subCoefs{20} = rand(height/(dec),width/(dec));
            subCoefs{21} = rand(height/(dec),width/(dec));
            subCoefs{22} = rand(height/(dec),width/(dec));
            subCoefs{23} = rand(height/(dec),width/(dec));
            subCoefs{24} = rand(height/(dec),width/(dec));
            subCoefs{25} = rand(height/(dec),width/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            phase = 0; % for phase adjustment required experimentaly
            subsubCoefs = cell(ch,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                imgExpctd = imfilter(...
                    upsample(...
                    upsample(subsubCoefs{1}.',dec,phase).',...
                    dec,phase),step(lppufb,[],[],1),'cir');
                for iSubSub = 2:ch
                    iSubband = (iLevel-1)*(ch-1)+iSubSub;
                    subbandImg = imfilter(...
                        upsample(...
                        upsample(subCoefs{iSubband}.',dec,phase).',...
                        dec,phase),step(lppufb,[],[],iSubSub),'cir');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(...
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec22Ch32Ord00(testCase)
            
            dec = 2;
            decch = [dec dec 3 2];
            nChs = sum(decch(3:4));
            height = 16;
            width = 16;
            subCoefs = rand(height*nChs/decch(1),width/decch(2));
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iCh = 1:nChs
                subImg = subCoefs(iCh:nChs:end,:);
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iCh,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end));
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            E = step(lppufb,[],[]);
            
            % Expected values
            fun = @(x) reshape(flipud(E.'*x.data(:)),decch(1),decch(2));
            imgExpctd = blockproc(subCoefs,[nChs 1],fun);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec22Ch32Ord22(testCase)
            
            dec = 2;
            decch = [dec dec 3 2];
            nChs = sum(decch(3:4));
            ord = 2;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',decch(2),phase).',...
                    decch(1),phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(decch(1)+1:end-decch(1),decch(2)+1:end-decch(2)); % ignore border
            imgActual = imgActual(decch(1)+1:end-decch(1),decch(2)+1:end-decch(2)); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec22Ch32Ord22PeriodicExt(testCase)
            
            dec = 2;
            decch = [ dec dec 3 2 ];
            nChs = sum(decch(3:4));
            ord = 2;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',decch(2),phase).',...
                    decch(1),phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec22Ch32Ord44(testCase)
            
            dec = 2;
            decch = [dec dec 3 2];
            nChs = sum(decch(3:4));
            ord = 4;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',decch(2),phase).',...
                    decch(1),phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(2*decch(1)+1:end-2*decch(1),2*decch(2)+1:end-2*decch(2)); % ignore border
            imgActual = imgActual(2*decch(1)+1:end-2*decch(1),2*decch(2)+1:end-2*decch(2)); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec22Ch32Ord44PeriodicExt(testCase)
            
            dec = 2;
            decch = [ dec dec 3 2];
            nChs = sum(decch(3:4));
            ord = 4;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',decch(2),phase).',...
                    decch(1),phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec22Ch42Ord00(testCase)
            
            dec  = 2;
            decch = [ dec dec 4 2];
            nChs = sum(decch(3:4));
            height = 16;
            width = 16;
            subCoefs = rand(height*nChs/decch(1),width/decch(2));
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iCh = 1:nChs
                subImg = subCoefs(iCh:nChs:end,:);
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iCh,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end));
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            E = step(lppufb,[],[]);
            
            
            % Expected values
            fun = @(x) reshape(flipud(E.'*x.data(:)),decch(1),decch(2));
            imgExpctd = blockproc(subCoefs,[nChs 1],fun);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec22Ch42Ord22(testCase)
            
            dec = 2;
            decch = [dec dec 4 2];
            nChs = sum(decch(3:4));
            ord = 2;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',decch(2),phase).',...
                    decch(1),phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(decch(1)+1:end-decch(1),decch(2)+1:end-decch(2)); % ignore border
            imgActual = imgActual(decch(1)+1:end-decch(1),decch(2)+1:end-decch(2)); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec22Ch42Ord22PeriodicExt(testCase)
            
            dec = 2;
            decch = [ dec dec 4 2 ];
            nChs = sum(decch(3:4));
            ord = 2;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',decch(2),phase).',...
                    decch(1),phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec22Ch42Ord44(testCase)
            
            dec = 2;
            decch = [dec dec 4 2];
            nChs = sum(decch(3:4));
            ord = 4;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',decch(2),phase).',...
                    decch(1),phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(2*decch(1)+1:end-2*decch(1),2*decch(2)+1:end-2*decch(2)); % ignore border
            imgActual = imgActual(2*decch(1)+1:end-2*decch(1),2*decch(2)+1:end-2*decch(2)); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec22Ch42Ord44PeriodicExt(testCase)
            
            dec = 2;
            decch = [dec dec 4 2];
            nChs = sum(decch(3:4));
            ord = 4;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',decch(2),phase).',...
                    decch(1),phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch32Ord22Level1
        function testStepDec22Ch32Ord22Level1(testCase)
            
            dec = 2;
            decch = [dec dec 3 2];
            nChs = sum(decch(3:4));
            ord = 2;
            height = 32;
            width = 32;
            %nLevels = 1;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',decch(2),phase).',...
                    decch(1),phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border1 = decch(1);
            border2 = decch(2);
            imgExpctd = imgExpctd(border1+1:end-border1,border2+1:end-border2); % ignore border
            imgActual = imgActual(border1+1:end-border1,border2+1:end-border2); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        
        %Dec22Ch32Ord22Level2PeriodicExt
        function testStepDec32Ch5Ord22Level2PeriodicExt(testCase)
            
            decch = [2 2 3 2];
            nChs = sum(decch(3:4));
            ord = 2;
            height = 32;
            width = 32;
            nLevels = 2;
            subCoefs = cell(nLevels*(nChs-1)+1,1);
            subCoefs{1} = rand(height/(decch(1)^2),width/(decch(2)^2));
            subCoefs{2} = rand(height/(decch(1)^2),width/(decch(2)^2));
            subCoefs{3} = rand(height/(decch(1)^2),width/(decch(2)^2));
            subCoefs{4} = rand(height/(decch(1)^2),width/(decch(2)^2));
            subCoefs{5} = rand(height/(decch(1)^2),width/(decch(2)^2));
            subCoefs{6} = rand(height/(decch(1)),width/(decch(2)));
            subCoefs{7} = rand(height/(decch(1)),width/(decch(2)));
            subCoefs{8} = rand(height/(decch(1)),width/(decch(2)));
            subCoefs{9} = rand(height/(decch(1)),width/(decch(2)));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            phase = 1; % for phase adjustment required experimentaly
            subsubCoefs = cell(nChs,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                imgExpctd = imfilter(...
                    upsample(...
                    upsample(subsubCoefs{1}.',decch(2),phase).',...
                    decch(1),phase),step(lppufb,[],[],1),'cir');
                for iSubSub = 2:nChs
                    iSubband = (iLevel-1)*(nChs-1)+iSubSub;
                    subbandImg = imfilter(...
                        upsample(...
                        upsample(subCoefs{iSubband}.',decch(2),phase).',...
                        decch(2),phase),step(lppufb,[],[],iSubSub),'cir');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch32Ord22Level1
        function testStepDec42Ch5Ord22Level1(testCase)
            
            dec = 2;
            decch = [dec dec 4 2];
            nChs = sum(decch(3:4));
            ord = 2;
            height = 32;
            width = 32;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',decch(2),phase).',...
                    decch(1),phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border1 = decch(1);
            border2 = decch(2);
            imgExpctd = imgExpctd(border1+1:end-border1,border2+1:end-border2); % ignore border
            imgActual = imgActual(border1+1:end-border1,border2+1:end-border2); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch32Ord22Level2PeriodicExt
        function testStepDec22Ch42Ord22Level2PeriodicExt(testCase)
            
            decch = [2 2 4 2];
            nChs = sum(decch(3:4));
            ord = 2;
            height = 32;
            width = 32;
            nLevels = 2;
            subCoefs = cell(nLevels*(nChs-1)+1,1);
            subCoefs{1} = rand(height/(decch(1)^2),width/(decch(2)^2));
            subCoefs{2} = rand(height/(decch(1)^2),width/(decch(2)^2));
            subCoefs{3} = rand(height/(decch(1)^2),width/(decch(2)^2));
            subCoefs{4} = rand(height/(decch(1)^2),width/(decch(2)^2));
            subCoefs{5} = rand(height/(decch(1)^2),width/(decch(2)^2));
            subCoefs{6} = rand(height/(decch(1)^2),width/(decch(2)^2));
            subCoefs{7} = rand(height/(decch(1)),width/(decch(2)));
            subCoefs{8} = rand(height/(decch(1)),width/(decch(2)));
            subCoefs{9} = rand(height/(decch(1)),width/(decch(2)));
            subCoefs{10} = rand(height/(decch(1)),width/(decch(2)));
            subCoefs{11} = rand(height/(decch(1)),width/(decch(2)));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            phase = 1; % for phase adjustment required experimentaly
            subsubCoefs = cell(nChs,1);
            subsubCoefs{1} = subCoefs{1};
            for iLevel = 1:nLevels
                imgExpctd = imfilter(...
                    upsample(...
                    upsample(subsubCoefs{1}.',decch(2),phase).',...
                    decch(1),phase),step(lppufb,[],[],1),'cir');
                for iSubSub = 2:nChs
                    iSubband = (iLevel-1)*(nChs-1)+iSubSub;
                    subbandImg = imfilter(...
                        upsample(...
                        upsample(subCoefs{iSubband}.',decch(2),phase).',...
                        decch(2),phase),step(lppufb,[],[],iSubSub),'cir');
                    imgExpctd = imgExpctd + subbandImg;
                end
                subsubCoefs{1}=imgExpctd;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        function testSetLpPuFb2dDec22Ch62Ord44(testCase)
            
            dec = 2;
            ch = [ 6 2 ];
            ord = 4;
            height = 32;
            width = 32;
            subCoefs{1} = rand(height/(dec),width/(dec));
            subCoefs{2} = rand(height/(dec),width/(dec));
            subCoefs{3} = rand(height/(dec),width/(dec));
            subCoefs{4} = rand(height/(dec),width/(dec));
            subCoefs{5} = rand(height/(dec),width/(dec));
            subCoefs{6} = rand(height/(dec),width/(dec));
            subCoefs{7} = rand(height/(dec),width/(dec));
            subCoefs{8} = rand(height/(dec),width/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(...
                'LpPuFb2d',lppufb);
            imgPre = step(testCase.synthesizer,coefs,scales);
            
              % Update lppufb
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            imgPst = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan;
            diff = norm(imgPst(:)-imgPre(:));
            testCase.verifyTrue(diff<1e-15);
                        
            % ReInstantiation of target class
            testCase.synthesizer = NsoltSynthesis2dSystem(...
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Termination');
            imgPst = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan;
            diff = norm(imgPst(:)-imgPre(:));
            testCase.verifyThat(diff,IsGreaterThan(0));
        end
        
        function testSetLpPuFb2dDec22Ch52Ord44(testCase)
            
            dec = 2;
            ch = [ 5 2 ];
            ord = 4;
            height = 32;
            width = 32;
            subCoefs{1} = rand(height/(dec),width/(dec));
            subCoefs{2} = rand(height/(dec),width/(dec));
            subCoefs{3} = rand(height/(dec),width/(dec));
            subCoefs{4} = rand(height/(dec),width/(dec));
            subCoefs{5} = rand(height/(dec),width/(dec));
            subCoefs{6} = rand(height/(dec),width/(dec));
            subCoefs{7} = rand(height/(dec),width/(dec));
            nSubbands = length(subCoefs);
            scales = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scales(iSubband,:) = size(subCoefs{iSubband});
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                coefs(sIdx:eIdx) = subCoefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord]);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(...
                'LpPuFb2d',lppufb);
            imgPre = step(testCase.synthesizer,coefs,scales);
            
            % Update lppufb
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            imgPst = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan;
            diff = norm(imgPst(:)-imgPre(:));
            testCase.verifyTrue(diff<1e-15);
            
            % ReInstantiation of target class
            testCase.synthesizer = NsoltSynthesis2dSystem(...
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Termination');
            imgPst = step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan;
            diff = norm(imgPst(:)-imgPre(:));
            testCase.verifyThat(diff,IsGreaterThan(0));
        end
        
        % Test
        function testCloneTypeII(testCase)
            
            dec = [ 2 2 ];
            ch =  [ 5 3 ];
            ord = [ 4 4 ];
            height = 64;
            width  = 64;
            coefs = rand(sum(ch)/prod(dec)*height*width,1);
            scales = repmat([height/dec(1) width/dec(2)],[sum(ch) 1]);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord,...
                'OutputMode','ParameterMatrixSet');
            
            % Instantiation of target class
            testCase.synthesizer = NsoltSynthesis2dSystem(...
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Termination');

            % Clone
            cloneSynthesizer = clone(testCase.synthesizer);

            % Evaluation
            testCase.verifyEqual(cloneSynthesizer,testCase.synthesizer);
            testCase.verifyFalse(cloneSynthesizer == testCase.synthesizer);
            prpOrg = get(testCase.synthesizer,'LpPuFb2d');
            prpCln = get(cloneSynthesizer,'LpPuFb2d');
            testCase.verifyEqual(prpCln,prpOrg);
            testCase.verifyFalse(prpCln == prpOrg);
            %
            recImgExpctd = step(testCase.synthesizer,coefs,scales);
            recImgActual = step(cloneSynthesizer,coefs,scales);
            testCase.assertEqual(recImgActual,recImgExpctd);
            
        end

        % Test
        function testStepDec22Ch23Ord00(testCase)
            
            dec = 2;
            decch = [dec dec 2 3];
            nChs = sum(decch(3:4));
            height = 16;
            width = 16;
            subCoefs = rand(height*nChs/decch(1),width/decch(2));
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iCh = 1:nChs
                subImg = subCoefs(iCh:nChs:end,:);
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iCh,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end));
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            E = step(lppufb,[],[]);
            
            % Expected values
            fun = @(x) reshape(flipud(E.'*x.data(:)),decch(1),decch(2));
            imgExpctd = blockproc(subCoefs,[nChs 1],fun);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec22Ch23Ord22(testCase)
            
            dec = 2;
            decch = [dec dec 2 3];
            nChs = sum(decch(3:4));
            ord = 2;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',decch(2),phase).',...
                    decch(1),phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(decch(1)+1:end-decch(1),decch(2)+1:end-decch(2)); % ignore border
            imgActual = imgActual(decch(1)+1:end-decch(1),decch(2)+1:end-decch(2)); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec22Ch23Ord22PeriodicExt(testCase)
            
            dec = 2;
            decch = [ dec dec 2 3 ];
            nChs = sum(decch(3:4));
            ord = 2;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',decch(2),phase).',...
                    decch(1),phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec22Ch23Ord44(testCase)
            
            dec = 2;
            decch = [dec dec 2 3];
            nChs = sum(decch(3:4));
            ord = 4;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',decch(2),phase).',...
                    decch(1),phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(2*decch(1)+1:end-2*decch(1),2*decch(2)+1:end-2*decch(2)); % ignore border
            imgActual = imgActual(2*decch(1)+1:end-2*decch(1),2*decch(2)+1:end-2*decch(2)); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec22Ch23Ord44PeriodicExt(testCase)
            
            dec = 2;
            decch = [ dec dec 2 3];
            nChs = sum(decch(3:4));
            ord = 4;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',decch(2),phase).',...
                    decch(1),phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec22Ch24Ord00(testCase)
            
            dec  = 2;
            decch = [ dec dec 2 4 ];
            nChs = sum(decch(3:4));
            height = 16;
            width = 16;
            subCoefs = rand(height*nChs/decch(1),width/decch(2));
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iCh = 1:nChs
                subImg = subCoefs(iCh:nChs:end,:);
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iCh,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end));
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            E = step(lppufb,[],[]);
            
            % Expected values
            fun = @(x) reshape(flipud(E.'*x.data(:)),decch(1),decch(2));
            imgExpctd = blockproc(subCoefs,[nChs 1],fun);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec22Ch24Ord22(testCase)
            
            dec = 2;
            decch = [dec dec 2 4];
            nChs = sum(decch(3:4));
            ord = 2;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',decch(2),phase).',...
                    decch(1),phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(decch(1)+1:end-decch(1),decch(2)+1:end-decch(2)); % ignore border
            imgActual = imgActual(decch(1)+1:end-decch(1),decch(2)+1:end-decch(2)); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec22Ch24Ord22PeriodicExt(testCase)
            
            dec = 2;
            decch = [ dec dec 2 4 ];
            nChs = sum(decch(3:4));
            ord = 2;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',decch(2),phase).',...
                    decch(1),phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec22Ch24Ord44(testCase)
            
            dec = 2;
            decch = [dec dec 2 4];
            nChs = sum(decch(3:4));
            ord = 4;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',decch(2),phase).',...
                    decch(1),phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            imgExpctd = imgExpctd(2*decch(1)+1:end-2*decch(1),2*decch(2)+1:end-2*decch(2)); % ignore border
            imgActual = imgActual(2*decch(1)+1:end-2*decch(1),2*decch(2)+1:end-2*decch(2)); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec22Ch24Ord44PeriodicExt(testCase)
            
            dec = 2;
            decch = [dec dec 2 4];
            nChs = sum(decch(3:4));
            ord = 4;
            height = 16;
            width = 16;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',decch(2),phase).',...
                    decch(1),phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb,...
                'BoundaryOperation','Circular');
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Dec22Ch32Ord22Level1
        function testStepDec22Ch23Ord22Level1(testCase)
            
            dec = 2;
            decch = [dec dec 2 3];
            nChs = sum(decch(3:4));
            ord = 2;
            height = 32;
            width = 32;
            %nLevels = 1;
            subCoefs = cell(nChs,1);
            coefs = zeros(1,height*width);
            scales = zeros(nChs,2);
            sIdx = 1;
            for iSubband = 1:nChs
                subImg = rand(height/dec,width/dec);
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = 1; % for phase adjustment required experimentaly
            for iSubband = 1:nChs
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',decch(2),phase).',...
                    decch(1),phase),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border1 = decch(1);
            border2 = decch(2);
            imgExpctd = imgExpctd(border1+1:end-border1,border2+1:end-border2); % ignore border
            imgActual = imgActual(border1+1:end-border1,border2+1:end-border2); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Test
        function testStepDec12Ch22Ord22Level1(testCase)
            
            nDecs = [ 1 2 ];
            nChs  = [ 2 2 ];
            nOrds = [ 2 2 ];            
            nch_ = sum(nChs);
            height = 32;
            width = 32;
            %nLevels = 1;
            subCoefs = cell(nch_,1);
            coefs = zeros(1,height*width);
            scales = zeros(nch_,2);
            sIdx = 1;
            for iSubband = 1:nch_
                subImg = rand(height/nDecs(1),width/nDecs(2));
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',nOrds);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = nDecs-1; % for phase adjustment required experimentaly
            for iSubband = 1:nch_
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',nDecs(2),phase(2)).',...
                    nDecs(1),phase(1)),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border1 = nDecs(1);
            border2 = nDecs(2);
            imgExpctd = imgExpctd(border1+1:end-border1,border2+1:end-border2); % ignore border
            imgActual = imgActual(border1+1:end-border1,border2+1:end-border2); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Test
        function testStepDec21Ch22Ord22Level1(testCase)
            
            nDecs = [ 2 1 ];
            nChs  = [ 2 2 ];
            nOrds = [ 2 2 ];            
            nch_ = sum(nChs);
            height = 32;
            width = 32;
            %nLevels = 1;
            subCoefs = cell(nch_,1);
            coefs = zeros(1,height*width);
            scales = zeros(nch_,2);
            sIdx = 1;
            for iSubband = 1:nch_
                subImg = rand(height/nDecs(1),width/nDecs(2));
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',nOrds);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = nDecs-1; % for phase adjustment required experimentaly
            for iSubband = 1:nch_
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',nDecs(2),phase(2)).',...
                    nDecs(1),phase(1)),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border1 = nDecs(1);
            border2 = nDecs(2);
            imgExpctd = imgExpctd(border1+1:end-border1,border2+1:end-border2); % ignore border
            imgActual = imgActual(border1+1:end-border1,border2+1:end-border2); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Test
        function testStepDec12Ch23Ord22Level1(testCase)
            
            nDecs = [ 1 2 ];
            nChs  = [ 2 3 ];
            nOrds = [ 2 2 ];            
            nch_ = sum(nChs);
            height = 32;
            width = 32;
            %nLevels = 1;
            subCoefs = cell(nch_,1);
            coefs = zeros(1,height*width);
            scales = zeros(nch_,2);
            sIdx = 1;
            for iSubband = 1:nch_
                subImg = rand(height/nDecs(1),width/nDecs(2));
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',nOrds);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = nDecs-1; % for phase adjustment required experimentaly
            for iSubband = 1:nch_
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',nDecs(2),phase(2)).',...
                    nDecs(1),phase(1)),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border1 = nDecs(1);
            border2 = nDecs(2);
            imgExpctd = imgExpctd(border1+1:end-border1,border2+1:end-border2); % ignore border
            imgActual = imgActual(border1+1:end-border1,border2+1:end-border2); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Test
        function testStepDec21Ch23Ord22Level1(testCase)
            
            nDecs = [ 2 1 ];
            nChs  = [ 2 3 ];
            nOrds = [ 2 2 ];            
            nch_ = sum(nChs);
            height = 32;
            width = 32;
            %nLevels = 1;
            subCoefs = cell(nch_,1);
            coefs = zeros(1,height*width);
            scales = zeros(nch_,2);
            sIdx = 1;
            for iSubband = 1:nch_
                subImg = rand(height/nDecs(1),width/nDecs(2));
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',nOrds);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = nDecs-1; % for phase adjustment required experimentaly
            for iSubband = 1:nch_
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',nDecs(2),phase(2)).',...
                    nDecs(1),phase(1)),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border1 = nDecs(1);
            border2 = nDecs(2);
            imgExpctd = imgExpctd(border1+1:end-border1,border2+1:end-border2); % ignore border
            imgActual = imgActual(border1+1:end-border1,border2+1:end-border2); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
                %Test
        function testStepDec12Ch32Ord22Level1(testCase)
            
            nDecs = [ 1 2 ];
            nChs  = [ 3 2 ];
            nOrds = [ 2 2 ];            
            nch_ = sum(nChs);
            height = 32;
            width = 32;
            %nLevels = 1;
            subCoefs = cell(nch_,1);
            coefs = zeros(1,height*width);
            scales = zeros(nch_,2);
            sIdx = 1;
            for iSubband = 1:nch_
                subImg = rand(height/nDecs(1),width/nDecs(2));
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',nOrds);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = nDecs-1; % for phase adjustment required experimentaly
            for iSubband = 1:nch_
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',nDecs(2),phase(2)).',...
                    nDecs(1),phase(1)),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border1 = nDecs(1);
            border2 = nDecs(2);
            imgExpctd = imgExpctd(border1+1:end-border1,border2+1:end-border2); % ignore border
            imgActual = imgActual(border1+1:end-border1,border2+1:end-border2); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        %Test
        function testStepDec21Ch32Ord22Level1(testCase)
            
            nDecs = [ 2 1 ];
            nChs  = [ 3 2 ];
            nOrds = [ 2 2 ];            
            nch_ = sum(nChs);
            height = 32;
            width = 32;
            %nLevels = 1;
            subCoefs = cell(nch_,1);
            coefs = zeros(1,height*width);
            scales = zeros(nch_,2);
            sIdx = 1;
            for iSubband = 1:nch_
                subImg = rand(height/nDecs(1),width/nDecs(2));
                subCoefs{iSubband} = subImg;
                eIdx = sIdx + numel(subImg) - 1;
                coefs(sIdx:eIdx) = subImg(:).';
                scales(iSubband,:) = size(subImg);
                sIdx = eIdx + 1;
            end
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',nDecs,...
                'NumberOfChannels',nChs,...
                'PolyPhaseOrder',nOrds);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            imgExpctd = zeros(height,width);
            phase = nDecs-1; % for phase adjustment required experimentaly
            for iSubband = 1:nch_
                subbandImg = imfilter(...
                    upsample(...
                    upsample(subCoefs{iSubband}.',nDecs(2),phase(2)).',...
                    nDecs(1),phase(1)),step(lppufb,[],[],iSubband),'cir');
                imgExpctd = imgExpctd + subbandImg;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.synthesizer = NsoltSynthesis2dSystem(....
                'LpPuFb2d',lppufb);
            
            % Actual values
            imgActual = ...
                step(testCase.synthesizer,coefs,scales);
            
            % Evaluation
            testCase.verifySize(imgActual,size(imgExpctd),...
                'Actual image size is different from the expected one.');
            border1 = nDecs(1);
            border2 = nDecs(2);
            imgExpctd = imgExpctd(border1+1:end-border1,border2+1:end-border2); % ignore border
            imgActual = imgActual(border1+1:end-border1,border2+1:end-border2); % ignore border
            diff = max(abs(imgExpctd(:) - imgActual(:)));
            testCase.verifyEqual(imgActual,imgExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
"""
def permuteDctCoefs_(x):
    cee = x[:,0::2,0::2].reshape(x.size(0),-1)
    coo = x[:,1::2,1::2].reshape(x.size(0),-1)
    coe = x[:,1::2,0::2].reshape(x.size(0),-1)
    ceo = x[:,0::2,1::2].reshape(x.size(0),-1)
    return torch.cat((cee,coo,coe,ceo),dim=-1)

def permuteIdctCoefs_(x,block_size):
    coefs = x.view(-1,block_size[Direction.VERTICAL]*block_size[Direction.HORIZONTAL]) # x.view(-1,math.prod(block_size))
    decY_ = block_size[Direction.VERTICAL]
    decX_ = block_size[Direction.HORIZONTAL]
    chDecY = int(math.ceil(decY_/2.))
    chDecX = int(math.ceil(decX_/2.))
    fhDecY = int(math.floor(decY_/2.))
    fhDecX = int(math.floor(decX_/2.))
    nQDecsee = chDecY*chDecX
    nQDecsoo = fhDecY*fhDecX
    nQDecsoe = fhDecY*chDecX
    cee = coefs[:,:nQDecsee]
    coo = coefs[:,nQDecsee:nQDecsee+nQDecsoo]
    coe = coefs[:,nQDecsee+nQDecsoo:nQDecsee+nQDecsoo+nQDecsoe]
    ceo = coefs[:,nQDecsee+nQDecsoo+nQDecsoe:]
    nBlocks = coefs.size(0)
    value = torch.zeros(nBlocks,decY_,decX_,dtype=x.dtype).to(x.device)
    value[:,0::2,0::2] = cee.view(nBlocks,chDecY,chDecX)
    value[:,1::2,1::2] = coo.view(nBlocks,fhDecY,fhDecX)
    value[:,1::2,0::2] = coe.view(nBlocks,fhDecY,chDecX)
    value[:,0::2,1::2] = ceo.view(nBlocks,chDecY,fhDecX)
    return value

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
        X[:,:,:,ps:] = torch.roll(X[:,:,:,ps:],shifts=tuple(shift),dims=(0,1,2,3))
    else: # Sum channel
        X[:,:,:,:ps] = torch.roll(X[:,:,:,:ps],shifts=tuple(shift),dims=(0,1,2,3))
    return X

def intermediate_rotation(X,nchs,R):
    """
    Intermediate rotation
    """
    Y = X.clone()
    ps,pa = nchs
    nSamples = X.size(0)
    nrows = X.size(1)
    ncols = X.size(2)
    Za = R @ X[:,:,:,ps:].view(-1,pa).T 
    Y[:,:,:,ps:] = Za.T.view(nSamples,nrows,ncols,pa)
    return Y

if __name__ == '__main__':
    unittest.main()