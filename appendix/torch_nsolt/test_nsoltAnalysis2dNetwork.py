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
from nsoltAnalysis2dNetwork import NsoltAnalysis2dNetwork
from nsoltLayerExceptions import InvalidNumberOfChannels, InvalidPolyPhaseOrder, InvalidNumberOfVanishingMoments, InvalidNumberOfLevels
from nsoltUtility import Direction, dct_2d

nchs = [ [2, 2], [3, 3], [4, 4] ]
stride = [ [1, 1], [1, 2], [2, 1], [2, 2] ]
ppord = [ [0, 0], [0, 2], [2, 0], [2, 2], [4, 4] ]
datatype = [ torch.float, torch.double ]
height = [ 8, 16, 32 ]
width = [ 8, 16, 32 ]
nvm = [ 0, 1 ]
nlevels = [ 1, 2, 3 ]
isdevicetest = True

class NsoltAnalysis2dNetworkTestCase(unittest.TestCase):
    """
    NSOLTANLAYSIS2DNETWORKTESTCASE Test cases for NsoltAnalysis2dNetwork
    
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
        network = NsoltAnalysis2dNetwork(
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
        nComponents = 1
        nDecs = stride[0]*stride[1] #math.prod(stride)
        nChsTotal = sum(nchs)
        # Source (nSamples x nComponents x (Stride[0]xnRows) x (Stride[1]xnCols))
        X = torch.rand(nSamples,nComponents,height,width,dtype=datatype,device=device,requires_grad=True)
        
        # Expected values
        nrows = int(math.ceil(height/stride[Direction.VERTICAL])) #.astype(int)
        ncols = int(math.ceil(width/stride[Direction.HORIZONTAL])) #.astype(int)
        # Block DCT (nSamples x nComponents x nrows x ncols) x decV x decH
        arrayshape = stride.copy()
        arrayshape.insert(0,-1)
        Y = dct_2d(X.view(arrayshape))
        # Rearrange the DCT Coefs. (nSamples x nComponents x nrows x ncols) x (decV x decH)
        A = permuteDctCoefs_(Y)
        V = A.view(nSamples,nrows,ncols,nDecs)
        # nSamplex x nRows x nCols x nChs
        ps, pa = nchs
        W0 = torch.eye(ps,dtype=datatype).to(device)
        U0 = torch.eye(pa,dtype=datatype).to(device)
        ms,ma = int(math.ceil(nDecs/2.)), int(math.floor(nDecs/2.))        
        Zsa = torch.zeros(nChsTotal,nrows*ncols*nSamples,dtype=datatype)
        Zsa = Zsa.to(device)        
        Ys = V[:,:,:,:ms].view(-1,ms).T
        Zsa[:ps,:] = W0[:,:ms] @ Ys
        if ma > 0:
            Ya = V[:,:,:,ms:].view(-1,ma).T
            Zsa[ps:,:] = U0[:,:ma] @ Ya
        expctdZ = Zsa.T.view(nSamples,nrows,ncols,nChsTotal)

        # Instantiation of target class
        network = NsoltAnalysis2dNetwork(
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
            NsoltAnalysis2dNetwork(
                number_of_channels = [ps,ps+1],
                decimation_factor = stride
            )

        with self.assertRaises(InvalidNumberOfChannels):
            NsoltAnalysis2dNetwork(
                number_of_channels = [pa+1,pa],
                decimation_factor = stride
            )

    @parameterized.expand(
        list(itertools.product(nchs,stride,ppord))
    )
    def testNumberOfPolyPhaseOrderException(self,
        nchs,stride,ppord):
        with self.assertRaises(InvalidPolyPhaseOrder):
            NsoltAnalysis2dNetwork(
                polyphase_order = [ ppord[0]+1, ppord[1] ],
                number_of_channels = nchs,
                decimation_factor = stride
            )

        with self.assertRaises(InvalidPolyPhaseOrder):
            NsoltAnalysis2dNetwork(
                polyphase_order = [ ppord[0], ppord[1]+1 ],
                number_of_channels = nchs,
                decimation_factor = stride
            )

        with self.assertRaises(InvalidPolyPhaseOrder):
            NsoltAnalysis2dNetwork(
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
            NsoltAnalysis2dNetwork(
                number_of_channels = nchs,
                decimation_factor = stride,
                polyphase_order = ppord,
                number_of_vanishing_moments = nVm
            )

        nVm = 2
        with self.assertRaises(InvalidNumberOfVanishingMoments):
            NsoltAnalysis2dNetwork(
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
            NsoltAnalysis2dNetwork(
                number_of_channels = nchs,
                decimation_factor = stride,
                polyphase_order = ppord,
                number_of_levels = nlevels
            )

        nlevels = 0.5
        with self.assertRaises(InvalidNumberOfLevels):
            NsoltAnalysis2dNetwork(
                number_of_channels = nchs,
                decimation_factor = stride,
                polyphase_order = ppord,
                number_of_levels = nlevels
            )


    @parameterized.expand(
        list(itertools.product(nchs,stride,height,width,datatype))
    )
    def testForwardGrayScaleWithInitilization(self,
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
        nComponents = 1
        nDecs = stride[0]*stride[1] #math.prod(stride)
        nChsTotal = sum(nchs)
        # Source (nSamples x nComponents x (Stride[0]xnRows) x (Stride[1]xnCols))
        X = torch.rand(nSamples,nComponents,height,width,dtype=datatype,device=device,requires_grad=True)

        # Expected values
        nrows = int(math.ceil(height/stride[Direction.VERTICAL])) #.astype(int)
        ncols = int(math.ceil(width/stride[Direction.HORIZONTAL])) #.astype(int)
        # Block DCT (nSamples x nComponents x nrows x ncols) x decV x decH
        arrayshape = stride.copy()
        arrayshape.insert(0,-1)
        Y = dct_2d(X.view(arrayshape))
        # Rearrange the DCT Coefs. (nSamples x nComponents x nrows x ncols) x (decV x decH)
        A = permuteDctCoefs_(Y)
        V = A.view(nSamples,nrows,ncols,nDecs)
        # nSamples x nRows x nCols x nChs
        ps, pa = nchs
        angles = angle0*torch.ones(int((nChsTotal-2)*nChsTotal/4)).to(device) #,dtype=datatype)
        nAngsW = int(len(angles)/2)
        angsW,angsU = angles[:nAngsW],angles[nAngsW:]
        W0,U0 = gen(angsW),gen(angsU)        
        ms,ma = int(math.ceil(nDecs/2.)), int(math.floor(nDecs/2.))        
        Zsa = torch.zeros(nChsTotal,nrows*ncols*nSamples,dtype=datatype) 
        Zsa = Zsa.to(device)       
        Ys = V[:,:,:,:ms].view(-1,ms).T
        Zsa[:ps,:] = W0[:,:ms] @ Ys
        if ma > 0:
            Ya = V[:,:,:,ms:].view(-1,ma).T
            Zsa[ps:,:] = U0[:,:ma] @ Ya
        expctdZ = Zsa.T.view(nSamples,nrows,ncols,nChsTotal)

        # Instantiation of target class
        network = NsoltAnalysis2dNetwork(
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
        nComponents = 1
        nDecs = stride[0]*stride[1] #math.prod(stride)
        nChsTotal = sum(nchs)
        # Source (nSamples x nComponents x (Stride[0]xnRows) x (Stride[1]xnCols))
        X = torch.rand(nSamples,nComponents,height,width,dtype=datatype,device=device,requires_grad=True)
        
        # Expected values
        nrows = int(math.ceil(height/stride[Direction.VERTICAL])) #.astype(int)
        ncols = int(math.ceil(width/stride[Direction.HORIZONTAL])) #.astype(int)
        # Block DCT (nSamples x nComponents x nrows x ncols) x decV x decH
        arrayshape = stride.copy()
        arrayshape.insert(0,-1)
        Y = dct_2d(X.view(arrayshape))
        # Rearrange the DCT Coefs. (nSamples x nComponents x nrows x ncols) x (decV x decH)
        A = permuteDctCoefs_(Y)
        V = A.view(nSamples,nrows,ncols,nDecs)
        # nSamples x nRows x nCols x nChs
        ps, pa = nchs
        # Initial rotation
        W0 = torch.eye(ps,dtype=datatype).to(device)
        U0 = torch.eye(pa,dtype=datatype).to(device)
        ms,ma = int(math.ceil(nDecs/2.)), int(math.floor(nDecs/2.))        
        Zsa = torch.zeros(nChsTotal,nrows*ncols*nSamples,dtype=datatype)  
        Zsa = Zsa.to(device)      
        Ys = V[:,:,:,:ms].view(-1,ms).T
        Zsa[:ps,:] = W0[:,:ms] @ Ys
        if ma > 0:
            Ya = V[:,:,:,ms:].view(-1,ma).T
            Zsa[ps:,:] = U0[:,:ma] @ Ya
        Z = Zsa.T.view(nSamples,nrows,ncols,nChsTotal)
        # Horizontal atom extention
        Z = block_butterfly(Z,nchs)
        Z = block_shift(Z,nchs,0,[0,0,1,0]) # target=diff, shift=right
        Z = block_butterfly(Z,nchs)/2.
        Uh1 = -torch.eye(pa,dtype=datatype).to(device)
        Z = intermediate_rotation(Z,nchs,Uh1)
        Z = block_butterfly(Z,nchs)
        Z = block_shift(Z,nchs,1,[0,0,-1,0]) # target=sum, shift=left
        Z = block_butterfly(Z,nchs)/2.
        Uh2 = -torch.eye(pa,dtype=datatype).to(device)
        Z = intermediate_rotation(Z,nchs,Uh2)
        # Vertical atom extention
        Z = block_butterfly(Z,nchs)
        Z = block_shift(Z,nchs,0,[0,1,0,0]) # target=diff, shift=down
        Z = block_butterfly(Z,nchs)/2.
        Uv1 = -torch.eye(pa,dtype=datatype).to(device)
        Z = intermediate_rotation(Z,nchs,Uv1)
        Z = block_butterfly(Z,nchs)
        Z = block_shift(Z,nchs,1,[0,-1,0,0]) # target=sum, shift=up
        Z = block_butterfly(Z,nchs)/2.
        Uv2 = -torch.eye(pa,dtype=datatype).to(device)
        Z = intermediate_rotation(Z,nchs,Uv2)
        expctdZ = Z

        # Instantiation of target class
        network = NsoltAnalysis2dNetwork(
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
        nComponents = 1
        nDecs = stride[0]*stride[1] #math.prod(stride)
        nChsTotal = sum(nchs)
        # Source (nSamples x nComponents x (Stride[0]xnRows) x (Stride[1]xnCols))
        X = torch.rand(nSamples,nComponents,height,width,dtype=datatype,device=device,requires_grad=True)
        
        # Expected values
        nrows = int(math.ceil(height/stride[Direction.VERTICAL])) #.astype(int)
        ncols = int(math.ceil(width/stride[Direction.HORIZONTAL])) #.astype(int)
        # Block DCT (nSamples x nComponents x nrows x ncols) x decV x decH
        arrayshape = stride.copy()
        arrayshape.insert(0,-1)
        Y = dct_2d(X.view(arrayshape))
        # Rearrange the DCT Coefs. (nSamples x nComponents x nrows x ncols) x (decV x decH)
        A = permuteDctCoefs_(Y)
        V = A.view(nSamples,nrows,ncols,nDecs)
        # nSamples x nRows x nCols x nChs
        ps, pa = nchs
        # Initial rotation
        W0 = torch.eye(ps,dtype=datatype).to(device)
        U0 = torch.eye(pa,dtype=datatype).to(device)
        ms,ma = int(math.ceil(nDecs/2.)), int(math.floor(nDecs/2.))        
        Zsa = torch.zeros(nChsTotal,nrows*ncols*nSamples,dtype=datatype)    
        Zsa = Zsa.to(device)    
        Ys = V[:,:,:,:ms].view(-1,ms).T
        Zsa[:ps,:] = W0[:,:ms] @ Ys
        if ma > 0:
            Ya = V[:,:,:,ms:].view(-1,ma).T
            Zsa[ps:,:] = U0[:,:ma] @ Ya
        Z = Zsa.T.view(nSamples,nrows,ncols,nChsTotal)
        # Vertical atom extention
        Z = block_butterfly(Z,nchs)
        Z = block_shift(Z,nchs,0,[0,1,0,0]) # target=diff, shift=down
        Z = block_butterfly(Z,nchs)/2.
        Uv1 = -torch.eye(pa,dtype=datatype).to(device)
        Z = intermediate_rotation(Z,nchs,Uv1)
        Z = block_butterfly(Z,nchs)
        Z = block_shift(Z,nchs,1,[0,-1,0,0]) # target=sum, shift=up
        Z = block_butterfly(Z,nchs)/2.
        Uv2 = -torch.eye(pa,dtype=datatype).to(device)
        Z = intermediate_rotation(Z,nchs,Uv2)
        expctdZ = Z

        # Instantiation of target class
        network = NsoltAnalysis2dNetwork(
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
        nComponents = 1
        nDecs = stride[0]*stride[1] #math.prod(stride)
        nChsTotal = sum(nchs)
        # Source (nSamples x nComponents x (Stride[0]xnRows) x (Stride[1]xnCols))
        X = torch.rand(nSamples,nComponents,height,width,dtype=datatype,device=device,requires_grad=True)
        
        # Expected values
        nrows = int(math.ceil(height/stride[Direction.VERTICAL])) #.astype(int)
        ncols = int(math.ceil(width/stride[Direction.HORIZONTAL])) #.astype(int)
        # Block DCT (nSamples x nComponents x nrows x ncols) x decV x decH
        arrayshape = stride.copy()
        arrayshape.insert(0,-1)
        Y = dct_2d(X.view(arrayshape))
        # Rearrange the DCT Coefs. (nSamples x nComponents x nrows x ncols) x (decV x decH)
        A = permuteDctCoefs_(Y)
        V = A.view(nSamples,nrows,ncols,nDecs)
        # nSamples x nRows x nCols x nChs
        ps, pa = nchs
        # Initial rotation
        W0 = torch.eye(ps,dtype=datatype).to(device)
        U0 = torch.eye(pa,dtype=datatype).to(device)
        ms,ma = int(math.ceil(nDecs/2.)), int(math.floor(nDecs/2.))        
        Zsa = torch.zeros(nChsTotal,nrows*ncols*nSamples,dtype=datatype)        
        Zsa = Zsa.to(device)
        Ys = V[:,:,:,:ms].view(-1,ms).T
        Zsa[:ps,:] = W0[:,:ms] @ Ys
        if ma > 0:
            Ya = V[:,:,:,ms:].view(-1,ma).T
            Zsa[ps:,:] = U0[:,:ma] @ Ya
        Z = Zsa.T.view(nSamples,nrows,ncols,nChsTotal)
        # Horizontal atom extention
        Z = block_butterfly(Z,nchs)
        Z = block_shift(Z,nchs,0,[0,0,1,0]) # target=diff, shift=right
        Z = block_butterfly(Z,nchs)/2.
        Uh1 = -torch.eye(pa,dtype=datatype).to(device)
        Z = intermediate_rotation(Z,nchs,Uh1)
        Z = block_butterfly(Z,nchs)
        Z = block_shift(Z,nchs,1,[0,0,-1,0]) # target=sum, shift=left
        Z = block_butterfly(Z,nchs)/2.
        Uh2 = -torch.eye(pa,dtype=datatype).to(device)
        Z = intermediate_rotation(Z,nchs,Uh2)
        expctdZ = Z

        # Instantiation of target class
        network = NsoltAnalysis2dNetwork(
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
        nComponents = 1
        nDecs = stride[0]*stride[1] #math.prod(stride)
        nChsTotal = sum(nchs)
        # Source (nSamples x nComponents x (Stride[0]xnRows) x (Stride[1]xnCols))
        X = torch.rand(nSamples,nComponents,height,width,dtype=datatype,device=device,requires_grad=True)
        
        # Expected values
        nrows = int(math.ceil(height/stride[Direction.VERTICAL])) #.astype(int)
        ncols = int(math.ceil(width/stride[Direction.HORIZONTAL])) #.astype(int)
        # Block DCT (nSamples x nComponents x nrows x ncols) x decV x decH
        arrayshape = stride.copy()
        arrayshape.insert(0,-1)
        Y = dct_2d(X.view(arrayshape))
        # Rearrange the DCT Coefs. (nSamples x nComponents x nrows x ncols) x (decV x decH)
        A = permuteDctCoefs_(Y)
        V = A.view(nSamples,nrows,ncols,nDecs)
        # nSamples x nRows x nCols x nChs
        ps, pa = nchs
        # Initial rotation
        W0 = torch.eye(ps,dtype=datatype).to(device)
        U0 = torch.eye(pa,dtype=datatype).to(device)
        ms,ma = int(math.ceil(nDecs/2.)), int(math.floor(nDecs/2.))        
        Zsa = torch.zeros(nChsTotal,nrows*ncols*nSamples,dtype=datatype)  
        Zsa = Zsa.to(device)      
        Ys = V[:,:,:,:ms].view(-1,ms).T
        Zsa[:ps,:] = W0[:,:ms] @ Ys
        if ma > 0:
            Ya = V[:,:,:,ms:].view(-1,ma).T
            Zsa[ps:,:] = U0[:,:ma] @ Ya
        Z = Zsa.T.view(nSamples,nrows,ncols,nChsTotal)
        # Horizontal atom extention
        for ordH in range(int(ppOrd[Direction.HORIZONTAL]/2)):
            Z = block_butterfly(Z,nchs)
            Z = block_shift(Z,nchs,0,[0,0,1,0]) # target=diff, shift=right
            Z = block_butterfly(Z,nchs)/2.
            Uh1 = -torch.eye(pa,dtype=datatype).to(device)
            Z = intermediate_rotation(Z,nchs,Uh1)
            Z = block_butterfly(Z,nchs)
            Z = block_shift(Z,nchs,1,[0,0,-1,0]) # target=sum, shift=left
            Z = block_butterfly(Z,nchs)/2.
            Uh2 = -torch.eye(pa,dtype=datatype).to(device)
            Z = intermediate_rotation(Z,nchs,Uh2)
        # Vertical atom extention
        for ordV in range(int(ppOrd[Direction.VERTICAL]/2)):
            Z = block_butterfly(Z,nchs)
            Z = block_shift(Z,nchs,0,[0,1,0,0]) # target=diff, shift=down
            Z = block_butterfly(Z,nchs)/2.
            Uv1 = -torch.eye(pa,dtype=datatype).to(device)
            Z = intermediate_rotation(Z,nchs,Uv1)
            Z = block_butterfly(Z,nchs)
            Z = block_shift(Z,nchs,1,[0,-1,0,0]) # target=sum, shift=up
            Z = block_butterfly(Z,nchs)/2.
            Uv2 = -torch.eye(pa,dtype=datatype).to(device)
            Z = intermediate_rotation(Z,nchs,Uv2)
        expctdZ = Z

        # Instantiation of target class
        network = NsoltAnalysis2dNetwork(
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
    def testForwardGrayScaleOverlappingWithInitialization(self,
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
        nComponents = 1
        nDecs = stride[0]*stride[1] #math.prod(stride)
        nChsTotal = sum(nchs)
        # Source (nSamples x nComponents x (Stride[0]xnRows) x (Stride[1]xnCols))
        X = torch.rand(nSamples,nComponents,height,width,dtype=datatype,device=device,requires_grad=True)
        
        # Expected values
        nrows = int(math.ceil(height/stride[Direction.VERTICAL])) #.astype(int)
        ncols = int(math.ceil(width/stride[Direction.HORIZONTAL])) #.astype(int)
        # Block DCT (nSamples x nComponents x nrows x ncols) x decV x decH
        arrayshape = stride.copy()
        arrayshape.insert(0,-1)
        Y = dct_2d(X.view(arrayshape))
        # Rearrange the DCT Coefs. (nSamples x nComponents x nrows x ncols) x (decV x decH)
        A = permuteDctCoefs_(Y)
        V = A.view(nSamples,nrows,ncols,nDecs)
        # nSamples x nRows x nCols x nChs
        ps, pa = nchs
        # Initial rotation
        angles = angle0*torch.ones(int((nChsTotal-2)*nChsTotal/4)).to(device) #,dtype=datatype)
        nAngsW = int(len(angles)/2)
        angsW,angsU = angles[:nAngsW],angles[nAngsW:]
        W0,U0 = gen(angsW),gen(angsU)        
        ms,ma = int(math.ceil(nDecs/2.)), int(math.floor(nDecs/2.))        
        Zsa = torch.zeros(nChsTotal,nrows*ncols*nSamples,dtype=datatype)
        Zsa = Zsa.to(device)        
        Ys = V[:,:,:,:ms].view(-1,ms).T
        Zsa[:ps,:] = W0[:,:ms] @ Ys
        if ma > 0:
            Ya = V[:,:,:,ms:].view(-1,ma).T
            Zsa[ps:,:] = U0[:,:ma] @ Ya
        Z = Zsa.T.view(nSamples,nrows,ncols,nChsTotal)
        # Horizontal atom extention
        for ordH in range(int(ppOrd[Direction.HORIZONTAL]/2)):
            Z = block_butterfly(Z,nchs)
            Z = block_shift(Z,nchs,0,[0,0,1,0]) # target=diff, shift=right
            Z = block_butterfly(Z,nchs)/2.
            Uh1 = -gen(angsU)
            Z = intermediate_rotation(Z,nchs,Uh1)
            Z = block_butterfly(Z,nchs)
            Z = block_shift(Z,nchs,1,[0,0,-1,0]) # target=sum, shift=left
            Z = block_butterfly(Z,nchs)/2.
            Uh2 = -gen(angsU)
            Z = intermediate_rotation(Z,nchs,Uh2)
        # Vertical atom extention
        for ordV in range(int(ppOrd[Direction.VERTICAL]/2)):
            Z = block_butterfly(Z,nchs)
            Z = block_shift(Z,nchs,0,[0,1,0,0]) # target=diff, shift=down
            Z = block_butterfly(Z,nchs)/2.
            Uv1 = -gen(angsU)
            Z = intermediate_rotation(Z,nchs,Uv1)
            Z = block_butterfly(Z,nchs)
            Z = block_shift(Z,nchs,1,[0,-1,0,0]) # target=sum, shift=up
            Z = block_butterfly(Z,nchs)/2.
            Uv2 = -gen(angsU)
            Z = intermediate_rotation(Z,nchs,Uv2)
        expctdZ = Z

        # Instantiation of target class
        network = NsoltAnalysis2dNetwork(
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
    def testForwardGrayScaleOverlappingWithNoDcLeakage(self,
            nchs, stride, ppord, datatype):
        rtol,atol = 1e-3,1e-6
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")                
        gen = OrthonormalMatrixGenerationSystem(dtype=datatype)

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
        nComponents = 1
        nDecs = stride[0]*stride[1] #math.prod(stride)
        nChsTotal = sum(nchs)
        # Source (nSamples x nComponents x (Stride[0]xnRows) x (Stride[1]xnCols))
        X = torch.ones(nSamples,nComponents,height,width,dtype=datatype,device=device,requires_grad=True)
        
        # Expected values
        nrows = int(math.ceil(height/stride[Direction.VERTICAL])) #.astype(int)
        ncols = int(math.ceil(width/stride[Direction.HORIZONTAL])) #.astype(int)
        # nSamples x nRows x nCols x nChs
        expctdZ = torch.cat(
                    [math.sqrt(nDecs)*torch.ones(nSamples,nrows,ncols,1,dtype=datatype,device=device),
                    torch.zeros(nSamples,nrows,ncols,nChsTotal-1,dtype=datatype,device=device)],
                    dim=3)

        # Instantiation of target class
        network = NsoltAnalysis2dNetwork(
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
        nComponents = 1
        nDecs = stride[0]*stride[1] #math.prod(stride)
        nChsTotal = sum(nchs)

        # Source (nSamples x nComponents x ((Stride[0]**nlevels) x nRows) x ((Stride[1]**nlevels) x nCols))
        X = torch.randn(nSamples,nComponents,height,width,dtype=datatype,device=device,requires_grad=True)
        
        # Expected values
        nrows = int(math.ceil(height/(stride[Direction.VERTICAL]))) #.astype(int)
        ncols = int(math.ceil(width/(stride[Direction.HORIZONTAL]))) #.astype(int)
        # Block DCT (nSamples x nComponents x nrows x ncols) x decV x decH
        arrayshape = stride.copy() 
        arrayshape.insert(0,-1)
        # Multi-level decomposition
        coefs = []
        X_ = X
        for iStage in range(nlevels):
            iLevel = iStage+1
            Y = dct_2d(X_.view(arrayshape))
            # Rearrange the DCT Coefs. (nSamples x nComponents x nrows x ncols) x (decV x decH)
            A = permuteDctCoefs_(Y)
            V = A.view(nSamples,nrows,ncols,nDecs)
            # nSamples x nRows x nCols x nChs
            ps, pa = nchs
            # Initial rotation
            angles = angle0*torch.ones(int((nChsTotal-2)*nChsTotal/4)).to(device) #,dtype=datatype)
            nAngsW = int(len(angles)/2)
            angsW,angsU = angles[:nAngsW],angles[nAngsW:]
            if nVm > 0:
                angsW[:(ps-1)] = torch.zeros_like(angsW[:(ps-1)])
            W0,U0 = gen(angsW),gen(angsU)        
            ms,ma = int(math.ceil(nDecs/2.)), int(math.floor(nDecs/2.))        
            Zsa = torch.zeros(nChsTotal,nrows*ncols*nSamples,dtype=datatype)        
            Zsa = Zsa.to(device)
            Ys = V[:,:,:,:ms].view(-1,ms).T
            Zsa[:ps,:] = W0[:,:ms] @ Ys
            if ma > 0:
                Ya = V[:,:,:,ms:].view(-1,ma).T
                Zsa[ps:,:] = U0[:,:ma] @ Ya
            Z = Zsa.T.view(nSamples,nrows,ncols,nChsTotal)
            # Horizontal atom extention
            for ordH in range(int(ppOrd[Direction.HORIZONTAL]/2)):
                Z = block_butterfly(Z,nchs)
                Z = block_shift(Z,nchs,0,[0,0,1,0]) # target=diff, shift=right
                Z = block_butterfly(Z,nchs)/2.
                Uh1 = -gen(angsU)
                Z = intermediate_rotation(Z,nchs,Uh1)
                Z = block_butterfly(Z,nchs)
                Z = block_shift(Z,nchs,1,[0,0,-1,0]) # target=sum, shift=left
                Z = block_butterfly(Z,nchs)/2.
                Uh2 = -gen(angsU)
                Z = intermediate_rotation(Z,nchs,Uh2)
            # Vertical atom extention
            for ordV in range(int(ppOrd[Direction.VERTICAL]/2)):
                Z = block_butterfly(Z,nchs)
                Z = block_shift(Z,nchs,0,[0,1,0,0]) # target=diff, shift=down
                Z = block_butterfly(Z,nchs)/2.
                Uv1 = -gen(angsU)
                Z = intermediate_rotation(Z,nchs,Uv1)
                Z = block_butterfly(Z,nchs)
                Z = block_shift(Z,nchs,1,[0,-1,0,0]) # target=sum, shift=up
                Z = block_butterfly(Z,nchs)/2.
                Uv2 = -gen(angsU)
                Z = intermediate_rotation(Z,nchs,Uv2)
            # Xac
            coefs.insert(0,Z[:,:,:,1:])
            if iLevel < nlevels:
                X_ = Z[:,:,:,0].view(nSamples,nComponents,nrows,ncols)
                nrows = int(nrows/stride[Direction.VERTICAL])
                ncols = int(ncols/stride[Direction.HORIZONTAL])            
            else: # Xdc
                coefs.insert(0,Z[:,:,:,0])
        expctdZ = tuple(coefs)

        # Instantiation of target class
        network = NsoltAnalysis2dNetwork(
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
        for iStage in range(nlevels+1):
            self.assertEqual(actualZ[iStage].dtype,datatype)         
            self.assertTrue(torch.allclose(actualZ[iStage],expctdZ[iStage],rtol=rtol,atol=atol))
            self.assertFalse(actualZ[iStage].requires_grad) 


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

        # Source (nSamples x nComponents x ((Stride[0]**nlevels) x nRows) x ((Stride[1]**nlevels) x nCols))
        X = torch.randn(nSamples,nComponents,height,width,dtype=datatype,device=device,requires_grad=True)

        # Coefficients nSamples x nRows x nCols x nChsTotal
        nrows_ = nrows
        ncols_ = ncols
        dLdZ = []
        for iLevel in range(1,nlevels+1):
            if iLevel == 1:
                dLdZ.append(torch.randn(nSamples,nrows_,ncols_,dtype=datatype,device=device)) 
            dLdZ.append(torch.randn(nSamples,nrows_,ncols_,nChsTotal-1,dtype=datatype,device=device))     
            nrows_ *= stride[Direction.VERTICAL]
            ncols_ *= stride[Direction.HORIZONTAL]
        dLdZ = tuple(dLdZ)

        # Instantiation of target class
        network = NsoltAnalysis2dNetwork(
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
        for iCh in range(len(Z)):
            Z[iCh].backward(dLdZ[iCh],retain_graph=True)
        actualdLdX = X.grad

        # Evaluation
        self.assertEqual(actualdLdX.dtype,datatype)
        self.assertTrue(torch.allclose(actualdLdX,expctddLdX,rtol=rtol,atol=atol))
        for iCh in range(len(Z)):
            self.assertTrue(Z[iCh].requires_grad)

"""
                
        % Test
        function testStepDec11Ch4Ord00Level2PeriodicExtVm0(testCase)
            
            dec = 1;
            nChs = [ 2 2 ];
            ch = sum(nChs);
            ord = 0;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', ch,...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...,...
                    'conv','circ').',dec).',dec);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...,...
                    'conv','circ').',dec).',dec);
            end            
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv1{2};
            coefs{6} = coefsExpctdLv1{3};
            coefs{7} = coefsExpctdLv1{4};            
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,2);
            sIdx = 1; 
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',nChs(1),...
                'NumberOfAntisymmetricChannels',nChs(2),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,...
                sprintf('%g',diff));  
            
        end
                       
        % Test
        function testStepDec11Ch4Ord00Level2PeriodicExtVm1(testCase)
            
            dec = 1;
            nChs = [ 2 2 ];
            ch = sum(nChs);
            ord = 0;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', ch,...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...,...
                    'conv','circ').',dec).',dec);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...,...
                    'conv','circ').',dec).',dec);
            end            
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv1{2};
            coefs{6} = coefsExpctdLv1{3};
            coefs{7} = coefsExpctdLv1{4};            
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,2);
            sIdx = 1; 
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',nChs(1),...
                'NumberOfAntisymmetricChannels',nChs(2),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,...
                sprintf('%g',diff));  
            
        end
      
        % Test
        function testStepDec22Ch22Ord00Level1PeriodicExt(testCase)
            
            dec = 2;
            nChs = [ 2 2 ];
            ch = sum(nChs);
            ord = 0;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            
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
           nSubCoefs = numel(srcImg)/(dec*dec);
           coefsExpctd = zeros(1,ch*nSubCoefs);
           for iSubband = 1:ch
                subCoef = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...,...
                    'conv','circ').',dec).',dec);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./[dec dec],ch,1);

            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',nChs(1),...
                'NumberOfAntisymmetricChannels',nChs(2),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,sprintf('%g',diff));
            
        end

        % Test
        function testStepDec22Ch4Ord00Level2eriodicExt(testCase)
            
            dec = 2;
            nChs = [ 2 2 ];
            ch = sum(nChs);
            ord = 0;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...,...
                    'conv','circ').',dec).',dec);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...,...
                    'conv','circ').',dec).',dec);
            end            
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv1{2};
            coefs{6} = coefsExpctdLv1{3};
            coefs{7} = coefsExpctdLv1{4};            
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,2);
            sIdx = 1; 
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',nChs(1),...
                'NumberOfAntisymmetricChannels',nChs(2),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,...
                sprintf('%g',diff));  

            
        end
       
        % Test
        function testStepDec22Ch6Ord00Level1(testCase)
            
            dec = 2;
            nChs = [ 3 3 ];
            ch = sum(nChs);
            ord = 0;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/(dec*dec);
            coefsExpctd = zeros(1,ch*nSubCoefs);
            for iSubband = 1:ch
                subCoef = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...,...
                    'conv','circ').',dec).',dec);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./[dec dec],ch,1);

            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',nChs(1),...
                'NumberOfAntisymmetricChannels',nChs(2),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,...
                sprintf('%g',diff));
        end
        
        % Test
        function testIterDecompDec22Ch6Ord00Level2PeriodicExt(testCase)
            
            dec = 2;
            nChs = [ 3 3 ];
            ch = sum(nChs);
            ord = 0;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...,...
                    'conv','circ').',dec).',dec);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...,...
                    'conv','circ').',dec).',dec);
            end            
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv2{5};
            coefs{6} = coefsExpctdLv2{6};
            coefs{7} = coefsExpctdLv1{2};
            coefs{8} = coefsExpctdLv1{3};
            coefs{9} = coefsExpctdLv1{4};            
            coefs{10} = coefsExpctdLv1{5};
            coefs{11} = coefsExpctdLv1{6};
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,2);
            sIdx = 1; 
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',nChs(1),...
                'NumberOfAntisymmetricChannels',nChs(2),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,...
                sprintf('%g',diff));  
            
        end
        
       % Test
        function testStepDec22Ch8Ord00Level1PeriodicExt(testCase)
            
            dec = 2;
            nChs = [ 4 4 ];
            ch = sum(nChs);
            ord = 0;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
          % Expected values
          release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/(dec*dec);
            coefsExpctd = zeros(1,ch*nSubCoefs);
            for iSubband = 1:ch
                subCoef = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...,...
                    'conv','circ').',dec).',dec);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./[dec dec],ch,1);

            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',nChs(1),...
                'NumberOfAntisymmetricChannels',nChs(2),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec22Ch8Ord00Level2PeriodicExt(testCase)
            
            dec = 2;
            nChs = [ 4 4 ];
            ch = sum(nChs);
            ord = 0;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...,...
                    'conv','circ').',dec).',dec);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...,...
                    'conv','circ').',dec).',dec);
            end            
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv2{5};
            coefs{6} = coefsExpctdLv2{6};
            coefs{7} = coefsExpctdLv2{7};
            coefs{8} = coefsExpctdLv2{8};            
            coefs{9} = coefsExpctdLv1{2};
            coefs{10} = coefsExpctdLv1{3};
            coefs{11} = coefsExpctdLv1{4};            
            coefs{12} = coefsExpctdLv1{5};
            coefs{13} = coefsExpctdLv1{6};
            coefs{14} = coefsExpctdLv1{7};
            coefs{15} = coefsExpctdLv1{8};
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,2);
            sIdx = 1; 
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',nChs(1),...
                'NumberOfAntisymmetricChannels',nChs(2),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,...
                sprintf('%g',diff));            
            
        end
          
        % Test
        function testStepDec11Ch4Ord22Level1(testCase)
            
            dec = 1;
            nChs = [ 2 2 ];
            ch = sum(nChs);
            ord = 2;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/(dec*dec);
            coefsExpctd = zeros(1,ch*nSubCoefs);
            for iSubband = 1:ch
                subCoef = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...,...
                    'conv','circ').',dec).',dec);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./[dec dec],ch,1);

            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',nChs(1),...
                'NumberOfAntisymmetricChannels',nChs(2),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec11Ch4Ord22Level2PeriodicExt(testCase)
            
            dec = 1;
            nChs = [ 2 2 ];
            ch = sum(nChs);
            ord = 2;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...,...
                    'conv','circ').',dec).',dec);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...,...
                    'conv','circ').',dec).',dec);
            end            
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv1{2};
            coefs{6} = coefsExpctdLv1{3};
            coefs{7} = coefsExpctdLv1{4};            
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,2);
            sIdx = 1; 
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',nChs(1),...
                'NumberOfAntisymmetricChannels',nChs(2),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,...
                sprintf('%g',diff));            
        end
      
        % Test
        function testStepDec22Ch22Ord22Level1PeridicExt(testCase)
            
            dec = 2;
            chs = [ 2 2 ];
            nChs = sum(chs);
            ord = 2;
            height = 32;
            width = 64;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', chs,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/(dec*dec);
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            for iSubband = 1:nChs
                subCoef = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...,...
                    'conv','circ').',dec).',dec);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./[dec dec],nChs,1);

            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',chs(1),...
                'NumberOfAntisymmetricChannels',chs(2),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,sprintf('%g',diff));
            
        end
        
      
        % Test
        function testStepDec22Ch22Ord02Level1PeridicExt(testCase)
            
            dec = 2;
            chs = [ 2 2 ];
            nChs = sum(chs);
            ord = [ 0 2 ];
            height = 32;
            width = 64;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', chs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/(dec*dec);
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            for iSubband = 1:nChs
                subCoef = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...,...
                    'conv','circ').',dec).',dec);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./[dec dec],nChs,1);

            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',chs(1),...
                'NumberOfAntisymmetricChannels',chs(2),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,sprintf('%g',diff));
            
        end

      
        % Test
        function testStepDec22Ch22Ord20Level1PeridicExt(testCase)
            
            dec = 2;
            chs = [ 2 2 ];
            nChs = sum(chs);
            ord = [ 2 0 ];
            height = 32;
            width = 64;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', chs,...
                'PolyPhaseOrder',ord);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/(dec*dec);
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            for iSubband = 1:nChs
                subCoef = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...,...
                    'conv','circ').',dec).',dec);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./[dec dec],nChs,1);

            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',chs(1),...
                'NumberOfAntisymmetricChannels',chs(2),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,sprintf('%g',diff));
            
        end
        

        % Test
        function testStepDec22Ch4Ord22Level2eriodicExt(testCase)
            
            dec = 2;
            nChs = [ 2 2 ];
            ch = sum(nChs);
            ord = 2;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...,...
                    'conv','circ').',dec).',dec);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...,...
                    'conv','circ').',dec).',dec);
            end            
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv1{2};
            coefs{6} = coefsExpctdLv1{3};
            coefs{7} = coefsExpctdLv1{4};            
            nSubbands = length(coefs);            
            scalesExpctd = zeros(nSubbands,2);
            sIdx = 1; 
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',nChs(1),...
                'NumberOfAntisymmetricChannels',nChs(2),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,...
                sprintf('%g',diff));
            
            
        end
       
        % Test
        function testStepDec22Ch6Ord22Level1(testCase)
            
            dec = 2;
            nChs = [ 3 3 ];
            ch = sum(nChs);
            ord = 2;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/(dec*dec);
            coefsExpctd = zeros(1,ch*nSubCoefs);
            for iSubband = 1:ch
                subCoef = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...,...
                    'conv','circ').',dec).',dec);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./[dec dec],ch,1);

            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',nChs(1),...
                'NumberOfAntisymmetricChannels',nChs(2),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,...
                sprintf('%g',diff));
            
        end
       
        % Test
        function testStepDec22Ch6Ord22Level2PeriodicExt(testCase)
            
            dec = 2;
            nChs = [ 3 3 ];            
            ch = sum(nChs);
            ord = 2;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...,...
                    'conv','circ').',dec).',dec);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...,...
                    'conv','circ').',dec).',dec);
            end            
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv2{5};
            coefs{6} = coefsExpctdLv2{6};
            coefs{7} = coefsExpctdLv1{2};
            coefs{8} = coefsExpctdLv1{3};
            coefs{9} = coefsExpctdLv1{4};            
            coefs{10} = coefsExpctdLv1{5};
            coefs{11} = coefsExpctdLv1{6};
            nSubbands = length(coefs);            
            scalesExpctd = zeros(nSubbands,2);
            sIdx = 1; 
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',nChs(1),...
                'NumberOfAntisymmetricChannels',nChs(2),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,...
                sprintf('%g',diff));
            
        end

       % Test
        function testStepDec22Ch8Ord22Level1(testCase)
            
            dec = 2;
            nChs = [ 4 4 ];
            ch = sum(nChs);
            ord = 2;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/(dec*dec);
            coefsExpctd = zeros(1,ch*nSubCoefs);
            for iSubband = 1:ch
                subCoef = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...,...
                    'conv','circ').',dec).',dec);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./[dec dec],ch,1);

            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',nChs(1),...
                'NumberOfAntisymmetricChannels',nChs(2),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,...
                sprintf('%g',diff));
            
            
        end
       
        % Test
        function testStepDec22Ch8Ord22Level2PeriodicExt(testCase)
            
            dec = 2;
            nChs = [ 4 4 ];
            ch = sum(nChs);
            ord = 2;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', nChs,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...,...
                    'conv','circ').',dec).',dec);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...,...
                    'conv','circ').',dec).',dec);
            end            
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv2{5};
            coefs{6} = coefsExpctdLv2{6};
            coefs{7} = coefsExpctdLv2{7};
            coefs{8} = coefsExpctdLv2{8};            
            coefs{9} = coefsExpctdLv1{2};
            coefs{10} = coefsExpctdLv1{3};
            coefs{11} = coefsExpctdLv1{4};            
            coefs{12} = coefsExpctdLv1{5};
            coefs{13} = coefsExpctdLv1{6};
            coefs{14} = coefsExpctdLv1{7};
            coefs{15} = coefsExpctdLv1{8};
            nSubbands = length(coefs);                        
            scalesExpctd = zeros(nSubbands,2);
            sIdx = 1; 
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
                        
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',nChs(1),...
                'NumberOfAntisymmetricChannels',nChs(2),...
                'BoundaryOperation','Circular');

            % Actual values
            [coefsActual, scalesActual]= step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,...
                sprintf('%g',diff));            
        end

        % Level 3, dec 22 ch 44 order 44 
        function testStepDec22Ch4plus4Ord44Level3PeriodicExt(testCase)
            
            dec = 2;
            nChs =  [ 4 4 ];
            ch = sum(nChs);
            ord = 4;
            height = 64;
            width = 64;
            srcImg = rand(height,width);
            nLevels = 3;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels', ch,...
                'PolyPhaseOrder',[ord ord]);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...,...
                    'conv','circ').',dec).',dec);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...,...
                    'conv','circ').',dec).',dec);
            end      
            coefsExpctdLv3 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv3{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv2{1},...
                    step(lppufb,[],[],iSubband),...,...
                    'conv','circ').',dec).',dec);
            end
            coefs{1} = coefsExpctdLv3{1};
            coefs{2} = coefsExpctdLv3{2};
            coefs{3} = coefsExpctdLv3{3};
            coefs{4} = coefsExpctdLv3{4};
            coefs{5} = coefsExpctdLv3{5};
            coefs{6} = coefsExpctdLv3{6};
            coefs{7} = coefsExpctdLv3{7};
            coefs{8} = coefsExpctdLv3{8};            
            coefs{9} = coefsExpctdLv2{2};
            coefs{10} = coefsExpctdLv2{3};
            coefs{11} = coefsExpctdLv2{4};
            coefs{12} = coefsExpctdLv2{5};
            coefs{13} = coefsExpctdLv2{6};
            coefs{14} = coefsExpctdLv2{7};
            coefs{15} = coefsExpctdLv2{8};            
            coefs{16} = coefsExpctdLv1{2};
            coefs{17} = coefsExpctdLv1{3};
            coefs{18} = coefsExpctdLv1{4};            
            coefs{19} = coefsExpctdLv1{5};
            coefs{20} = coefsExpctdLv1{6};
            coefs{21} = coefsExpctdLv1{7};
            coefs{22} = coefsExpctdLv1{8};
            nSubbands = length(coefs);            
            scalesExpctd = zeros(nSubbands,2);
            sIdx = 1; 
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',nChs(1),...
                'NumberOfAntisymmetricChannels',nChs(2),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,...
                sprintf('%g',diff));
            
        end        
        
        % Level 3, dec 22 ch 8  order 44 
        function testSetLpPuFb2dDec22Ch44Ord44(testCase)
            
            dec = [ 2 2 ];
            ch =  [ 4 4 ];
            ord = [ 4 4 ];
            height = 64;
            width = 64;
            nLevels = 1;
            srcImg = rand(height,width);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'BoundaryOperation','Termination');
            coefsPre = step(testCase.analyzer,srcImg);
            %
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs); 
            coefsPst1 = step(testCase.analyzer,srcImg);
            
            % Evaluation
            diff = norm(coefsPst1(:)-coefsPre(:));
            testCase.verifyEqual(diff,0,'AbsTol',1e-15,...
                sprintf('%g',diff));
            
            % Reinstatiation 
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'BoundaryOperation','Termination');
            coefsPst2 = step(testCase.analyzer,srcImg);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan
            diff = norm(coefsPst2(:)-coefsPre(:));
            testCase.verifyThat(diff,IsGreaterThan(0),sprintf('%g',diff));            
        end
        
       % Level 1, dec 22 ch 8  order 44 
        function testSetLpPuFb2dDec44Ch88Ord22(testCase)
            
            dec = [ 4 4 ];
            ch =  [ 8 8 ];
            ord = [ 2 2 ];
            height = 64;
            width = 64;
            nLevels = 1;
            srcImg = rand(height,width);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'BoundaryOperation','Termination');
            coefsPre = step(testCase.analyzer,srcImg);
            %
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs); 
            coefsPst1 = step(testCase.analyzer,srcImg);
            
            % Evaluation
            diff = norm(coefsPst1(:)-coefsPre(:));
            testCase.verifyEqual(diff,0,'AbsTol',1e-15,...
                sprintf('%g',diff));
            
            % Reinstatiation 
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'BoundaryOperation','Termination');
            coefsPst2 = step(testCase.analyzer,srcImg);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan
            diff = norm(coefsPst2(:)-coefsPre(:));
            testCase.verifyThat(diff,IsGreaterThan(0),sprintf('%g',diff));            
        end
        
        % Level 3, dec 22 ch 8  order 44
        function testIsCloneFalse(testCase)
            
            dec = [ 2 2 ];
            ch =  [ 4 4 ];
            ord = [ 4 4 ];
            height = 64;
            width = 64;
            nLevels = 1;
            srcImg = rand(height,width);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord,...
                'OutputMode','ParameterMatrixSet');
            
            % Instantiation of target class
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'BoundaryOperation','Termination',...
                'IsCloneLpPuFb',true);
            
            % Pre
            coefsPre1 = step(testCase.analyzer,srcImg);
            
            % Pst
            angs = randn(size(get(lppufb,'Angles')));
            set(lppufb,'Angles',angs);
            coefsPst1 = step(testCase.analyzer,srcImg);
            
            % Evaluation
            diff = norm(coefsPst1(:)-coefsPre1(:));
            testCase.verifyEqual(diff,0,'AbsTol',1e-15,sprintf('%g',diff));
            
            % Instantiation of target class
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'BoundaryOperation','Termination',...
                'IsCloneLpPuFb',false);
            
            % Pre
            coefsPre1 = step(testCase.analyzer,srcImg);
            
            % Pst
            angs = randn(size(get(lppufb,'Angles')));
            set(lppufb,'Angles',angs);
            coefsPst1 = step(testCase.analyzer,srcImg);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan
            diff = norm(coefsPst1(:)-coefsPre1(:));
            testCase.verifyThat(diff,IsGreaterThan(0),sprintf('%g',diff));            
           
        end
        
        % Test
        function testClone(testCase)
            
            dec = [ 2 2 ];
            ch =  [ 4 4 ];
            ord = [ 4 4 ];
            height = 64;
            width  = 64;
            nLevels = 1;
            srcImg = rand(height,width);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord,...
                'OutputMode','ParameterMatrixSet');
            
            % Instantiation of target class
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'BoundaryOperation','Termination');
            %s = matlab.System.saveObject(testCase.analyzer);
            
            % Clone
            cloneAnalyzer = clone(testCase.analyzer);
            
            % Evaluation
            testCase.verifyEqual(cloneAnalyzer,testCase.analyzer);
            testCase.verifyFalse(cloneAnalyzer == testCase.analyzer);
            prpOrg = get(testCase.analyzer,'LpPuFb2d');
            prpCln = get(cloneAnalyzer,'LpPuFb2d');
            testCase.verifyEqual(prpCln,prpOrg);
            testCase.verifyFalse(prpCln == prpOrg);
            %
            [coefExpctd,scaleExpctd] = step(testCase.analyzer,srcImg);
            [coefActual,scaleActual] = step(cloneAnalyzer,srcImg);
            testCase.assertEqual(coefActual,coefExpctd);
            testCase.assertEqual(scaleActual,scaleExpctd);
            
        end     
        
       % Test
        function testDefaultConstructionTypeII(testCase)
            
            % Expected values
            import saivdr.dictionary.nsoltx.*
            lppufbExpctd = OvsdLpPuFb2dTypeIIVm1System(...
                'OutputMode','ParameterMatrixSet');
            
            % Instantiation
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufbExpctd);
            
            % Actual value
            lppufbActual = get(testCase.analyzer,'LpPuFb2d');
            
            % Evaluation
            testCase.verifyEqual(lppufbActual,lppufbExpctd);
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
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'NumberOfSymmetricChannels',nChs(ChannelGroup.UPPER),...
                'NumberOfAntisymmetricChannels',nChs(ChannelGroup.LOWER));
            
            % Actual value
            lppufbActual = get(testCase.analyzer,'LpPuFb2d');
            
            % Evaluation
            testCase.verifyEqual(lppufbActual,lppufbExpctd);
        end
                 
        % Test
        function testStepDec11Ch32Ord00Level1PeriodicExtVm0(testCase)
            
            dec = 1;
            decch = [ dec dec 3 2 ];
            nChs = sum(decch(3:4));
            ord = 0;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/(decch(1)*decch(2));
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            for iSubband = 1:nChs
                subCoef = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',decch(2)).',decch(1));
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./decch(1:2),nChs,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec11Ch32Ord00Level1PeriodicExtVm1(testCase)
            
            dec = 1;
            decch = [ dec dec 3 2 ];
            nChs = sum(decch(3:4));
            ord = 0;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb);
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/(decch(1)*decch(2));
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            for iSubband = 1:nChs
                subCoef = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',decch(2)).',decch(1));
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./decch(1:2),nChs,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec11Ch32Ord00Level2PeriodicExtVm0(testCase)
            
            dec = 1;
            decch = [ dec dec 3 2 ];
            nChs = sum(decch(3:4));
            ord = 0;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels', decch(3:end),...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefsExpctdLv2 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv2{5};
            coefs{6} = coefsExpctdLv1{2};
            coefs{7} = coefsExpctdLv1{3};
            coefs{8} = coefsExpctdLv1{4};
            coefs{9} = coefsExpctdLv1{5};
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec11Ch32Ord00Level2PeriodicExtVm1(testCase)
            
            dec = 1;
            decch = [ dec dec 3 2 ];
            nChs = sum(decch(3:4));
            ord = 0;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels', decch(3:end),...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',1);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefsExpctdLv2 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv2{5};
            coefs{6} = coefsExpctdLv1{2};
            coefs{7} = coefsExpctdLv1{3};
            coefs{8} = coefsExpctdLv1{4};
            coefs{9} = coefsExpctdLv1{5};
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec22Ch32Ord00Level1(testCase)
            
            dec = 2;
            decch = [ dec dec 3 2 ];
            nChs = sum(decch(3:4));
            ord = 0;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/(decch(1)*decch(2));
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            for iSubband = 1:nChs
                subCoef = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',decch(2)).',decch(1));
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./decch(1:2),nChs,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec44Ch898Ord22Level1(testCase)
            
            dec = 4;
            decch = [ dec dec 9 8 ];
            nChs = sum(decch(3:4));
            ord = 2;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/(decch(1)*decch(2));
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            for iSubband = 1:nChs
                subCoef = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',decch(2),1).',decch(1),1);
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./decch(1:2),nChs,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec22Ch32Ord00Level2eriodicExt(testCase)
            
            dec = 2;
            decch = [ dec dec 3 2];
            ch = sum(decch(3:4));
            ord = 0;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv2{5};
            coefs{6} = coefsExpctdLv1{2};
            coefs{7} = coefsExpctdLv1{3};
            coefs{8} = coefsExpctdLv1{4};
            coefs{9} = coefsExpctdLv1{5};
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec22Ch43Ord00Level1(testCase)
            
            dec = 2;
            decch = [ dec dec 4 3 ];
            nChs = sum(decch(3:4));
            ord = 0;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/(decch(1)*decch(2));
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            for iSubband = 1:nChs
                subCoef = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',decch(2)).',decch(1));
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./decch(1:2),nChs,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec22Ch43Ord00Level2PeriodicExt(testCase)
            
            dec = 2;
            decch = [ dec dec 4 3 ];
            ch = sum(decch(3:4));
            ord = 0;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv2{5};
            coefs{6} = coefsExpctdLv2{6};
            coefs{7} = coefsExpctdLv2{7};
            coefs{8} = coefsExpctdLv1{2};
            coefs{9} = coefsExpctdLv1{3};
            coefs{10} = coefsExpctdLv1{4};
            coefs{11} = coefsExpctdLv1{5};
            coefs{12} = coefsExpctdLv1{6};
            coefs{13} = coefsExpctdLv1{7};
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,...
                sprintf('%g',diff));
            
            
        end
        
        % Test
        function testStepDec22Ch54Ord00Level1(testCase)
            
            dec = 2;
            decch = [ dec dec 5 4 ];
            nChs= sum(decch(3:4));
            ord = 0;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/(decch(1)*decch(2));
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            for iSubband = 1:nChs
                subCoef = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',decch(2)).',decch(1));
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./decch(1:2),nChs,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,sprintf('%g',diff));
        end
        
        % Test
        function testStepDec22Ch54Ord00Level2PeriodicExt(testCase)
            
            dec = 2;
            decch = [ dec dec 5 4 ];
            ch = sum(decch(3:4));
            ord = 0;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv2{5};
            coefs{6} = coefsExpctdLv2{6};
            coefs{7} = coefsExpctdLv2{7};
            coefs{8} = coefsExpctdLv2{8};
            coefs{9} = coefsExpctdLv2{9};
            coefs{10} = coefsExpctdLv1{2};
            coefs{11} = coefsExpctdLv1{3};
            coefs{12} = coefsExpctdLv1{4};
            coefs{13} = coefsExpctdLv1{5};
            coefs{14} = coefsExpctdLv1{6};
            coefs{15} = coefsExpctdLv1{7};
            coefs{16} = coefsExpctdLv1{8};
            coefs{17} = coefsExpctdLv1{9};
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,...
                sprintf('%g',diff));
            
            
        end
        
        % Test
        function testStepDec11Ch32Ord22Level1(testCase)
            
            dec = 1;
            decch = [ dec dec 3 2 ];
            nChs = sum(decch(3:4));
            ord = 2;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/(decch(1)*decch(2));
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            for iSubband = 1:nChs
                subCoef = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',decch(2)).',decch(1));
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./decch(1:2),nChs,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec11Ch32Ord22Level2PeriodicExt(testCase)
            
            dec = 1;
            decch = [ dec dec 3 2 ];
            ch = sum(decch(3:4));
            ord = 2;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv2{5};
            coefs{6} = coefsExpctdLv1{2};
            coefs{7} = coefsExpctdLv1{3};
            coefs{8} = coefsExpctdLv1{4};
            coefs{9} = coefsExpctdLv1{5};
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,...
                sprintf('%g',diff));
            
            
        end
        
        % Test
        function testStepDec22Ch32Ord22Level1(testCase)
            
            dec = 2;
            decch = [ dec dec 3 2 ];
            nChs = sum(decch(3:4));
            ord = 2;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/(decch(1)*decch(2));
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            for iSubband = 1:nChs
                subCoef = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',decch(2)).',decch(1));
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./decch(1:2),nChs,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec22Ch32Ord22Level2eriodicExt(testCase)
            
            dec = 2;
            decch = [ dec dec 3 2 ];
            ch = sum(decch(3:4));
            ord = 2;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv2{5};
            coefs{6} = coefsExpctdLv1{2};
            coefs{7} = coefsExpctdLv1{3};
            coefs{8} = coefsExpctdLv1{4};
            coefs{9} = coefsExpctdLv1{5};
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,...
                sprintf('%g',diff));
            
            
        end
        
        % Test
        function testStepDec22Ch43Ord22Level1(testCase)
            
            dec = 2;
            decch = [ dec dec 4 3 ];
            nChs = sum(decch(3:4));
            ord = 2;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/(decch(1)*decch(2));
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            for iSubband = 1:nChs
                subCoef = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',decch(2)).',decch(1));
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./decch(1:2),nChs,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec22Ch43Ord22Level2PeriodicExt(testCase)
            
            dec = 2;
            decch = [ dec dec 4 3 ];
            ch = sum(decch(3:4));
            ord = 2;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv2{5};
            coefs{6} = coefsExpctdLv2{6};
            coefs{7} = coefsExpctdLv2{7};
            coefs{8} = coefsExpctdLv1{2};
            coefs{9} = coefsExpctdLv1{3};
            coefs{10} = coefsExpctdLv1{4};
            coefs{11} = coefsExpctdLv1{5};
            coefs{12} = coefsExpctdLv1{6};
            coefs{13} = coefsExpctdLv1{7};
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,...
                sprintf('%g',diff));
            
            
        end
 
        % Test
        function testStepDec22Ch9Ord22Level1(testCase)
            
            dec = 2;
            decch = [ dec dec 5 4 ];
            nChs = sum(decch(3:4));
            ord = 2;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels', decch(3:end),...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/(decch(1)*decch(2));
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            for iSubband = 1:nChs
                subCoef = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',decch(2)).',decch(1));
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./decch(1:2),nChs,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,sprintf('%g',diff));
            
            
        end
        
        % Test
        function testSteppDec22Ch9Ord22Level2PeriodicExt(testCase)
            
            dec = 2;
            nChs = [ 5 4 ];
            decch = [ dec dec nChs ];
            ch = sum(nChs);
            ord = 2;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv2{5};
            coefs{6} = coefsExpctdLv2{6};
            coefs{7} = coefsExpctdLv2{7};
            coefs{8} = coefsExpctdLv2{8};
            coefs{9} = coefsExpctdLv2{9};
            coefs{10} = coefsExpctdLv1{2};
            coefs{11} = coefsExpctdLv1{3};
            coefs{12} = coefsExpctdLv1{4};
            coefs{13} = coefsExpctdLv1{5};
            coefs{14} = coefsExpctdLv1{6};
            coefs{15} = coefsExpctdLv1{7};
            coefs{16} = coefsExpctdLv1{8};
            coefs{17} = coefsExpctdLv1{9};
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,...
                sprintf('%g',diff));
            
            
        end
        
        % Level 3, dec 11 ch 54 order 88
        function testStepDec11Ch54Ord88Level3PeriodicExt(testCase)
            
            dec = 1;
            nChs = [ 5 4 ];
            decch = [ dec dec nChs ];
            ch = sum(nChs);
            ord = 8;
            height = 64;
            width = 64;
            srcImg = rand(height,width);
            nLevels = 3;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefsExpctdLv3 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv3{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv2{1},...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefs{1} = coefsExpctdLv3{1};
            coefs{2} = coefsExpctdLv3{2};
            coefs{3} = coefsExpctdLv3{3};
            coefs{4} = coefsExpctdLv3{4};
            coefs{5} = coefsExpctdLv3{5};
            coefs{6} = coefsExpctdLv3{6};
            coefs{7} = coefsExpctdLv3{7};
            coefs{8} = coefsExpctdLv3{8};
            coefs{9} = coefsExpctdLv3{9};
            coefs{10} = coefsExpctdLv2{2};
            coefs{11} = coefsExpctdLv2{3};
            coefs{12} = coefsExpctdLv2{4};
            coefs{13} = coefsExpctdLv2{5};
            coefs{14} = coefsExpctdLv2{6};
            coefs{15} = coefsExpctdLv2{7};
            coefs{16} = coefsExpctdLv2{8};
            coefs{17} = coefsExpctdLv2{9};
            coefs{18} = coefsExpctdLv1{2};
            coefs{19} = coefsExpctdLv1{3};
            coefs{20} = coefsExpctdLv1{4};
            coefs{21} = coefsExpctdLv1{5};
            coefs{22} = coefsExpctdLv1{6};
            coefs{23} = coefsExpctdLv1{7};
            coefs{24} = coefsExpctdLv1{8};
            coefs{25} = coefsExpctdLv1{9};
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,...
                sprintf('%g',diff));
            
        end
        
        % Level 3, dec 22 ch 54 order 44
        function testStepDec22Ch54Ord44Level3PeriodicExt(testCase)
            
            dec = 2;
            nChs = [ 5 4 ];
            decch = [ dec dec nChs ];
            ch = sum(nChs);
            ord = 4;
            height = 64;
            width = 64;
            srcImg = rand(height,width);
            nLevels = 3;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefsExpctdLv3 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv3{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv2{1},...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefs{1} = coefsExpctdLv3{1};
            coefs{2} = coefsExpctdLv3{2};
            coefs{3} = coefsExpctdLv3{3};
            coefs{4} = coefsExpctdLv3{4};
            coefs{5} = coefsExpctdLv3{5};
            coefs{6} = coefsExpctdLv3{6};
            coefs{7} = coefsExpctdLv3{7};
            coefs{8} = coefsExpctdLv3{8};
            coefs{9} = coefsExpctdLv3{9};
            coefs{10} = coefsExpctdLv2{2};
            coefs{11} = coefsExpctdLv2{3};
            coefs{12} = coefsExpctdLv2{4};
            coefs{13} = coefsExpctdLv2{5};
            coefs{14} = coefsExpctdLv2{6};
            coefs{15} = coefsExpctdLv2{7};
            coefs{16} = coefsExpctdLv2{8};
            coefs{17} = coefsExpctdLv2{9};
            coefs{18} = coefsExpctdLv1{2};
            coefs{19} = coefsExpctdLv1{3};
            coefs{20} = coefsExpctdLv1{4};
            coefs{21} = coefsExpctdLv1{5};
            coefs{22} = coefsExpctdLv1{6};
            coefs{23} = coefsExpctdLv1{7};
            coefs{24} = coefsExpctdLv1{8};
            coefs{25} = coefsExpctdLv1{9};
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec22Ch32Ord22Level1PeriodicExt(testCase)
            
            decch = [2 2 3 2];
            nChs = sum(decch(3:4));
            ord = 2;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/(decch(1)*decch(2));
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            for iSubband = 1:nChs
                subCoef = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',decch(2)).',decch(1));
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./decch(1:2),nChs,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,sprintf('%g',diff));
            
            
        end
        
        % Test
        function testStepDec22Ch32Ord22Level2PeriodicExt(testCase)
            
            decch = [2 2 3 2];
            nChs = sum(decch(3:4));
            ord = 2;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',decch(1)).',decch(2));
            end
            coefsExpctdLv2 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',decch(1)).',decch(2));
            end
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv2{5};
            coefs{6} = coefsExpctdLv1{2};
            coefs{7} = coefsExpctdLv1{3};
            coefs{8} = coefsExpctdLv1{4};
            coefs{9} = coefsExpctdLv1{5};
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,...
                sprintf('%g',diff));
        end
        
        % Test
        function testStepDec22Ch42Ord22Level1PeriodicExt(testCase)
            
            decch = [2 2 4 2];
            nChs = sum(decch(3:4));
            ord = 2;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/(decch(1)*decch(2));
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            for iSubband = 1:nChs
                subCoef = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',decch(2)).',decch(1));
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./decch(1:2),nChs,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec22Ch42Ord22Level2PeriodicExt(testCase)
            
            decch = [2 2 4 2];
            nChs = sum(decch(3:4));
            ord = 2;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',decch(1)).',decch(2));
            end
            coefsExpctdLv2 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',decch(1)).',decch(2));
            end
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv2{5};
            coefs{6} = coefsExpctdLv2{6};
            coefs{7} = coefsExpctdLv1{2};
            coefs{8} = coefsExpctdLv1{3};
            coefs{9} = coefsExpctdLv1{4};
            coefs{10} = coefsExpctdLv1{5};
            coefs{11} = coefsExpctdLv1{6};
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testSetLpPuFb2dDec22Ch52Ord44(testCase)
            
            dec = [ 2 2 ];
            ch =  [ 5 2 ];
            ord = [ 4 4 ];
            height = 64;
            width = 64;
            nLevels = 1;
            srcImg = rand(height,width);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'BoundaryOperation','Termination');
            coefsPre = step(testCase.analyzer,srcImg);
            %
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            coefsPst1 = step(testCase.analyzer,srcImg);
            
            % Evaluation
            diff = norm(coefsPst1(:)-coefsPre(:));
            testCase.verifyEqual(diff,0,'AbsTol',1e-15,...
                sprintf('%g',diff));
            
            % Reinstatiation
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'BoundaryOperation','Termination');
            coefsPst2 = step(testCase.analyzer,srcImg);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan
            diff = norm(coefsPst2(:)-coefsPre(:));
            testCase.verifyThat(diff,IsGreaterThan(0),sprintf('%g',diff));
        end
        
        % Test
        function testIsCloneFalseTypeII(testCase)
            
            dec = [ 2 2 ];
            ch =  [ 6 2 ];
            ord = [ 4 4 ];
            height = 64;
            width = 64;
            nLevels = 1;
            srcImg = rand(height,width);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord,...
                'OutputMode','ParameterMatrixSet');
            
            % Instantiation of target class
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'BoundaryOperation','Termination',...
                'IsCloneLpPuFb',true);
            
            % Pre
            coefsPre1 = step(testCase.analyzer,srcImg);
            
            % Pst
            angs = randn(size(get(lppufb,'Angles')));
            set(lppufb,'Angles',angs);
            coefsPst1 = step(testCase.analyzer,srcImg);
            
            % Evaluation
            diff = norm(coefsPst1(:)-coefsPre1(:));
            testCase.verifyEqual(diff,0,'AbsTol',1e-15,sprintf('%g',diff));
            
            % Instantiation of target class
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'BoundaryOperation','Termination',...
                'IsCloneLpPuFb',false);
            
            % Pre
            coefsPre1 = step(testCase.analyzer,srcImg);
            
            % Pst
            angs = randn(size(get(lppufb,'Angles')));
            set(lppufb,'Angles',angs);
            coefsPst1 = step(testCase.analyzer,srcImg);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan
            diff = norm(coefsPst1(:)-coefsPre1(:));
            testCase.verifyThat(diff,IsGreaterThan(0),sprintf('%g',diff));            
           
        end
        
        % Test
        function testCloneTypeII(testCase)
            
            dec = [ 2 2 ];
            ch =  [ 5 3 ];
            ord = [ 4 4 ];
            height = 64;
            width  = 64;
            nLevels = 1;
            srcImg = rand(height,width);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord,...
                'OutputMode','ParameterMatrixSet');
            
            % Instantiation of target class
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'BoundaryOperation','Termination');
            %s = matlab.System.saveObject(testCase.analyzer);

            % Clone
            cloneAnalyzer = clone(testCase.analyzer);
            
            % Evaluation
            testCase.verifyEqual(cloneAnalyzer,testCase.analyzer);
            testCase.verifyFalse(cloneAnalyzer == testCase.analyzer);
            prpOrg = get(testCase.analyzer,'LpPuFb2d');
            prpCln = get(cloneAnalyzer,'LpPuFb2d');
            testCase.verifyEqual(prpCln,prpOrg);
            testCase.verifyFalse(prpCln == prpOrg);
            %            
            [coefExpctd,scaleExpctd] = step(testCase.analyzer,srcImg);
            [coefActual,scaleActual] = step(cloneAnalyzer,srcImg);
            testCase.assertEqual(coefActual,coefExpctd);
            testCase.assertEqual(scaleActual,scaleExpctd);
            
        end
        
        function testStepDec11Ch45Ord88Level3PeriodicExt(testCase)
            
            dec = 1;
            nChs = [ 4 5 ];
            decch = [ dec dec nChs ];
            ch = sum(nChs);
            ord = 8;
            height = 64;
            width = 64;
            srcImg = rand(height,width);
            nLevels = 3;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefsExpctdLv3 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv3{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv2{1},...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefs{1} = coefsExpctdLv3{1};
            coefs{2} = coefsExpctdLv3{2};
            coefs{3} = coefsExpctdLv3{3};
            coefs{4} = coefsExpctdLv3{4};
            coefs{5} = coefsExpctdLv3{5};
            coefs{6} = coefsExpctdLv3{6};
            coefs{7} = coefsExpctdLv3{7};
            coefs{8} = coefsExpctdLv3{8};
            coefs{9} = coefsExpctdLv3{9};
            coefs{10} = coefsExpctdLv2{2};
            coefs{11} = coefsExpctdLv2{3};
            coefs{12} = coefsExpctdLv2{4};
            coefs{13} = coefsExpctdLv2{5};
            coefs{14} = coefsExpctdLv2{6};
            coefs{15} = coefsExpctdLv2{7};
            coefs{16} = coefsExpctdLv2{8};
            coefs{17} = coefsExpctdLv2{9};
            coefs{18} = coefsExpctdLv1{2};
            coefs{19} = coefsExpctdLv1{3};
            coefs{20} = coefsExpctdLv1{4};
            coefs{21} = coefsExpctdLv1{5};
            coefs{22} = coefsExpctdLv1{6};
            coefs{23} = coefsExpctdLv1{7};
            coefs{24} = coefsExpctdLv1{8};
            coefs{25} = coefsExpctdLv1{9};
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,...
                sprintf('%g',diff));
            
        end
        
        function testStepDec22Ch45Ord44Level3PeriodicExt(testCase)
            
            dec = 2;
            nChs = [ 4 5 ];
            decch = [ dec dec nChs ];
            ch = sum(nChs);
            ord = 4;
            height = 64;
            width = 64;
            srcImg = rand(height,width);
            nLevels = 3;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',[dec dec],...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefsExpctdLv2 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefsExpctdLv3 = cell(ch,1);
            for iSubband = 1:ch
                coefsExpctdLv3{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv2{1},...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',dec).',dec);
            end
            coefs{1} = coefsExpctdLv3{1};
            coefs{2} = coefsExpctdLv3{2};
            coefs{3} = coefsExpctdLv3{3};
            coefs{4} = coefsExpctdLv3{4};
            coefs{5} = coefsExpctdLv3{5};
            coefs{6} = coefsExpctdLv3{6};
            coefs{7} = coefsExpctdLv3{7};
            coefs{8} = coefsExpctdLv3{8};
            coefs{9} = coefsExpctdLv3{9};
            coefs{10} = coefsExpctdLv2{2};
            coefs{11} = coefsExpctdLv2{3};
            coefs{12} = coefsExpctdLv2{4};
            coefs{13} = coefsExpctdLv2{5};
            coefs{14} = coefsExpctdLv2{6};
            coefs{15} = coefsExpctdLv2{7};
            coefs{16} = coefsExpctdLv2{8};
            coefs{17} = coefsExpctdLv2{9};
            coefs{18} = coefsExpctdLv1{2};
            coefs{19} = coefsExpctdLv1{3};
            coefs{20} = coefsExpctdLv1{4};
            coefs{21} = coefsExpctdLv1{5};
            coefs{22} = coefsExpctdLv1{6};
            coefs{23} = coefsExpctdLv1{7};
            coefs{24} = coefsExpctdLv1{8};
            coefs{25} = coefsExpctdLv1{9};
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec22Ch23Ord22Level1PeriodicExt(testCase)
            
            decch = [2 2 2 3];
            nChs = sum(decch(3:4));
            ord = 2;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/(decch(1)*decch(2));
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            for iSubband = 1:nChs
                subCoef = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',decch(2)).',decch(1));
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./decch(1:2),nChs,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,sprintf('%g',diff));
            
            
        end

        % Test
        function testStepDec22Ch23Ord22Level2PeriodicExt(testCase)
            
            decch = [2 2 2 3];
            nChs = sum(decch(3:4));
            ord = 2;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',decch(1)).',decch(2));
            end
            coefsExpctdLv2 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',decch(1)).',decch(2));
            end
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv2{5};
            coefs{6} = coefsExpctdLv1{2};
            coefs{7} = coefsExpctdLv1{3};
            coefs{8} = coefsExpctdLv1{4};
            coefs{9} = coefsExpctdLv1{5};
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,...
                sprintf('%g',diff));
        end
        
        % Test
        function testStepDec22Ch24Ord22Level1PeriodicExt(testCase)
            
            decch = [2 2 2 4];
            nChs = sum(decch(3:4));
            ord = 2;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 1;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            nSubCoefs = numel(srcImg)/(decch(1)*decch(2));
            coefsExpctd = zeros(1,nChs*nSubCoefs);
            for iSubband = 1:nChs
                subCoef = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',decch(2)).',decch(1));
                coefsExpctd((iSubband-1)*nSubCoefs+1:iSubband*nSubCoefs) = ...
                    subCoef(:).';
            end
            scalesExpctd = repmat(size(srcImg)./decch(1:2),nChs,1);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual, scalesActual] = step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,sprintf('%g',diff));
            
        end
        
        % Test
        function testStepDec22Ch24Ord22Level2PeriodicExt(testCase)
            
            decch = [2 2 2 4];
            nChs = sum(decch(3:4));
            ord = 2;
            height = 32;
            width = 32;
            srcImg = rand(height,width);
            nLevels = 2;
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',decch(1:2),...
                'NumberOfChannels',decch(3:end),...
                'PolyPhaseOrder',[ord ord],...
                'NumberOfVanishingMoments',0);
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            
            % Expected values
            release(lppufb)
            set(lppufb,'OutputMode','AnalysisFilterAt');
            coefsExpctdLv1 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv1{iSubband} = downsample(...
                    downsample(...
                    imfilter(srcImg,...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',decch(1)).',decch(2));
            end
            coefsExpctdLv2 = cell(nChs,1);
            for iSubband = 1:nChs
                coefsExpctdLv2{iSubband} = downsample(...
                    downsample(...
                    imfilter(coefsExpctdLv1{1},...
                    step(lppufb,[],[],iSubband),...
                    'conv','circ').',decch(1)).',decch(2));
            end
            coefs{1} = coefsExpctdLv2{1};
            coefs{2} = coefsExpctdLv2{2};
            coefs{3} = coefsExpctdLv2{3};
            coefs{4} = coefsExpctdLv2{4};
            coefs{5} = coefsExpctdLv2{5};
            coefs{6} = coefsExpctdLv2{6};
            coefs{7} = coefsExpctdLv1{2};
            coefs{8} = coefsExpctdLv1{3};
            coefs{9} = coefsExpctdLv1{4};
            coefs{10} = coefsExpctdLv1{5};
            coefs{11} = coefsExpctdLv1{6};
            nSubbands = length(coefs);
            scalesExpctd = zeros(nSubbands,2);
            sIdx = 1;
            for iSubband = 1:nSubbands
                scalesExpctd(iSubband,:) = size(coefs{iSubband});
                eIdx = sIdx + prod(scalesExpctd(iSubband,:))-1;
                coefsExpctd(sIdx:eIdx) = coefs{iSubband}(:).';
                sIdx = eIdx + 1;
            end
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'NumberOfSymmetricChannels',decch(3),...
                'NumberOfAntisymmetricChannels',decch(4),...
                'BoundaryOperation','Circular');
            
            % Actual values
            [coefsActual,scalesActual] = ...
                step(testCase.analyzer,srcImg);
            
            % Evaluation
            testCase.verifyEqual(scalesActual,scalesExpctd);
            diff = max(abs(coefsExpctd - coefsActual));
            testCase.verifyEqual(coefsActual,coefsExpctd,'AbsTol',1e-13,...
                sprintf('%g',diff));
            
        end
        
        % Test
        function testSetLpPuFb2dDec22Ch25Ord44(testCase)
            
            dec = [ 2 2 ];
            ch =  [ 2 5 ];
            ord = [ 4 4 ];
            height = 64;
            width = 64;
            nLevels = 1;
            srcImg = rand(height,width);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                
                'BoundaryOperation','Termination');
            coefsPre = step(testCase.analyzer,srcImg);
            %
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            coefsPst1 = step(testCase.analyzer,srcImg);
            
            % Evaluation
            diff = norm(coefsPst1(:)-coefsPre(:));
            testCase.verifyEqual(diff,0,'AbsTol',1e-15,...
                sprintf('%g',diff));
            
            % Reinstatiation
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                    
                'BoundaryOperation','Termination');
            coefsPst2 = step(testCase.analyzer,srcImg);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan
            diff = norm(coefsPst2(:)-coefsPre(:));
            testCase.verifyThat(diff,IsGreaterThan(0),sprintf('%g',diff));
        end
       
        % Test
        function testSetLpPuFb2dDec12Ch22Ord22(testCase)
            
            dec = [ 1 2 ];
            ch =  [ 2 2 ];
            ord = [ 2 2 ];
            height = 64;
            width = 64;
            nLevels = 1;
            srcImg = rand(height,width);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                                    
                'BoundaryOperation','Termination');
            coefsPre = step(testCase.analyzer,srcImg);
            %
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            coefsPst1 = step(testCase.analyzer,srcImg);
            
            % Evaluation
            diff = norm(coefsPst1(:)-coefsPre(:));
            testCase.verifyEqual(diff,0,'AbsTol',1e-15,...
                sprintf('%g',diff));
            
            % Reinstatiation
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                                    
                'BoundaryOperation','Termination');
            coefsPst2 = step(testCase.analyzer,srcImg);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan
            diff = norm(coefsPst2(:)-coefsPre(:));
            testCase.verifyThat(diff,IsGreaterThan(0),sprintf('%g',diff));
        end
        
        % Test
        function testSetLpPuFb2dDec21Ch22Ord22(testCase)
            
            dec = [ 2 1 ];
            ch =  [ 2 2 ];
            ord = [ 2 2 ];
            height = 64;
            width = 64;
            nLevels = 1;
            srcImg = rand(height,width);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                                    
                'BoundaryOperation','Termination');
            coefsPre = step(testCase.analyzer,srcImg);
            %
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            coefsPst1 = step(testCase.analyzer,srcImg);
            
            % Evaluation
            diff = norm(coefsPst1(:)-coefsPre(:));
            testCase.verifyEqual(diff,0,'AbsTol',1e-15,...
                sprintf('%g',diff));
            
            % Reinstatiation
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                                    
                'BoundaryOperation','Termination');
            coefsPst2 = step(testCase.analyzer,srcImg);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan
            diff = norm(coefsPst2(:)-coefsPre(:));
            testCase.verifyThat(diff,IsGreaterThan(0),sprintf('%g',diff));
        end
        
        % Test
        function testSetLpPuFb2dDec12Ch23Ord22(testCase)
            
            dec = [ 1 2 ];
            ch =  [ 2 3 ];
            ord = [ 2 2 ];
            height = 64;
            width = 64;
            nLevels = 1;
            srcImg = rand(height,width);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                                    
                'BoundaryOperation','Termination');
            coefsPre = step(testCase.analyzer,srcImg);
            %
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            coefsPst1 = step(testCase.analyzer,srcImg);
            
            % Evaluation
            diff = norm(coefsPst1(:)-coefsPre(:));
            testCase.verifyEqual(diff,0,'AbsTol',1e-15,...
                sprintf('%g',diff));
            
            % Reinstatiation
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                                    
                'BoundaryOperation','Termination');
            coefsPst2 = step(testCase.analyzer,srcImg);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan
            diff = norm(coefsPst2(:)-coefsPre(:));
            testCase.verifyThat(diff,IsGreaterThan(0),sprintf('%g',diff));
        end
        
        % Test
        function testSetLpPuFb2dDec21Ch23Ord22(testCase)
            
            dec = [ 2 1 ];
            ch =  [ 2 3 ];
            ord = [ 2 2 ];
            height = 64;
            width = 64;
            nLevels = 1;
            srcImg = rand(height,width);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                                    
                'BoundaryOperation','Termination');
            coefsPre = step(testCase.analyzer,srcImg);
            %
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            coefsPst1 = step(testCase.analyzer,srcImg);
            
            % Evaluation
            diff = norm(coefsPst1(:)-coefsPre(:));
            testCase.verifyEqual(diff,0,'AbsTol',1e-15,...
                sprintf('%g',diff));
            
            % Reinstatiation
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                                    
                'BoundaryOperation','Termination');
            coefsPst2 = step(testCase.analyzer,srcImg);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan
            diff = norm(coefsPst2(:)-coefsPre(:));
            testCase.verifyThat(diff,IsGreaterThan(0),sprintf('%g',diff));
        end
        
                % Test
        function testSetLpPuFb2dDec12Ch32Ord22(testCase)
            
            dec = [ 1 2 ];
            ch =  [ 3 2 ];
            ord = [ 2 2 ];
            height = 64;
            width = 64;
            nLevels = 1;
            srcImg = rand(height,width);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                                    
                'BoundaryOperation','Termination');
            coefsPre = step(testCase.analyzer,srcImg);
            %
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            coefsPst1 = step(testCase.analyzer,srcImg);
            
            % Evaluation
            diff = norm(coefsPst1(:)-coefsPre(:));
            testCase.verifyEqual(diff,0,'AbsTol',1e-15,...
                sprintf('%g',diff));
            
            % Reinstatiation
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                                    
                'BoundaryOperation','Termination');
            coefsPst2 = step(testCase.analyzer,srcImg);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan
            diff = norm(coefsPst2(:)-coefsPre(:));
            testCase.verifyThat(diff,IsGreaterThan(0),sprintf('%g',diff));
        end
        
        % Test
        function testSetLpPuFb2dDec21Ch32Ord22(testCase)
            
            dec = [ 2 1 ];
            ch =  [ 3 2 ];
            ord = [ 2 2 ];
            height = 64;
            width = 64;
            nLevels = 1;
            srcImg = rand(height,width);
            
            % Preparation
            import saivdr.dictionary.nsoltx.*
            lppufb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',dec,...
                'NumberOfChannels',ch,...
                'PolyPhaseOrder',ord);
            
            % Instantiation of target class
            release(lppufb)
            set(lppufb,'OutputMode','ParameterMatrixSet');
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                                    
                'BoundaryOperation','Termination');
            coefsPre = step(testCase.analyzer,srcImg);
            %
            angs = get(lppufb,'Angles');
            angs = randn(size(angs));
            set(lppufb,'Angles',angs);
            coefsPst1 = step(testCase.analyzer,srcImg);
            
            % Evaluation
            diff = norm(coefsPst1(:)-coefsPre(:));
            testCase.verifyEqual(diff,0,'AbsTol',1e-15,...
                sprintf('%g',diff));
            
            % Reinstatiation
            testCase.analyzer = NsoltAnalysis2dSystem(...
                'LpPuFb2d',lppufb,...
                'NumberOfLevels',nLevels,...                                    
                'BoundaryOperation','Termination');
            coefsPst2 = step(testCase.analyzer,srcImg);
            
            % Evaluation
            import matlab.unittest.constraints.IsGreaterThan
            diff = norm(coefsPst2(:)-coefsPre(:));
            testCase.verifyThat(diff,IsGreaterThan(0),sprintf('%g',diff));
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
    chDecY = int(math.ceil(decY_/2.)) #.astype(int)
    chDecX = int(math.ceil(decX_/2.)) #.astype(int)
    fhDecY = int(math.floor(decY_/2.)) #.astype(int)
    fhDecX = int(math.floor(decX_/2.)) #.astype(int)
    nQDecsee = chDecY*chDecX
    nQDecsoo = fhDecY*fhDecX
    nQDecsoe = fhDecY*chDecX
    cee = coefs[:,:nQDecsee]
    coo = coefs[:,nQDecsee:nQDecsee+nQDecsoo]
    coe = coefs[:,nQDecsee+nQDecsoo:nQDecsee+nQDecsoo+nQDecsoe]
    ceo = coefs[:,nQDecsee+nQDecsoo+nQDecsoe:]
    nBlocks = coefs.size(0)
    value = torch.zeros(nBlocks,decY_,decX_,dtype=x.dtype)
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