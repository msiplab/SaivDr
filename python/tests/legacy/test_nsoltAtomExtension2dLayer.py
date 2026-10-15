import itertools
import unittest
from parameterized import parameterized
import torch
import torch.nn as nn
from nsoltAtomExtension2dLayer import NsoltAtomExtension2dLayer

nchs = [ [3,3], [4,4] ]
datatype = [ torch.float, torch.double ]
nrows = [ 4, 8, 16 ]
ncols = [ 4, 8, 16 ]
dir = [ 'Right', 'Left', 'Up', 'Down' ]
target = [ 'Sum', 'Difference' ]

class NsoltAtomExtention2dLayerTestCase(unittest.TestCase):
    """
    NSOLTATOMEXTENSION2DLAYERTESTCASE
    
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

    @parameterized.expand(
        list(itertools.product(nchs,target))
    )
    def testConstructor(self,nchs,target):

        # Expctd values
        expctdName = 'Qn'
        expctdDirection = 'Right'
        expctdTargetChannels = target
        expctdDescription = "Right shift the " \
            + target.lower() \
            + "-channel Coefs. " \
            + "(ps,pa) = (" + str(nchs[0]) + "," + str(nchs[1]) + ")"
        
        # Instantiation of target class
        layer = NsoltAtomExtension2dLayer(
            number_of_channels=nchs,
            name=expctdName,
            direction=expctdDirection,
            target_channels=expctdTargetChannels
        )

        # Actual values
        actualName = layer.name 
        actualDirection = layer.direction 
        actualTargetChannels = layer.target_channels 
        actualDescription = layer.description 

        # Evaluation
        self.assertTrue(isinstance(layer, nn.Module))
        self.assertEqual(actualName,expctdName)
        self.assertEqual(actualDirection,expctdDirection)
        self.assertEqual(actualTargetChannels,expctdTargetChannels)
        self.assertEqual(actualDescription,expctdDescription)

    @parameterized.expand(
        list(itertools.product(nchs,nrows,ncols,dir,datatype))
    )
    def testPredictGrayscaleShiftDifferenceCoefs(self, 
            nchs, nrows, ncols, dir, datatype):
        rtol,atol=  0,1e-8
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")        

        # Parameters
        nSamples = 8
        nChsTotal = sum(nchs)
        target = 'Difference'
        # nSamples x nRows x nCols x nChsTotal  
        X = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype,device=device,requires_grad=True)
                
        # Expected values
        if dir=='Right':
            shift = ( 0, 0, 1, 0 )
        elif dir=='Left':
            shift = ( 0, 0, -1, 0 )
        elif dir=='Down':
            shift = ( 0, 1, 0, 0 )
        elif dir=='Up':
            shift = ( 0, -1, 0, 0 )
        else:
            shift = ( 0, 0, 0, 0 )
    
        # nSamples x nRows x nCols x nChsTotal 
        ps, pa = nchs
        Y = X 
        # Block butterfly
        Ys = Y[:,:,:,:ps]
        Ya = Y[:,:,:,ps:]
        Y =  torch.cat((Ys+Ya, Ys-Ya),dim=-1)
        # Block circular shift
        Y[:,:,:,ps:] = torch.roll(Y[:,:,:,ps:],shifts=shift,dims=(0,1,2,3))
        # Block butterfly
        Ys = Y[:,:,:,:ps]
        Ya = Y[:,:,:,ps:]
        Y =  torch.cat((Ys+Ya ,Ys-Ya),dim=-1)
        # Output
        expctdZ = Y/2. 

        # Instantiation of target class
        layer = NsoltAtomExtension2dLayer( 
            number_of_channels=nchs, 
            name='Qn~', 
            direction=dir, 
            target_channels=target
        )

        # Actual values
        with torch.no_grad():
            actualZ = layer.forward(X)
        
        # Evaluation
        self.assertEqual(actualZ.dtype,datatype) 
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))
        self.assertFalse(actualZ.requires_grad)

    @parameterized.expand(
        list(itertools.product(nchs,nrows,ncols,dir,datatype))
    )
    def testPredictGrayscaleShiftSumCoefs(self, 
                nchs, nrows, ncols, dir, datatype):
        rtol, atol= 1e-5, 1e-8
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")        

        # Parameters
        nSamples = 8
        nChsTotal = sum(nchs)
        target = 'Sum'
        # nSamples x nRows x nCols x nChsTotal 
        X = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype,device=device,requires_grad=True)
        
        # Expected values
        if dir=='Right':
            shift = ( 0, 0, 1, 0 )
        elif dir=='Left':
            shift = ( 0, 0, -1, 0 )
        elif dir=='Down':
            shift = ( 0, 1, 0, 0 )
        elif dir=='Up':
            shift = ( 0, -1, 0, 0 )
        else:
            shift = ( 0, 0, 0, 0 )

        # nSamples x nRows x nCols x nChsTotal 
        ps, pa = nchs
        Y = X
        # Block butterfly
        Ys = Y[:,:,:,:ps]
        Ya = Y[:,:,:,ps:]
        Y =  torch.cat((Ys+Ya, Ys-Ya),dim=-1)
        # Block circular shift
        Y[:,:,:,:ps] = torch.roll(Y[:,:,:,:ps],shifts=shift,dims=(0,1,2,3))
        # Block butterfly
        Ys = Y[:,:,:,:ps]
        Ya = Y[:,:,:,ps:]
        Y =  torch.cat((Ys+Ya, Ys-Ya),dim=-1)
        # Output
        expctdZ = Y/2.
        
        # Instantiation of target class
        layer = NsoltAtomExtension2dLayer( 
            number_of_channels=nchs, 
            name='Qn~', 
            direction=dir, 
            target_channels=target
        )
        
        # Actual values
        with torch.no_grad():
            actualZ = layer.forward(X)
            
        # Evaluation
        self.assertEqual(actualZ.dtype,datatype) 
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))
        self.assertFalse(actualZ.requires_grad)

    @parameterized.expand(
        list(itertools.product(nchs,nrows,ncols,dir,datatype))
    )
    def testBackwardGrayscaleShiftDifferenceCoefs(self, 
                nchs, nrows, ncols, dir, datatype):
        rtol,atol = 1e-5,1e-8
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")        

        # Parameters
        nSamples = 8
        nChsTotal = sum(nchs)
        target = 'Difference'

        # nSamples x nRows x nCols x nChsTotal
        X = torch.zeros(nSamples,nrows,ncols,nChsTotal,dtype=datatype,device=device,requires_grad=True)   
        dLdZ = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype)
        dLdZ = dLdZ.to(device)

        # Expected values        
        if dir=='Right':
            shift = ( 0, 0, -1, 0 ) # Reverse
        elif dir=='Left':
            shift = ( 0, 0, 1, 0 ) # Reverse
        elif dir=='Down':
            shift = ( 0, -1, 0, 0 ) # Reverse
        elif dir=='Up':
            shift = ( 0, 1, 0, 0 ) # Reverse
        else:
            shift = ( 0, 0, 0, 0 ) # Reverse

        # nSamples x nRows x nCols x nChsTotal 
        ps, pa = nchs
        Y = dLdZ
        
        # Block butterfly        
        Ys = Y[:,:,:,:ps]
        Ya = Y[:,:,:,ps:]
        Y = torch.cat((Ys+Ya,Ys-Ya),dim=-1)
        # Block circular shift
        Y[:,:,:,ps:] = torch.roll(Y[:,:,:,ps:],shifts=shift,dims=(0,1,2,3))        
        # Block butterfly        
        Ys = Y[:,:,:,:ps]
        Ya = Y[:,:,:,ps:]
        Y = torch.cat((Ys+Ya,Ys-Ya),dim=-1)

        # Output
        expctddLdX = Y/2.

        # Instantiation of target class
        layer = NsoltAtomExtension2dLayer(
            number_of_channels=nchs,
            name='Qn',
            direction=dir,
            target_channels=target
        )

        # Actual values
        Z = layer.forward(X)
        Z.backward(dLdZ)
        actualdLdX = X.grad
        
        # Evaluation
        self.assertEqual(actualdLdX.dtype,datatype) 
        self.assertTrue(torch.allclose(actualdLdX,expctddLdX,rtol=rtol,atol=atol))
        self.assertTrue(Z.requires_grad)

    @parameterized.expand(
        list(itertools.product(nchs,nrows,ncols,dir,datatype))
    )
    def testBackwardGrayscaleShiftSumCoefs(self, 
                nchs, nrows, ncols, dir, datatype):
        rtol,atol = 1e-5,1e-8
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")        

        # Parameters
        nSamples = 8
        nChsTotal = sum(nchs)
        target = 'Sum'
        
        # nSamples x nRows x nCols x nChsTotal
        X = torch.zeros(nSamples,nrows,ncols,nChsTotal,dtype=datatype,device=device,requires_grad=True)       
        dLdZ = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype)
        dLdZ = dLdZ.to(device)

        # Expected values
        if dir=='Right':
            shift = ( 0, 0, -1, 0) # Reverse
        elif dir=='Left':
            shift = ( 0, 0, 1, 0 ) # Reverse
        elif dir=='Down':
            shift = ( 0, -1, 0, 0 ) # Reverse
        elif dir=='Up':
            shift = ( 0, 1, 0, 0 ) # Reverse
        else:
            shift = ( 0, 0, 0, 0 )

        # nSamples x nRows x nCols x nChsTotal
        ps, pa = nchs
        Y = dLdZ

        # Block butterfly
        Ys = Y[:,:,:,:ps]
        Ya = Y[:,:,:,ps:]
        Y = torch.cat((Ys+Ya, Ys-Ya),dim=-1)
        # Block circular shift
        Y[:,:,:,:ps] = torch.roll(Y[:,:,:,:ps],shifts=shift,dims=(0,1,2,3))
        # Block butterfly
        Ys = Y[:,:,:,:ps]
        Ya = Y[:,:,:,ps:]
        Y = torch.cat((Ys+Ya, Ys-Ya),dim=-1)

        # Output
        expctddLdX = Y/2.

        # Instantiation of target class
        layer = NsoltAtomExtension2dLayer(
                number_of_channels=nchs,
                name='Qn',
                direction=dir,
                target_channels=target
        )
            
        # Actual values
        Z = layer.forward(X)
        Z.backward(dLdZ)
        actualdLdX = X.grad
        
        # Evaluation
        self.assertEqual(actualdLdX.dtype,datatype) 
        self.assertTrue(torch.allclose(actualdLdX,expctddLdX,rtol=rtol,atol=atol))
        self.assertTrue(Z.requires_grad)
        
if __name__ == '__main__':
    unittest.main()