import itertools
import unittest
from parameterized import parameterized
import math
import torch
import torch.nn as nn
from nsoltIntermediateRotation2dLayer import NsoltIntermediateRotation2dLayer
from nsoltUtility import Direction, OrthonormalMatrixGenerationSystem
from nsoltLayerExceptions import InvalidMode, InvalidMus

nchs = [ [2, 2], [3, 3], [4, 4] ]
mus = [ -1, 1 ]
datatype = [ torch.float, torch.double ]
nrows = [ 4, 8, 16 ]
ncols = [ 4, 8, 16 ]
isdevicetest = True

class NsoltIntermediateRotation2dLayerTestCase(unittest.TestCase):
    """
    NSOLTINTERMEDIATEROTATION2DLAYERTESTCASE
    
       コンポーネント別に入力(nComponents):
          nSamples x nRows x nCols x nChs
    
       コンポーネント別に出力(nComponents):
          nSamples x nRows x nCols x nChs
    
    Requirements: Python 3.7.x, PyTorch 1.7.x
    
    Copyright (c) 2021, Shogo MURAMATSU
    
    All rights reserved.
    
    Contact address: Shogo MURAMATSU,
        Faculty of Engineering, Niigata University,
        8050 2-no-cho Ikarashi, Nishi-ku,
        Niigata, 950-2181, JAPAN
    
        http://msiplab.eng.niigata-u.ac.jp/ 
    """

    @parameterized.expand(
        list(itertools.product(nchs))
    )
    def testConstructor(self,
        nchs):

        # Expected values
        expctdName = 'Vn~'
        expctdMode = 'Synthesis'
        expctdDescription = "Synthesis NSOLT intermediate rotation " \
                + "(ps,pa) = (" \
                + str(nchs[0]) + "," + str(nchs[1]) + ")"

        # Instantiation of target class
        layer = NsoltIntermediateRotation2dLayer(
                number_of_channels=nchs,
                name=expctdName
            )
        
        # Actual values
        actualName = layer.name
        actualMode = layer.mode
        actualDescription = layer.description

        # Evaluation
        self.assertTrue(isinstance(layer, nn.Module))
        self.assertEqual(actualName,expctdName)
        self.assertEqual(actualMode,expctdMode)
        self.assertEqual(actualDescription,expctdDescription)

    @parameterized.expand(
        list(itertools.product(nchs,nrows,ncols,mus,datatype))
    )
    def testPredictGrayscale(self,
        nchs, nrows, ncols, mus, datatype):
        rtol,atol=1e-5,1e-8
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")        

        # Parameters
        nSamples = 8
        nChsTotal = sum(nchs)
        # nSamples x nRows x nCols x nChsTotal
        X = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype,device=device,requires_grad=True)

        # Expected values
        # nSamples x nRows x nCols x nChsTotal
        ps,pa = nchs
        UnT = mus*torch.eye(pa,dtype=datatype).to(device)
        expctdZ = X.clone()
        Ya = X[:,:,:,ps:].view(-1,pa).T
        Za = UnT @ Ya
        expctdZ[:,:,:,ps:] = Za.T.view(nSamples,nrows,ncols,pa)

        # Instantiation of target class
        layer = NsoltIntermediateRotation2dLayer(
            number_of_channels=nchs,
            name='Vn~')
        layer.orthTransUn.mus = mus
        layer = layer.to(device)

        # Actual values
        with torch.no_grad():
            actualZ = layer.forward(X)

        # Evaluation
        self.assertEqual(actualZ.dtype,datatype)
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))
        self.assertFalse(actualZ.requires_grad)

    @parameterized.expand(
        list(itertools.product(nchs,nrows,ncols,mus,datatype))
    )
    def testPredictGrayscaleWithRandomAngles(self,
        nchs, nrows, ncols, mus, datatype):
        rtol,atol=1e-3,1e-6
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")           
        gen = OrthonormalMatrixGenerationSystem(dtype=datatype)

        # Parameters
        nSamples = 8
        nChsTotal = sum(nchs)
        # nSamples x nRows x nCols x nChsTotal
        X = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype,device=device,requires_grad=True)
        angles = torch.randn(int((nChsTotal-2)*nChsTotal/8),dtype=datatype)

        # Expected values
        # nSamples x nRows x nCols x nChsTotal
        ps,pa = nchs
        UnT = gen(angles,mus).T.to(device)
        expctdZ = X.clone()
        Ya = X[:,:,:,ps:].view(-1,pa).T
        Za = UnT @ Ya
        expctdZ[:,:,:,ps:] = Za.T.view(nSamples,nrows,ncols,pa)

        # Instantiation of target class
        layer = NsoltIntermediateRotation2dLayer(
            number_of_channels=nchs,
            name='Vn~')
        layer.orthTransUn.angles.data = angles
        layer.orthTransUn.mus = mus
        layer = layer.to(device)

        # Actual values
        with torch.no_grad():
            actualZ = layer.forward(X)

        # Evaluation
        self.assertEqual(actualZ.dtype,datatype)
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))
        self.assertFalse(actualZ.requires_grad)

    @parameterized.expand(
        list(itertools.product(nchs,nrows,ncols,mus,datatype))
    )
    def testPredictGrayscaleAnalysisMode(self,
        nchs, nrows, ncols, mus, datatype):
        rtol,atol=1e-3,1e-6
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")           
        gen = OrthonormalMatrixGenerationSystem(dtype=datatype)

        # Parameters
        nSamples = 8
        nChsTotal = sum(nchs)
        # nSamples x nRows x nCols x nChsTotal
        X = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype,device=device,requires_grad=True)
        angles = torch.randn(int((nChsTotal-2)*nChsTotal/8),dtype=datatype)

        # Expected values
        # nSamples x nRows x nCols x nChsTotal
        ps,pa = nchs
        Un = gen(angles,mus).to(device)
        expctdZ = X.clone()
        Ya = X[:,:,:,ps:].view(-1,pa).T
        Za = Un @ Ya
        expctdZ[:,:,:,ps:] = Za.T.view(nSamples,nrows,ncols,pa)

        # Instantiation of target class
        layer = NsoltIntermediateRotation2dLayer(
            number_of_channels=nchs,
            name='Vn',
            mode='Analysis')
        layer.orthTransUn.angles.data = angles
        layer.orthTransUn.mus = mus
        layer = layer.to(device)

        # Actual values
        with torch.no_grad():
            actualZ = layer.forward(X)

        # Evaluation
        self.assertEqual(actualZ.dtype,datatype)
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))
        self.assertFalse(actualZ.requires_grad)

    @parameterized.expand(
        list(itertools.product(datatype,nchs,nrows,ncols,mus)) 
    )
    def testBackwardGrayscale(self,
        datatype, nchs, nrows, ncols, mus):
        rtol,atol=1e-3,1e-6
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")          
        omgs = OrthonormalMatrixGenerationSystem(dtype=datatype,partial_difference=False)

        # Parameters
        nSamples = 8
        nChsTotal = sum(nchs)
        nAngles = int((nChsTotal-2)*nChsTotal/8)
        angles = torch.zeros(nAngles,dtype=datatype)
        # nSamples x nRows x nCols x nChsTotal
        X = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype,device=device,requires_grad=True)
        dLdZ = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype)       
        dLdZ = dLdZ.to(device)     

        # Expected values
        ps,pa = nchs
        Un = omgs(angles,mus).to(device)
        # dLdX = dZdX x dLdZ
        expctddLdX = dLdZ.clone()
        Ya = dLdZ[:,:,:,ps:].view(nSamples*nrows*ncols,pa).T # pa * n
        Za = Un @ Ya
        expctddLdX[:,:,:,ps:] = Za.T.view(nSamples,nrows,ncols,pa)
        # dLdWi = <dLdZ,(dVdWi)X>
        expctddLdW_U = torch.zeros(nAngles,dtype=datatype).to(device)        
        omgs.partial_difference = True
        for iAngle in range(nAngles):
            dUn_T = omgs(angles,mus,index_pd_angle=iAngle).T.to(device)
            Xa = X[:,:,:,ps:].view(-1,pa).T
            Za = dUn_T @ Xa # pa x n 
            expctddLdW_U[iAngle] = torch.sum(Ya * Za)

        # Instantiation of target class
        layer = NsoltIntermediateRotation2dLayer(
            number_of_channels=nchs,
            name='Vn~')
        layer.orthTransUn.angles.data = angles
        layer.orthTransUn.mus = mus
        layer = layer.to(device)

        # Actual values
        torch.autograd.set_detect_anomaly(True)        
        Z = layer.forward(X)
        layer.zero_grad()        
        Z.backward(dLdZ)
        actualdLdX = X.grad
        actualdLdW_U = layer.orthTransUn.angles.grad
    
        # Evaluation
        self.assertEqual(actualdLdX.dtype,datatype)
        self.assertEqual(actualdLdW_U.dtype,datatype)
        self.assertTrue(torch.allclose(actualdLdX,expctddLdX,rtol=rtol,atol=atol))
        self.assertTrue(torch.allclose(actualdLdW_U,expctddLdW_U,rtol=rtol,atol=atol))
        self.assertTrue(Z.requires_grad)

    @parameterized.expand(
        list(itertools.product(datatype,nchs,nrows,ncols,mus)) 
    )
    def testBackwardGrayscaleWithRandomAngles(self,
        datatype, nchs, nrows, ncols, mus):
        rtol,atol=1e-3,1e-6
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")           
        omgs = OrthonormalMatrixGenerationSystem(dtype=datatype,partial_difference=False)

        # Parameters
        nSamples = 8
        nChsTotal = sum(nchs)
        nAngles = int((nChsTotal-2)*nChsTotal/8)
        angles = torch.randn(nAngles,dtype=datatype)
        # nSamples x nRows x nCols x nChsTotal
        X = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype,device=device,requires_grad=True)
        dLdZ = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype)        
        dLdZ = dLdZ.to(device)    

        # Expected values
        ps,pa = nchs
        Un = omgs(angles,mus).to(device)
        # dLdX = dZdX x dLdZ
        expctddLdX = dLdZ.clone()
        Ya = dLdZ[:,:,:,ps:].view(nSamples*nrows*ncols,pa).T # pa * n
        Za = Un @ Ya
        expctddLdX[:,:,:,ps:] = Za.T.view(nSamples,nrows,ncols,pa)
        # dLdWi = <dLdZ,(dVdWi)X>
        expctddLdW_U = torch.zeros(nAngles,dtype=datatype).to(device)     
        omgs.partial_difference = True
        for iAngle in range(nAngles):
            dUn_T = omgs(angles,mus,index_pd_angle=iAngle).T.to(device)
            Xa = X[:,:,:,ps:].view(-1,pa).T
            Za = dUn_T @ Xa # pa x n 
            expctddLdW_U[iAngle] = torch.sum(Ya * Za)

        # Instantiation of target class
        layer = NsoltIntermediateRotation2dLayer(
            number_of_channels=nchs,
            name='Vn~')
        layer.orthTransUn.angles.data = angles
        layer.orthTransUn.mus = mus
        layer = layer.to(device)

        # Actual values
        torch.autograd.set_detect_anomaly(True)        
        Z = layer.forward(X)
        layer.zero_grad()        
        Z.backward(dLdZ)
        actualdLdX = X.grad
        actualdLdW_U = layer.orthTransUn.angles.grad
    
        # Evaluation
        self.assertEqual(actualdLdX.dtype,datatype)
        self.assertEqual(actualdLdW_U.dtype,datatype)
        self.assertTrue(torch.allclose(actualdLdX,expctddLdX,rtol=rtol,atol=atol))
        self.assertTrue(torch.allclose(actualdLdW_U,expctddLdW_U,rtol=rtol,atol=atol))
        self.assertTrue(Z.requires_grad)

    @parameterized.expand(
        list(itertools.product(datatype,nchs,nrows,ncols,mus)) 
    )
    def testBackwardGrayscaleAnalysisMode(self,
        datatype, nchs, nrows, ncols, mus):
        rtol,atol=1e-3,1e-6
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")         
        omgs = OrthonormalMatrixGenerationSystem(dtype=datatype,partial_difference=False)

        # Parameters
        nSamples = 8
        nChsTotal = sum(nchs)
        nAngles = int((nChsTotal-2)*nChsTotal/8)
        angles = torch.randn(nAngles,dtype=datatype)
        # nSamples x nRows x nCols x nChsTotal
        X = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype,device=device,requires_grad=True)
        dLdZ = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype)       
        dLdZ = dLdZ.to(device)     

        # Expected values
        ps,pa = nchs
        UnT = omgs(angles,mus).T.to(device)
        # dLdX = dZdX x dLdZ
        expctddLdX = dLdZ.clone()
        Ya = dLdZ[:,:,:,ps:].view(nSamples*nrows*ncols,pa).T # pa * n
        Za = UnT @ Ya
        expctddLdX[:,:,:,ps:] = Za.T.view(nSamples,nrows,ncols,pa)
        # dLdWi = <dLdZ,(dVdWi)X>
        expctddLdW_U = torch.zeros(nAngles,dtype=datatype).to(device)     
        omgs.partial_difference = True
        for iAngle in range(nAngles):
            dUn = omgs(angles,mus,index_pd_angle=iAngle).to(device)
            Xa = X[:,:,:,ps:].view(-1,pa).T
            Za = dUn @ Xa # pa x n 
            expctddLdW_U[iAngle] = torch.sum(Ya * Za)

        # Instantiation of target class
        layer = NsoltIntermediateRotation2dLayer(
            number_of_channels=nchs,
            mode='Analysis',
            name='Vn')
        layer.orthTransUn.angles.data = angles
        layer.orthTransUn.mus = mus
        layer = layer.to(device)

        # Actual values
        torch.autograd.set_detect_anomaly(True)        
        Z = layer.forward(X)
        layer.zero_grad()        
        Z.backward(dLdZ)
        actualdLdX = X.grad
        actualdLdW_U = layer.orthTransUn.angles.grad
    
        # Evaluation
        self.assertEqual(actualdLdX.dtype,datatype)
        self.assertEqual(actualdLdW_U.dtype,datatype)
        self.assertTrue(torch.allclose(actualdLdX,expctddLdX,rtol=rtol,atol=atol))
        self.assertTrue(torch.allclose(actualdLdW_U,expctddLdW_U,rtol=rtol,atol=atol))
        self.assertTrue(Z.requires_grad)
    
    def testInvalidModeException(self):
        nchs = [2,2]
        with self.assertRaises(InvalidMode):
            NsoltIntermediateRotation2dLayer(
                number_of_channels=nchs,
                mode='Dummy')

    def testInvalidMusException(self):
        nchs = [2,2]
        with self.assertRaises(InvalidMus):
            NsoltIntermediateRotation2dLayer(
                number_of_channels=nchs,
                mus=2)

    @parameterized.expand(
        list(itertools.product(nchs,nrows,ncols,mus,datatype))
    )
    def testConstructionWithMus(self,
        nchs, nrows, ncols, mus, datatype):
        rtol,atol=1e-5,1e-8
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")          

        # Parameters
        nSamples = 8
        nChsTotal = sum(nchs)
        # nSamples x nRows x nCols x nChsTotal
        X = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype,device=device,requires_grad=True)

        # Expected values
        # nSamples x nRows x nCols x nChsTotal
        ps,pa = nchs
        UnT = mus*torch.eye(pa,dtype=datatype).to(device)
        expctdZ = X.clone()
        Ya = X[:,:,:,ps:].view(-1,pa).T
        Za = UnT @ Ya
        expctdZ[:,:,:,ps:] = Za.T.view(nSamples,nrows,ncols,pa)

        # Instantiation of target class
        layer = NsoltIntermediateRotation2dLayer(
            number_of_channels=nchs,
            name='Vn~',
            mus = mus)
        layer = layer.to(device)

        # Actual values
        with torch.no_grad():
            actualZ = layer.forward(X)

        # Evaluation
        self.assertEqual(actualZ.dtype,datatype)
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))
        self.assertFalse(actualZ.requires_grad)

if __name__ == '__main__':
    unittest.main()
