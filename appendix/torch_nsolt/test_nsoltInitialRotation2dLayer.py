import itertools
import unittest
from parameterized import parameterized
import math
import torch
import torch.nn as nn
from nsoltInitialRotation2dLayer import NsoltInitialRotation2dLayer
from nsoltUtility import Direction, OrthonormalMatrixGenerationSystem

nchs = [ [2, 2], [3, 3], [4, 4] ]
stride = [ [1, 1], [1, 2], [2, 1], [2, 2] ]
mus = [ -1, 1 ]
datatype = [ torch.float, torch.double ]
nrows = [ 4, 8, 16 ]
ncols = [ 4, 8, 16 ]
isdevicetest = True

class NsoltInitialRotation2dLayerTestCase(unittest.TestCase):
    """
    NSOLTINITIALROTATION2DLAYERTESTCASE
    
       コンポーネント別に入力(nComponents):
          nSamples x nRows x nCols x nDecs
    
       コンポーネント別に出力(nComponents):
          nSamples x nRows x nCols x nChs
    
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
        list(itertools.product(nchs,stride))
    )
    def testConstructor(self,
        nchs, stride):
        
        # Expcted values
        expctdName = 'V0'
        expctdDescription = "NSOLT initial rotation " \
                + "(ps,pa) = (" \
                + str(nchs[0]) + "," + str(nchs[1]) + "), "  \
                + "(mv,mh) = (" \
                + str(stride[Direction.VERTICAL]) + "," + str(stride[Direction.HORIZONTAL]) + ")"

        # Instantiation of target class
        layer = NsoltInitialRotation2dLayer(
                number_of_channels=nchs,
                decimation_factor=stride,
                name=expctdName
            )            

        # Actual values
        actualName = layer.name
        actualDescription = layer.description

        # Evaluation
        self.assertTrue(isinstance(layer, nn.Module))
        self.assertEqual(actualName,expctdName)
        self.assertEqual(actualDescription,expctdDescription)

    @parameterized.expand(
        list(itertools.product(nchs,stride,nrows,ncols,datatype))
    )
    def testPredictGrayscale(self,
        nchs, stride, nrows, ncols, datatype):
        rtol,atol=1e-5,1e-8
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")     
        
        # Parameters
        nSamples = 8
        nDecs = stride[0]*stride[1] # math.prod(stride)
        nChsTotal = sum(nchs)
        # nSamples x nRows x nCols x nDecs
        X = torch.randn(nSamples,nrows,ncols,nDecs,dtype=datatype)
        X = X.to(device)

        # Expected values
        # nSamplex x nRows x nCols x nChs
        ps, pa = nchs
        W0 = torch.eye(ps,dtype=datatype).to(device)
        U0 = torch.eye(pa,dtype=datatype).to(device)
        ms,ma = int(math.ceil(nDecs/2.)), int(math.floor(nDecs/2.))        
        Zsa = torch.zeros(nChsTotal,nrows*ncols*nSamples,dtype=datatype).to(device)       
        Ys = X[:,:,:,:ms].view(-1,ms).T
        Zsa[:ps,:] = W0[:,:ms] @ Ys        
        if ma > 0:
            Ya = X[:,:,:,ms:].view(-1,ma).T
            Zsa[ps:,:] = U0[:,:ma] @ Ya
        expctdZ = Zsa.T.view(nSamples,nrows,ncols,nChsTotal)

        # Instantiation of target class
        layer = NsoltInitialRotation2dLayer(
                number_of_channels=nchs,
                decimation_factor=stride,
                name='V0'
            )
        layer = layer.to(device)
        
        # Actual values
        with torch.no_grad():
            actualZ = layer.forward(X)

        # Evaluation
        self.assertEqual(actualZ.dtype,datatype)
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))
        self.assertFalse(actualZ.requires_grad)

    @parameterized.expand(
        list(itertools.product(nchs,stride,nrows,ncols,datatype))
    )
    def testPredictGrayscaleWithRandomAngles(self,
        nchs, stride, nrows, ncols, datatype):
        rtol,atol=1e-3,1e-6
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")         
        gen = OrthonormalMatrixGenerationSystem(dtype=datatype)
        
        # Parameters
        nSamples = 8
        nDecs = stride[0]*stride[1] # math.prod(stride)
        nChsTotal = sum(nchs)
        # nSamples x nRows x nCols x nDecs
        X = torch.randn(nSamples,nrows,ncols,nDecs,dtype=datatype)
        X = X.to(device)
        angles = torch.randn(int((nChsTotal-2)*nChsTotal/4),dtype=datatype)

        # Expected values
        # nSamples x nRows x nCols x nChs
        ps,pa = nchs
        nAngsW = int(len(angles)/2)
        angsW,angsU = angles[:nAngsW],angles[nAngsW:]
        W0,U0 = gen(angsW).to(device),gen(angsU).to(device)
        ms,ma = int(math.ceil(nDecs/2.)), int(math.floor(nDecs/2.))                
        Zsa = torch.zeros(nChsTotal,nrows*ncols*nSamples,dtype=datatype)
        Zsa = Zsa.to(device)
        Ys = X[:,:,:,:ms].view(-1,ms).T 
        Zsa[:ps,:] = W0[:,:ms] @ Ys
        if ma > 0:
            Ya = X[:,:,:,ms:].view(-1,ma).T 
            Zsa[ps:,:] = U0[:,:ma] @ Ya
        expctdZ = Zsa.T.view(nSamples,nrows,ncols,nChsTotal)

        # Instantiation of target class
        layer = NsoltInitialRotation2dLayer(
            number_of_channels=nchs,
            decimation_factor=stride,
            name='V0')
        layer.orthTransW0.angles.data = angsW
        layer.orthTransW0.mus = 1
        layer.orthTransU0.angles.data = angsU
        layer.orthTransU0.mus = 1
        layer = layer.to(device)

        # Actual values
        with torch.no_grad():
            actualZ = layer.forward(X)
        
        # Evaluation
        self.assertEqual(actualZ.dtype,datatype)
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))
        self.assertFalse(actualZ.requires_grad)

    @parameterized.expand(
        list(itertools.product(nchs,stride,nrows,ncols,datatype,mus))
    )
    def testPredictGrayscaleWithRandomAnglesNoDcLeackage(self,
        nchs, stride, nrows, ncols, datatype,mus):
        rtol,atol=1e-3,1e-6
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")           
        gen = OrthonormalMatrixGenerationSystem(dtype=datatype)
        
        # Parameters
        nSamples = 8
        nDecs = stride[0]*stride[1] # math.prod(stride)
        nChsTotal = sum(nchs)
        # nSamples x nRows x nCols x nDecs
        X = torch.randn(nSamples,nrows,ncols,nDecs,dtype=datatype)
        X = X.to(device)
        angles = torch.randn(int((nChsTotal-2)*nChsTotal/4),dtype=datatype)

        # Expected values
        # nSamples x nRows x nCols x nChs
        ps,pa = nchs
        nAngsW = int(len(angles)/2)
        angsW,angsU = angles[:nAngsW],angles[nAngsW:]
        angsWNoDcLeak = angsW.clone()
        angsWNoDcLeak[:ps-1] = torch.zeros(ps-1,dtype=angles.dtype)
        musW,musU = mus*torch.ones(ps,dtype=datatype),mus*torch.ones(pa,dtype=datatype)
        musW[0] = 1
        W0,U0 = gen(angsWNoDcLeak,musW).to(device),gen(angsU,musU).to(device)
        ms,ma = int(math.ceil(nDecs/2.)), int(math.floor(nDecs/2.))                
        Zsa = torch.zeros(nChsTotal,nrows*ncols*nSamples,dtype=datatype).to(device)
        Ys = X[:,:,:,:ms].view(-1,ms).T 
        Zsa[:ps,:] = W0[:,:ms] @ Ys
        if ma > 0:
            Ya = X[:,:,:,ms:].view(-1,ma).T 
            Zsa[ps:,:] = U0[:,:ma] @ Ya
        expctdZ = Zsa.T.view(nSamples,nrows,ncols,nChsTotal)

        # Instantiation of target class
        layer = NsoltInitialRotation2dLayer(
            number_of_channels=nchs,
            decimation_factor=stride,
            no_dc_leakage=True,
            name='V0')
        layer.orthTransW0.angles.data = angsW
        layer.orthTransW0.mus = musW
        layer.orthTransU0.angles.data = angsU
        layer.orthTransU0.mus = musU
        layer = layer.to(device)

        # Actual values
        with torch.no_grad():
            actualZ = layer.forward(X)
        
        # Evaluation
        self.assertEqual(actualZ.dtype,datatype)
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))
        self.assertFalse(actualZ.requires_grad)

    @parameterized.expand(
        list(itertools.product(nchs,stride,nrows,ncols,datatype))
    )
    def testBackwardGrayscale(self,
        nchs, stride, nrows, ncols, datatype):
        rtol,atol=1e-5,1e-8
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")            
        omgs = OrthonormalMatrixGenerationSystem(dtype=datatype,partial_difference=False)

        # Parameters
        nSamples = 8
        nDecs = stride[0]*stride[1] # math.prod(stride)
        nChsTotal = sum(nchs)
        nAnglesH = int((nChsTotal-2)*nChsTotal/8)
        anglesW = torch.zeros(nAnglesH,dtype=datatype)
        anglesU = torch.zeros(nAnglesH,dtype=datatype)
        mus = 1
        # nSamples x nRows x nCols x nDecs
        X = torch.randn(nSamples,nrows,ncols,nDecs,dtype=datatype,device=device,requires_grad=True)
        dLdZ = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype)
        dLdZ = dLdZ.to(device)

        # Expected values
        # nSamples x nRows x nCols x nDecs
        ps,pa = nchs
        W0T = omgs(anglesW,mus).T.to(device)
        U0T = omgs(anglesU,mus).T.to(device)
        # dLdX = dZdX x dLdZ
        ms,ma = int(math.ceil(nDecs/2.)),int(math.floor(nDecs/2.))
        Ys = dLdZ[:,:,:,:ps].view(nSamples*nrows*ncols,ps).T # ps * n
        Ya = dLdZ[:,:,:,ps:].view(nSamples*nrows*ncols,pa).T # pa * n
        Y = torch.cat(
            ( W0T[:ms,:] @ Ys,          # ms x ps @ ps x n
              U0T[:ma,:] @ Ya ), dim=0) # ma x pa @ pa x n
        expctddLdX = Y.T.view(nSamples,nrows,ncols,nDecs) # n x (ms+ma)
        # dLdWi = <dLdZ,(dVdWi)X>
        expctddLdW_W = torch.zeros(nAnglesH,dtype=datatype).to(device)
        expctddLdW_U = torch.zeros(nAnglesH,dtype=datatype).to(device)
        omgs.partial_difference = True
        for iAngle in range(nAnglesH):
            dW0 = omgs(anglesW,mus,index_pd_angle=iAngle).to(device)
            Xs = X[:,:,:,:ms].view(-1,ms).T 
            Zs = dW0[:,:ms] @ Xs # ps x n
            expctddLdW_W[iAngle] = torch.sum(Ys * Zs) # ps x n
            if ma>0:
                dU0 = omgs(anglesU,mus,index_pd_angle=iAngle).to(device)
                Xa = X[:,:,:,ms:].view(-1,ma).T
                Za = dU0[:,:ma] @ Xa # pa x n            
                expctddLdW_U[iAngle] = torch.sum(Ya * Za) # pa x n
            
        # Instantiation of target class
        layer = NsoltInitialRotation2dLayer(
            number_of_channels=nchs,
            decimation_factor=stride,
            name='V0')
        layer.orthTransW0.angles.data = anglesW
        layer.orthTransW0.mus = mus
        layer.orthTransU0.angles.data = anglesU
        layer.orthTransU0.mus = mus
        layer = layer.to(device)

        # Actual values
        torch.autograd.set_detect_anomaly(True)
        Z = layer.forward(X)
        layer.zero_grad()
        Z.backward(dLdZ)
        actualdLdX = X.grad
        actualdLdW_W = layer.orthTransW0.angles.grad
        actualdLdW_U = layer.orthTransU0.angles.grad

        # Evaluation
        self.assertEqual(actualdLdX.dtype,datatype)
        self.assertEqual(actualdLdW_W.dtype,datatype)
        self.assertEqual(actualdLdW_U.dtype,datatype)
        self.assertTrue(torch.allclose(actualdLdX,expctddLdX,rtol=rtol,atol=atol))
        self.assertTrue(torch.allclose(actualdLdW_W,expctddLdW_W,rtol=rtol,atol=atol))
        self.assertTrue(torch.allclose(actualdLdW_U,expctddLdW_U,rtol=rtol,atol=atol))
        self.assertTrue(Z.requires_grad)


    @parameterized.expand(
        list(itertools.product(nchs,stride,nrows,ncols,datatype))
    )
    def testBackwardGrayscaleWithRandomAngles(self,
        nchs, stride, nrows, ncols, datatype):
        rtol,atol=1e-3,1e-6
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")          
        omgs = OrthonormalMatrixGenerationSystem(dtype=datatype,partial_difference=False)

        # Parameters
        nSamples = 8
        nDecs = stride[0]*stride[1] # math.prod(stride)
        nChsTotal = sum(nchs)
        nAnglesH = int((nChsTotal-2)*nChsTotal/8)
        anglesW = torch.randn(nAnglesH,dtype=datatype)
        anglesU = torch.randn(nAnglesH,dtype=datatype)
        mus = 1
        # nSamples x nRows x nCols x nDecs
        X = torch.randn(nSamples,nrows,ncols,nDecs,dtype=datatype,device=device,requires_grad=True)
        dLdZ = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype)
        dLdZ = dLdZ.to(device)

        # Expected values
        ps,pa = nchs
        W0T = omgs(anglesW,mus).T.to(device)
        U0T = omgs(anglesU,mus).T.to(device)
        # dLdX = dZdX x dLdZ
        ms,ma = int(math.ceil(nDecs/2.)),int(math.floor(nDecs/2.))
        Ys = dLdZ[:,:,:,:ps].view(nSamples*nrows*ncols,ps).T # ps * n
        Ya = dLdZ[:,:,:,ps:].view(nSamples*nrows*ncols,pa).T # pa * n
        Y = torch.cat(
            ( W0T[:ms,:] @ Ys,          # ms x ps @ ps x n
              U0T[:ma,:] @ Ya ), dim=0) # ma x pa @ pa x n
        expctddLdX = Y.T.view(nSamples,nrows,ncols,nDecs) # n x (ms+ma)
        # dLdWi = <dLdZ,(dVdWi)X>
        expctddLdW_W = torch.zeros(nAnglesH,dtype=datatype).to(device)
        expctddLdW_U = torch.zeros(nAnglesH,dtype=datatype).to(device)
        omgs.partial_difference = True
        for iAngle in range(nAnglesH):
            dW0 = omgs(anglesW,mus,index_pd_angle=iAngle).to(device)
            Xs = X[:,:,:,:ms].view(-1,ms).T 
            Zs = dW0[:,:ms] @ Xs # ps x n
            expctddLdW_W[iAngle] = torch.sum(Ys * Zs) # ps x n
            if ma>0:
                dU0 = omgs(anglesU,mus,index_pd_angle=iAngle).to(device)
                Xa = X[:,:,:,ms:].view(-1,ma).T
                Za = dU0[:,:ma] @ Xa # pa x n            
                expctddLdW_U[iAngle] = torch.sum(Ya * Za) # pa x n
            
        # Instantiation of target class
        layer = NsoltInitialRotation2dLayer(
            number_of_channels=nchs,
            decimation_factor=stride,
            name='V0')
        layer.orthTransW0.angles.data = anglesW
        layer.orthTransW0.mus = mus
        layer.orthTransU0.angles.data = anglesU
        layer.orthTransU0.mus = mus
        layer = layer.to(device)

        # Actual values
        torch.autograd.set_detect_anomaly(True)
        Z = layer.forward(X)
        layer.zero_grad()
        Z.backward(dLdZ)
        actualdLdX = X.grad
        actualdLdW_W = layer.orthTransW0.angles.grad
        actualdLdW_U = layer.orthTransU0.angles.grad

        # Evaluation
        self.assertEqual(actualdLdX.dtype,datatype)
        self.assertEqual(actualdLdW_W.dtype,datatype)
        self.assertEqual(actualdLdW_U.dtype,datatype)
        self.assertTrue(torch.allclose(actualdLdX,expctddLdX,rtol=rtol,atol=atol))
        self.assertTrue(torch.allclose(actualdLdW_W,expctddLdW_W,rtol=rtol,atol=atol))
        self.assertTrue(torch.allclose(actualdLdW_U,expctddLdW_U,rtol=rtol,atol=atol))
        self.assertTrue(Z.requires_grad)

    @parameterized.expand(
        list(itertools.product(nchs,stride,nrows,ncols,datatype,mus))
    )
    def testBackwardGrayscaleWithRandomAnglesNoDcLeackage(self,
        nchs, stride, nrows, ncols, datatype,mus):
        rtol,atol=1e-2,1e-5
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")          
        omgs = OrthonormalMatrixGenerationSystem(dtype=datatype,partial_difference=False)

        # Parameters
        nSamples = 8
        nDecs = stride[0]*stride[1] # math.prod(stride)
        nChsTotal = sum(nchs)
        nAnglesH = int((nChsTotal-2)*nChsTotal/8)
        anglesW = torch.randn(nAnglesH,dtype=datatype)
        anglesU = torch.randn(nAnglesH,dtype=datatype)
        mus = 1
        # nSamples x nRows x nCols x nDecs
        X = torch.randn(nSamples,nrows,ncols,nDecs,dtype=datatype,device=device,requires_grad=True)
        dLdZ = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype)
        dLdZ = dLdZ.to(device)

        # Expected values
        ps,pa = nchs
        anglesWNoDcLeak = anglesW.clone()
        anglesWNoDcLeak[:ps-1] = torch.zeros(ps-1,dtype=datatype)
        musW,musU = mus*torch.ones(ps,dtype=datatype),mus*torch.ones(pa,dtype=datatype)
        musW[0] = 1        
        W0T = omgs(anglesWNoDcLeak,musW).T.to(device)
        U0T = omgs(anglesU,musU).T.to(device)
        # dLdX = dZdX x dLdZ
        ms,ma = int(math.ceil(nDecs/2.)),int(math.floor(nDecs/2.))
        Ys = dLdZ[:,:,:,:ps].view(nSamples*nrows*ncols,ps).T # ps * n
        Ya = dLdZ[:,:,:,ps:].view(nSamples*nrows*ncols,pa).T # pa * n
        Y = torch.cat(
            ( W0T[:ms,:] @ Ys,          # ms x ps @ ps x n
              U0T[:ma,:] @ Ya ), dim=0) # ma x pa @ pa x n
        expctddLdX = Y.T.view(nSamples,nrows,ncols,nDecs).to(device) # n x (ms+ma)
        # dLdWi = <dLdZ,(dVdWi)X>
        expctddLdW_W = torch.zeros(nAnglesH,dtype=datatype).to(device)
        expctddLdW_U = torch.zeros(nAnglesH,dtype=datatype).to(device)
        omgs.partial_difference = True
        for iAngle in range(nAnglesH):
            dW0 = omgs(anglesWNoDcLeak,mus,index_pd_angle=iAngle).to(device)
            Xs = X[:,:,:,:ms].view(-1,ms).T 
            Zs = dW0[:,:ms] @ Xs # ps x n
            expctddLdW_W[iAngle] = torch.sum(Ys * Zs) # ps x n
            if ma>0:
                dU0 = omgs(anglesU,mus,index_pd_angle=iAngle).to(device)
                Xa = X[:,:,:,ms:].view(-1,ma).T
                Za = dU0[:,:ma] @ Xa # pa x n            
                expctddLdW_U[iAngle] = torch.sum(Ya * Za) # pa x n
            
        # Instantiation of target class
        layer = NsoltInitialRotation2dLayer(
            number_of_channels=nchs,
            decimation_factor=stride,
            no_dc_leakage=True,
            name='V0')
        layer.orthTransW0.angles.data = anglesW
        layer.orthTransW0.mus = musW
        layer.orthTransU0.angles.data = anglesU
        layer.orthTransU0.mus = musU
        layer = layer.to(device)

        # Actual values
        torch.autograd.set_detect_anomaly(True)
        Z = layer.forward(X)
        layer.zero_grad()
        Z.backward(dLdZ)
        actualdLdX = X.grad
        actualdLdW_W = layer.orthTransW0.angles.grad
        actualdLdW_U = layer.orthTransU0.angles.grad

        # Evaluation
        self.assertEqual(actualdLdX.dtype,datatype)
        self.assertEqual(actualdLdW_W.dtype,datatype)
        self.assertEqual(actualdLdW_U.dtype,datatype)
        self.assertTrue(torch.allclose(actualdLdX,expctddLdX,rtol=rtol,atol=atol))
        self.assertTrue(torch.allclose(actualdLdW_W,expctddLdW_W,rtol=rtol,atol=atol))
        self.assertTrue(torch.allclose(actualdLdW_U,expctddLdW_U,rtol=rtol,atol=atol))
        self.assertTrue(Z.requires_grad)

    @parameterized.expand(
        list(itertools.product(nchs,stride)) 
    )
    def testGradCheck(self,nchs,stride):
        datatype = torch.double
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")           

        # Configuration
        ps, pa = nchs
        nrows = 2
        ncols = 3
        nSamples = 2
        nDecs = stride[0]*stride[1] # math.prod(stride)
        nChsTotal = sum(nchs)
        nAnglesW = int((ps-1)*ps/2)
        anglesW = torch.randn(nAnglesW,dtype=datatype) 
        musW = (-1)**torch.randint(high=2,size=(ps,))                
        nAnglesU = int((pa-1)*pa/2)        
        anglesU = torch.randn(nAnglesU,dtype=datatype)        
        musU = (-1)**torch.randint(high=2,size=(pa,))        
        
        # nSamples x nRows x nCols x nDecs
        X = torch.randn(nSamples,nrows,ncols,nDecs,dtype=datatype,device=device,requires_grad=True)
        dLdZ = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype)
        dLdZ = dLdZ.to(device)

        # Instantiation of target class
        layer = NsoltInitialRotation2dLayer(
                number_of_channels=nchs,
                decimation_factor=stride,
                no_dc_leakage=False,
                name='V0'
            )
        layer.orthTransW0.angles.data = anglesW
        layer.orthTransW0.mus = musW
        layer.orthTransU0.angles.data = anglesU
        layer.orthTransU0.mus = musU
        layer = layer.to(device)

        # Forward
        torch.autograd.set_detect_anomaly(True)                
        Z = layer.forward(X)
        layer.zero_grad()

        # Evaluation        
        self.assertTrue(torch.autograd.gradcheck(layer,(X,)))

    @parameterized.expand(
        list(itertools.product(nchs,stride)) 
    )
    def testGradCheckNoDcLeakage(self,nchs,stride):
        datatype = torch.double
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")          

        # Configuration
        ps, pa = nchs
        nrows = 2
        ncols = 3
        nSamples = 2
        nDecs = stride[0]*stride[1] # math.prod(stride)
        nChsTotal = sum(nchs)
        nAnglesW = int((ps-1)*ps/2)
        anglesW = torch.randn(nAnglesW,dtype=datatype) 
        musW = (-1)**torch.randint(high=2,size=(ps,))                
        nAnglesU = int((pa-1)*pa/2)        
        anglesU = torch.randn(nAnglesU,dtype=datatype)        
        musU = (-1)**torch.randint(high=2,size=(pa,))        
        
        # nSamples x nRows x nCols x nDecs
        X = torch.randn(nSamples,nrows,ncols,nDecs,dtype=datatype,device=device,requires_grad=True)
        dLdZ = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype)
        dLdZ = dLdZ.to(device)

        # Instantiation of target class
        layer = NsoltInitialRotation2dLayer(
                number_of_channels=nchs,
                decimation_factor=stride,
                no_dc_leakage=True,
                name='V0'
            )
        layer.orthTransW0.angles.data = anglesW
        layer.orthTransW0.mus = musW
        layer.orthTransU0.angles.data = anglesU
        layer.orthTransU0.mus = musU
        layer = layer.to(device)

        # Forward
        torch.autograd.set_detect_anomaly(True)                
        Z = layer.forward(X)
        layer.zero_grad()

        # Evaluation        
        self.assertTrue(torch.autograd.gradcheck(layer,(X,)))
    
if __name__ == '__main__':
    unittest.main()
