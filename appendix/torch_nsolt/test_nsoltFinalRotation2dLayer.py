import itertools
import unittest
from parameterized import parameterized
import math
import torch
import torch.nn as nn
from nsoltFinalRotation2dLayer import NsoltFinalRotation2dLayer
from nsoltUtility import Direction, OrthonormalMatrixGenerationSystem

nchs = [ [2, 2], [3, 3], [4, 4] ]
stride = [ [1, 1], [1, 2], [2, 1], [2, 2] ]
mus = [ -1, 1 ]
datatype = [ torch.float, torch.double ]
nrows = [ 4, 8, 16 ]
ncols = [ 4, 8, 16 ]
isdevicetest = True

class NsoltFinalRotation2dLayerTestCase(unittest.TestCase):
    """
    NSOLTFINALROTATION2DLAYERTESTCASE 
    
       コンポーネント別に入力(nComponents):
          nSamples x nRows x nCols x nChs
    
       コンポーネント別に出力(nComponents):
          nSamples x nRows x nCols x nDecs
    
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
        
        # Expected values
        expctdName = 'V0~'
        expctdDescription = "NSOLT final rotation " \
                + "(ps,pa) = (" \
                + str(nchs[0]) + "," + str(nchs[1]) + "), " \
                + "(mv,mh) = (" \
                + str(stride[Direction.VERTICAL]) + "," + str(stride[Direction.HORIZONTAL]) + ")"

        # Instantiation of target class
        layer = NsoltFinalRotation2dLayer(
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
        # nSamples x nRows x nCols x nChs
        X = torch.randn(nSamples,nrows,ncols,sum(nchs),dtype=datatype,device=device,requires_grad=True)
        
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
        expctdZ = Zsa.T.view(nSamples,nrows,ncols,nDecs)

        # Instantiation of target class
        layer = NsoltFinalRotation2dLayer(
                number_of_channels=nchs,
                decimation_factor=stride,
                name='V0~')
        layer = layer.to(device)

        # Actual values
        with torch.no_grad():
            actualZ = layer.forward(X)

        # Evaluation
        self.assertEqual(actualZ.dtype,datatype)
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))
        self.assertFalse(actualZ.requires_grad)

    @parameterized.expand(
        list(itertools.product(datatype,nchs,stride,nrows,ncols))
    )
    def testPredictGrayscaleWithRandomAngles(self,
        datatype,nchs,stride,nrows,ncols):
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
        # nSamples x nRows x nCols x nChs
        X = torch.randn(nSamples,nrows,ncols,nChsTotal,device=device,dtype=datatype)
        angles = torch.randn(int((nChsTotal-2)*nChsTotal/4),dtype=datatype)
    
        # Expected values
        # nSamples x nRows x nCols x nDecs
        ps,pa = nchs
        nAngsW = int(len(angles)/2) 
        angsW,angsU = angles[:nAngsW],angles[nAngsW:]
        W0T,U0T = gen(angsW).T.to(device),gen(angsU).T.to(device)
        Ys = X[:,:,:,:ps].view(-1,ps).T
        Ya = X[:,:,:,ps:].view(-1,pa).T
        ms,ma = int(math.ceil(nDecs/2.)),int(math.floor(nDecs/2.))
        Zsa = torch.cat(
                ( W0T[:ms,:] @ Ys, 
                  U0T[:ma,:] @ Ya ),dim=0)
        expctdZ = Zsa.T.view(nSamples,nrows,ncols,nDecs)

        # Instantiation of target class
        layer = NsoltFinalRotation2dLayer(
                number_of_channels=nchs,
                decimation_factor=stride,
                name='V0~')
        layer.orthTransW0T.angles.data = angsW
        layer.orthTransW0T.mus = 1
        layer.orthTransU0T.angles.data = angsU
        layer.orthTransU0T.mus = 1
        layer = layer.to(device)

        # Actual values
        with torch.no_grad():
            actualZ = layer.forward(X)

        # Evaluation
        self.assertEqual(actualZ.dtype,datatype)
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))
        self.assertFalse(actualZ.requires_grad)

    @parameterized.expand(
        list(itertools.product(datatype,nchs,stride,nrows,ncols,mus))
    )
    def testPredictGrayscaleWithRandomAnglesNoDcLeackage(self,
        datatype,nchs,stride,nrows,ncols,mus):
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
        # nSamples x nRows x nCols x nChs
        X = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype)
        X = X.to(device)
        angles = torch.randn(int((nChsTotal-2)*nChsTotal/4),dtype=datatype)

        # Expected values
        ps, pa = nchs
        nAngsW = int(len(angles)/2) 
        angsW,angsU = angles[:nAngsW],angles[nAngsW:]
        angsWNoDcLeak = angsW.clone()
        angsWNoDcLeak[:ps-1] = torch.zeros(ps-1,dtype=angles.dtype)
        musW,musU = mus*torch.ones(ps,dtype=datatype),mus*torch.ones(pa,dtype=datatype)
        musW[0] = 1
        W0T,U0T = gen(angsWNoDcLeak,musW).T.to(device),gen(angsU,musU).T.to(device)
        Ys = X[:,:,:,:ps].view(-1,ps).T
        Ya = X[:,:,:,ps:].view(-1,pa).T
        Zsa = torch.cat(
                ( W0T[:int(math.ceil(nDecs/2.)),:] @ Ys, 
                  U0T[:int(math.floor(nDecs/2.)),:] @ Ya ),dim=0)
        expctdZ = Zsa.T.view(nSamples,nrows,ncols,nDecs)

        # Instantiation of target class
        layer = NsoltFinalRotation2dLayer(
                number_of_channels=nchs,
                decimation_factor=stride,
                no_dc_leakage=True,
                name='V0~')
        layer.orthTransW0T.angles.data = angsW
        layer.orthTransW0T.mus = musW
        layer.orthTransU0T.angles.data = angsU
        layer.orthTransU0T.mus = musU
        layer = layer.to(device)

        # Actual values
        with torch.no_grad():
            actualZ = layer.forward(X)

        # Evaluation
        self.assertEqual(actualZ.dtype,datatype)
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))
        self.assertFalse(actualZ.requires_grad)

    @parameterized.expand(
        list(itertools.product(datatype,nchs,stride,nrows,ncols)) 
    )
    def testBackwardGrayscale(self,
        datatype,nchs,stride,nrows,ncols): 
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
        anglesW = torch.zeros(nAnglesH,dtype=datatype) 
        anglesU = torch.zeros(nAnglesH,dtype=datatype)        
        mus = 1
        # nSamples x nRows x nCols x nChs
        X = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype,device=device,requires_grad=True)
        dLdZ = torch.randn(nSamples,nrows,ncols,nDecs,dtype=datatype)
        dLdZ = dLdZ.to(device)

        # Expected values
        ps,pa = nchs
        W0 = omgs(anglesW,mus).to(device)
        U0 = omgs(anglesU,mus).to(device)
        # dLdX = dZdX x dLdZ        
        ms,ma = int(math.ceil(nDecs/2.)),int(math.floor(nDecs/2.))
        Ys = dLdZ[:,:,:,:ms].view(nSamples*nrows*ncols,ms).T # ms x n
        Ya = dLdZ[:,:,:,ms:].view(nSamples*nrows*ncols,ma).T # ma x n
        Y = torch.cat(
                ( W0[:,:ms] @ Ys,           # ps x ms @ ms x n
                  U0[:,:ma] @ Ya ),dim=0)   # pa x ma @ ma x n
        expctddLdX = Y.T.view(nSamples,nrows,ncols,nChsTotal).to(device) # n x (ps+pa) -> N x R x C X P
        # dLdWi = <dLdZ,(dVdWi)X>   
        expctddLdW_W = torch.zeros(nAnglesH,dtype=datatype).to(device)
        expctddLdW_U = torch.zeros(nAnglesH,dtype=datatype).to(device)
        omgs.partial_difference = True
        for iAngle in range(nAnglesH):
            dW0_T = omgs(anglesW,mus,index_pd_angle=iAngle).T.to(device)
            dU0_T = omgs(anglesU,mus,index_pd_angle=iAngle).T.to(device)
            Xs = X[:,:,:,:ps].view(-1,ps).T 
            Xa = X[:,:,:,ps:].view(-1,pa).T
            Zs = dW0_T[:ms,:] @ Xs # ms x n
            Za = dU0_T[:ma,:] @ Xa # ma x n 
            expctddLdW_W[iAngle] = torch.sum(Ys[:ms,:] * Zs)
            expctddLdW_U[iAngle] = torch.sum(Ya[:ma,:] * Za)

        # Instantiation of target class
        layer = NsoltFinalRotation2dLayer(
                number_of_channels=nchs,
                decimation_factor=stride,
                name='V0~')
        layer.orthTransW0T.angles.data = anglesW
        layer.orthTransW0T.mus = mus
        layer.orthTransU0T.angles.data = anglesU
        layer.orthTransU0T.mus = mus
        layer = layer.to(device)

        # Actual values
        torch.autograd.set_detect_anomaly(True)        
        Z = layer.forward(X)
        layer.zero_grad()        
        Z.backward(dLdZ)
        actualdLdX = X.grad
        actualdLdW_W = layer.orthTransW0T.angles.grad
        actualdLdW_U = layer.orthTransU0T.angles.grad
    
        # Evaluation
        self.assertEqual(actualdLdX.dtype,datatype)
        self.assertEqual(actualdLdW_W.dtype,datatype)
        self.assertEqual(actualdLdW_U.dtype,datatype)
        self.assertTrue(torch.allclose(actualdLdX,expctddLdX,rtol=rtol,atol=atol))
        self.assertTrue(torch.allclose(actualdLdW_W,expctddLdW_W,rtol=rtol,atol=atol))
        self.assertTrue(torch.allclose(actualdLdW_U,expctddLdW_U,rtol=rtol,atol=atol))
        self.assertTrue(Z.requires_grad)

    @parameterized.expand(
        list(itertools.product(datatype,nchs,stride,nrows,ncols,mus)) 
    )
    def testBackwardGayscaleWithRandomAngles(self,
        datatype,nchs,stride,nrows,ncols,mus): 
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
        # nSamples x nRows x nCols x nChs
        X = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype,device=device,requires_grad=True)
        dLdZ = torch.randn(nSamples,nrows,ncols,nDecs,dtype=datatype)
        dLdZ = dLdZ.to(device)

        # Expected values
        ps,pa = nchs
        W0 = omgs(anglesW,mus).to(device)
        U0 = omgs(anglesU,mus).to(device)
        # dLdX = dZdX x dLdZ        
        ms,ma = int(math.ceil(nDecs/2.)),int(math.floor(nDecs/2.))
        Ys = dLdZ[:,:,:,:ms].view(nSamples*nrows*ncols,ms).T # ms x n
        Ya = dLdZ[:,:,:,ms:].view(nSamples*nrows*ncols,ma).T # ma x n
        Y = torch.cat(
                ( W0[:,:ms] @ Ys,           # ps x ms @ ms x n
                  U0[:,:ma] @ Ya ),dim=0)   # pa x ma @ ma x n
        expctddLdX = Y.T.view(nSamples,nrows,ncols,nChsTotal) # n x (ps+pa) -> N x R x C X P
        # dLdWi = <dLdZ,(dVdWi)X>   
        expctddLdW_W = torch.zeros(nAnglesH,dtype=datatype).to(device)
        expctddLdW_U = torch.zeros(nAnglesH,dtype=datatype).to(device)
        omgs.partial_difference = True
        for iAngle in range(nAnglesH):
            dW0_T = omgs(anglesW,mus,index_pd_angle=iAngle).T.to(device)
            dU0_T = omgs(anglesU,mus,index_pd_angle=iAngle).T.to(device)
            Xs = X[:,:,:,:ps].view(-1,ps).T 
            Xa = X[:,:,:,ps:].view(-1,pa).T
            Zs = dW0_T[:ms,:] @ Xs # ms x n
            Za = dU0_T[:ma,:] @ Xa # ma x n 
            expctddLdW_W[iAngle] = torch.sum(Ys[:ms,:] * Zs)
            expctddLdW_U[iAngle] = torch.sum(Ya[:ma,:] * Za)

        # Instantiation of target class
        layer = NsoltFinalRotation2dLayer(
                number_of_channels=nchs,
                decimation_factor=stride,
                name='V0~')
        layer.orthTransW0T.angles.data = anglesW
        layer.orthTransW0T.mus = mus
        layer.orthTransU0T.angles.data = anglesU
        layer.orthTransU0T.mus = mus
        layer = layer.to(device)

        # Actual values
        torch.autograd.set_detect_anomaly(True)        
        Z = layer.forward(X)
        layer.zero_grad()        
        Z.backward(dLdZ)
        actualdLdX = X.grad
        actualdLdW_W = layer.orthTransW0T.angles.grad
        actualdLdW_U = layer.orthTransU0T.angles.grad
    
        # Evaluation
        self.assertEqual(actualdLdX.dtype,datatype)
        self.assertEqual(actualdLdW_W.dtype,datatype)
        self.assertEqual(actualdLdW_U.dtype,datatype)
        self.assertTrue(torch.allclose(actualdLdX,expctddLdX,rtol=rtol,atol=atol))
        self.assertTrue(torch.allclose(actualdLdW_W,expctddLdW_W,rtol=rtol,atol=atol))
        self.assertTrue(torch.allclose(actualdLdW_U,expctddLdW_U,rtol=rtol,atol=atol))
        self.assertTrue(Z.requires_grad)

    @parameterized.expand(
        list(itertools.product(datatype,nchs,stride,nrows,ncols,mus)) 
    )
    def testBackwardWithRandomAnglesNoDcLeackage(self,
        datatype,nchs,stride,nrows,ncols,mus): 
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
        # nSamples x nRows x nCols x nChs
        X = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype,device=device,requires_grad=True)
        dLdZ = torch.randn(nSamples,nrows,ncols,nDecs,dtype=datatype)
        dLdZ = dLdZ.to(device)

        # Expected values
        ps, pa = nchs
        anglesWNoDcLeak = anglesW.clone()
        anglesWNoDcLeak[:ps-1] = torch.zeros(ps-1,dtype=datatype)
        musW,musU = mus*torch.ones(ps,dtype=datatype),mus*torch.ones(pa,dtype=datatype)
        musW[0] = 1
        W0 = omgs(anglesWNoDcLeak,musW).to(device)
        U0 = omgs(anglesU,musU).to(device)
        # dLdX = dZdX x dLdZ        
        ms,ma = int(math.ceil(nDecs/2.)),int(math.floor(nDecs/2.))
        Ys = dLdZ[:,:,:,:ms].view(nSamples*nrows*ncols,ms).T # ms x n
        Ya = dLdZ[:,:,:,ms:].view(nSamples*nrows*ncols,ma).T # ma x n
        Y = torch.cat(
                ( W0[:,:ms] @ Ys,           # ps x ms @ ms x n
                  U0[:,:ma] @ Ya ),dim=0)   # pa x ma @ ma x n
        expctddLdX = Y.T.view(nSamples,nrows,ncols,nChsTotal) # n x (ps+pa) -> N x R x C X P
        # dLdWi = <dLdZ,(dVdWi)X>   
        expctddLdW_W = torch.zeros(nAnglesH,dtype=datatype).to(device)
        expctddLdW_U = torch.zeros(nAnglesH,dtype=datatype).to(device)
        omgs.partial_difference = True
        for iAngle in range(nAnglesH):
            dW0_T = omgs(anglesWNoDcLeak,musW,index_pd_angle=iAngle).T.to(device)
            dU0_T = omgs(anglesU,musU,index_pd_angle=iAngle).T.to(device)
            Xs = X[:,:,:,:ps].view(-1,ps).T 
            Xa = X[:,:,:,ps:].view(-1,pa).T
            Zs = dW0_T[:ms,:] @ Xs # ms x n
            Za = dU0_T[:ma,:] @ Xa # ma x n 
            expctddLdW_W[iAngle] = torch.sum(Ys[:ms,:] * Zs)
            expctddLdW_U[iAngle] = torch.sum(Ya[:ma,:] * Za)

        # Instantiation of target class
        layer = NsoltFinalRotation2dLayer(
                number_of_channels=nchs,
                decimation_factor=stride,
                no_dc_leakage=True,
                name='V0~'
            )
        layer.orthTransW0T.angles.data = anglesW
        layer.orthTransW0T.mus = mus
        layer.orthTransU0T.angles.data = anglesU
        layer.orthTransU0T.mus = mus
        layer = layer.to(device)

        # Actual values
        torch.autograd.set_detect_anomaly(True)        
        Z = layer.forward(X)
        layer.zero_grad()
        Z.backward(dLdZ)
        actualdLdX = X.grad
        actualdLdW_W = layer.orthTransW0T.angles.grad
        actualdLdW_U = layer.orthTransU0T.angles.grad
    
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
        
        # nSamples x nRows x nCols x nChs
        X = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype,device=device,requires_grad=True)
        dLdZ = torch.randn(nSamples,nrows,ncols,nDecs,dtype=datatype)
        dLdZ = dLdZ.to(device)

        # Instantiation of target class
        layer = NsoltFinalRotation2dLayer(
                number_of_channels=nchs,
                decimation_factor=stride,
                no_dc_leakage=False,
                name='V0~'
            )
        layer.orthTransW0T.angles.data = anglesW
        layer.orthTransW0T.mus = musW
        layer.orthTransU0T.angles.data = anglesU
        layer.orthTransU0T.mus = musU
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
        
        # nSamples x nRows x nCols x nChs
        X = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype,device=device,requires_grad=True)
        dLdZ = torch.randn(nSamples,nrows,ncols,nDecs,dtype=datatype)
        dLdZ = dLdZ.to(device)

        # Instantiation of target class
        layer = NsoltFinalRotation2dLayer(
                number_of_channels=nchs,
                decimation_factor=stride,
                no_dc_leakage=True,
                name='V0~'
            )
        layer.orthTransW0T.angles.data = anglesW
        layer.orthTransW0T.mus = musW
        layer.orthTransU0T.angles.data = anglesU
        layer.orthTransU0T.mus = musU
        layer = layer.to(device)

        # Forward
        torch.autograd.set_detect_anomaly(True)                
        Z = layer.forward(X)
        layer.zero_grad()

        # Evaluation        
        self.assertTrue(torch.autograd.gradcheck(layer,(X,)))
    
if __name__ == '__main__':
    unittest.main()
