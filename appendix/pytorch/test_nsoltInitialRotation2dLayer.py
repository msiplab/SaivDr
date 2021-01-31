import itertools
import unittest
from parameterized import parameterized
import math
import torch
import torch.nn as nn
from nsoltInitialRotation2dLayer import NsoltInitialRotation2dLayer
from nsoltUtility import Direction, OrthonormalMatrixGenerationSystem

nchs = [ [2, 2], [3, 3], [4, 4] ]
stride = [ [1, 1], [1, 2], [2, 2] ]
mus = [ -1, 1 ]
datatype = [ torch.float, torch.double ]
nrows = [ 4, 8, 16 ]
ncols = [ 4, 8, 16 ]

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
        
        # Parameters
        nSamples = 8
        nDecs = stride[0]*stride[1] # math.prod(stride)
        nChsTotal = sum(nchs)
        # nSamples x nRows x nCols x nDecs
        X = torch.randn(nSamples,nrows,ncols,nDecs,dtype=datatype)

        # Expected values
        # nSamplex x nRows x nCols x nChs
        ps, pa = nchs
        W0 = torch.eye(ps,dtype=datatype)
        U0 = torch.eye(pa,dtype=datatype)
        ms,ma = int(math.ceil(nDecs/2.)), int(math.floor(nDecs/2.))        
        Zsa = torch.zeros(nChsTotal,nrows*ncols*nSamples,dtype=datatype)        
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
        rtol,atol=1e-5,1e-8
        gen = OrthonormalMatrixGenerationSystem(dtype=datatype)
        
        # Parameters
        nSamples = 8
        nDecs = stride[0]*stride[1] # math.prod(stride)
        nChsTotal = sum(nchs)
        # nSamples x nRows x nCols x nDecs
        X = torch.randn(nSamples,nrows,ncols,nDecs,dtype=datatype)
        angles = torch.randn(int((nChsTotal-2)*nChsTotal/4),dtype=datatype)

        # Expected values
        # nSamples x nRows x nCols x nChs
        ps,pa = nchs
        nAngsW = int(len(angles)/2)
        angsW,angsU = angles[:nAngsW],angles[nAngsW:]
        W0,U0 = gen(angsW),gen(angsU)
        ms,ma = int(math.ceil(nDecs/2.)), int(math.floor(nDecs/2.))                
        Zsa = torch.zeros(nChsTotal,nrows*ncols*nSamples,dtype=datatype)
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
        rtol,atol=1e-5,1e-8
        gen = OrthonormalMatrixGenerationSystem(dtype=datatype)
        
        # Parameters
        nSamples = 8
        nDecs = stride[0]*stride[1] # math.prod(stride)
        nChsTotal = sum(nchs)
        # nSamples x nRows x nCols x nDecs
        X = torch.randn(nSamples,nrows,ncols,nDecs,dtype=datatype)
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
        W0,U0 = gen(angsWNoDcLeak,musW),gen(angsU,musU)
        ms,ma = int(math.ceil(nDecs/2.)), int(math.floor(nDecs/2.))                
        Zsa = torch.zeros(nChsTotal,nrows*ncols*nSamples,dtype=datatype)
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
        X = torch.randn(nSamples,nrows,ncols,nDecs,dtype=datatype,requires_grad=True)
        dLdZ = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype)

        # Expected values
        # nSamples x nRows x nCols x nDecs
        ps,pa = nchs
        W0T = omgs(anglesW,mus).T
        U0T = omgs(anglesU,mus).T
        # dLdX = dZdX x dLdZ
        ms,ma = int(math.ceil(nDecs/2.)),int(math.floor(nDecs/2.))
        Ys = dLdZ[:,:,:,:ps].view(nSamples*nrows*ncols,ps).T # ps * n
        Ya = dLdZ[:,:,:,ps:].view(nSamples*nrows*ncols,pa).T # pa * n
        Y = torch.cat(
            ( W0T[:ms,:] @ Ys,          # ms x ps @ ps x n
              U0T[:ma,:] @ Ya ), dim=0) # ma x pa @ pa x n
        expctddLdX = Y.T.view(nSamples,nrows,ncols,nDecs) # n x (ms+ma)
        # dLdWi = <dLdZ,(dVdWi)X>
        expctddLdW_W = torch.zeros(nAnglesH,dtype=datatype)
        expctddLdW_U = torch.zeros(nAnglesH,dtype=datatype)
        omgs.partial_difference = True
        for iAngle in range(nAnglesH):
            dW0 = omgs(anglesW,mus,index_pd_angle=iAngle)
            Xs = X[:,:,:,:ms].view(-1,ms).T 
            Zs = dW0[:,:ms] @ Xs # ps x n
            expctddLdW_W[iAngle] = torch.sum(Ys * Zs) # ps x n
            if ma>0:
                dU0 = omgs(anglesU,mus,index_pd_angle=iAngle)
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

    """        
        function testBackwardGrayscaleWithRandomAngles(testCase, ...
                nchs, stride, nrows, ncols, datatype)
            
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-4,single(1e-4));
            import saivdr.dictionary.utility.*
            genW = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on');
            genU = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on');
            
            % Parameters
            nSamples = 8;
            nDecs = prod(stride);
            nChsTotal = sum(nchs);
            nAnglesH = (nChsTotal-2)*nChsTotal/8;
            anglesW = randn(nAnglesH,1,datatype);
            anglesU = randn(nAnglesH,1,datatype);
            mus_ = 1;
            
            % nDecs x nRows x nCols x nSamples
            %X = randn(nrows,ncols,nDecs,nSamples,datatype);
            %dLdZ = randn(nrows,ncols,sum(nchs),nSamples,datatype);
            X = randn(nDecs,nrows,ncols,nSamples,datatype);
            dLdZ = randn(sum(nchs),nrows,ncols,nSamples,datatype);
            
            % Expected values
            % nDecs x nRows x nCols x nSamples
            ps = nchs(1);
            pa = nchs(2);
            
            % dLdX = dZdX x dLdZ
            W0T = transpose(genW.step(anglesW,mus_,0));
            U0T = transpose(genU.step(anglesU,mus_,0));
            Y = dLdZ; %permute(dLdZ,[3 1 2 4]);
            Ys = reshape(Y(1:ps,:,:,:),ps,nrows*ncols*nSamples);
            Ya = reshape(Y(ps+1:ps+pa,:,:,:),pa,nrows*ncols*nSamples);
            Zsa = [ W0T(1:ceil(nDecs/2),:)*Ys; U0T(1:floor(nDecs/2),:)*Ya ];
            %expctddLdX = ipermute(reshape(Zsa,nDecs,nrows,ncols,nSamples),...
            %    [3 1 2 4]);
            expctddLdX = reshape(Zsa,nDecs,nrows,ncols,nSamples);
                        
            % dLdWi = <dLdZ,(dVdWi)X>
            expctddLdW = zeros(2*nAnglesH,1,datatype);
            dldz_ = dLdZ; %permute(dLdZ,[3 1 2 4]);
            dldz_upp = reshape(dldz_(1:ps,:,:,:),ps,nrows*ncols*nSamples);
            dldz_low = reshape(dldz_(ps+1:ps+pa,:,:,:),pa,nrows*ncols*nSamples);
            % (dVdWi)X
            for iAngle = 1:nAnglesH
                dW0 = genW.step(anglesW,mus_,iAngle);
                dU0 = genU.step(anglesU,mus_,iAngle);
                a_ = X; %permute(X,[3 1 2 4]);
                c_upp = reshape(a_(1:ceil(nDecs/2),:,:,:),ceil(nDecs/2),nrows*ncols*nSamples);
                c_low = reshape(a_(ceil(nDecs/2)+1:nDecs,:,:,:),floor(nDecs/2),nrows*ncols*nSamples);
                d_upp = dW0(:,1:ceil(nDecs/2))*c_upp;
                d_low = dU0(:,1:floor(nDecs/2))*c_low;
                expctddLdW(iAngle) = sum(dldz_upp.*d_upp,'all');
                expctddLdW(nAnglesH+iAngle) = sum(dldz_low.*d_low,'all');
            end
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltInitialRotation2dLayer(...
                'NumberOfChannels',nchs,...
                'DecimationFactor',stride,...
                'Name','V0');
            layer.Mus = mus_;
            layer.Angles = [anglesW; anglesU];
            %expctdZ = layer.predict(X);
            
            % Actual values
            [actualdLdX,actualdLdW] = layer.backward(X,[],dLdZ,[]);
            
            % Evaluation
            testCase.verifyInstanceOf(actualdLdX,datatype);
            testCase.verifyInstanceOf(actualdLdW,datatype);
            testCase.verifyThat(actualdLdX,...
                IsEqualTo(expctddLdX,'Within',tolObj));
            testCase.verifyThat(actualdLdW,...
                IsEqualTo(expctddLdW,'Within',tolObj));
            
        end
        
        function testBackwardGrayscaleWithRandomAnglesNoDcLeackage(testCase, ...
                nchs, stride, nrows, ncols, mus, datatype)
            
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-4,single(1e-4));
            import saivdr.dictionary.utility.*
            genW = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on');
            genU = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on');
            
            % Parameters
            nSamples = 8;
            nDecs = prod(stride);
            nChsTotal = sum(nchs);
            nAnglesH = (nChsTotal-2)*nChsTotal/8;
            anglesW = randn(nAnglesH,1,datatype);
            anglesU = randn(nAnglesH,1,datatype);
            
            % nDecs x nRows x nCols x nSamples
            %X = randn(nrows,ncols,nDecs,nSamples,datatype);
            %dLdZ = randn(nrows,ncols,sum(nchs),nSamples,datatype);
            X = randn(nDecs,nrows,ncols,nSamples,datatype);
            dLdZ = randn(sum(nchs),nrows,ncols,nSamples,datatype);            
            
            % Expected values
            % nDecs x nRows x nCols x nSamples
            ps = nchs(1);
            pa = nchs(2);
            
            % dLdX = dZdX x dLdZ
            anglesW_NoDc = anglesW;
            anglesW_NoDc(1:ps-1,1)=zeros(ps-1,1);
            musW = mus*ones(ps,1);
            musW(1,1) = 1;
            musU = mus*ones(pa,1);            
            W0T = transpose(genW.step(anglesW_NoDc,musW,0));
            U0T = transpose(genU.step(anglesU,musU,0));
            Y = dLdZ; %permute(dLdZ,[3 1 2 4]);
            Ys = reshape(Y(1:ps,:,:,:),ps,nrows*ncols*nSamples);
            Ya = reshape(Y(ps+1:ps+pa,:,:,:),pa,nrows*ncols*nSamples);
            Zsa = [ W0T(1:ceil(nDecs/2),:)*Ys; U0T(1:floor(nDecs/2),:)*Ya ];
            %expctddLdX = ipermute(reshape(Zsa,nDecs,nrows,ncols,nSamples),...
            %    [3 1 2 4]);
            expctddLdX = reshape(Zsa,nDecs,nrows,ncols,nSamples);
            
            % dLdWi = <dLdZ,(dVdWi)X>
            expctddLdW = zeros(2*nAnglesH,1,datatype);
            dldz_ = dLdZ; %permute(dLdZ,[3 1 2 4]);
            dldz_upp = reshape(dldz_(1:ps,:,:,:),ps,nrows*ncols*nSamples);
            dldz_low = reshape(dldz_(ps+1:ps+pa,:,:,:),pa,nrows*ncols*nSamples);
            % (dVdWi)X
            for iAngle = 1:nAnglesH
                dW0 = genW.step(anglesW_NoDc,musW,iAngle);
                dU0 = genU.step(anglesU,musU,iAngle);
                a_ = X; %permute(X,[3 1 2 4]);
                c_upp = reshape(a_(1:ceil(nDecs/2),:,:,:),ceil(nDecs/2),nrows*ncols*nSamples);
                c_low = reshape(a_(ceil(nDecs/2)+1:nDecs,:,:,:),floor(nDecs/2),nrows*ncols*nSamples);
                d_upp = dW0(:,1:ceil(nDecs/2))*c_upp;
                d_low = dU0(:,1:floor(nDecs/2))*c_low;
                expctddLdW(iAngle) = sum(dldz_upp.*d_upp,'all');
                expctddLdW(nAnglesH+iAngle) = sum(dldz_low.*d_low,'all');
            end
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltInitialRotation2dLayer(...
                'NumberOfChannels',nchs,...
                'DecimationFactor',stride,...
                'NoDcLeakage',true,...
                'Name','V0');
            layer.Mus = mus;
            layer.Angles = [anglesW; anglesU];
            %expctdZ = layer.predict(X);
            
            % Actual values
            [actualdLdX,actualdLdW] = layer.backward(X,[],dLdZ,[]);
            
            % Evaluation
            testCase.verifyInstanceOf(actualdLdX,datatype);
            testCase.verifyInstanceOf(actualdLdW,datatype);
            testCase.verifyThat(actualdLdX,...
                IsEqualTo(expctddLdX,'Within',tolObj));
            testCase.verifyThat(actualdLdW,...
                IsEqualTo(expctddLdW,'Within',tolObj));
            
        end
    """

if __name__ == '__main__':
    unittest.main()
