import itertools
import unittest
from parameterized import parameterized
import math
import torch
import torch.nn as nn
from nsoltIntermediateRotation2dLayer import NsoltIntermediateRotation2dLayer
from nsoltUtility import Direction, OrthonormalMatrixGenerationSystem

nchs = [ [2, 2], [3, 3], [4, 4] ]
stride = [ [1, 1], [1, 2], [2, 2] ]
mus = [ -1, 1 ]
datatype = [ torch.float, torch.double ]
nrows = [ 4, 8, 16 ]
ncols = [ 4, 8, 16 ]

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
        list(itertools.product(nchs,stride))
    )
    def testConstructor(self,
        nchs, stride):

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
        list(itertools.product(nchs,stride,nrows,ncols,mus,datatype))
    )
    def testPredictGrayscale(self,
        nchs, stride, nrows, ncols, mus, datatype):
        rtol,atol=1e-5,1e-8

        # Parameters
        nSamples = 8
        nChsTotal = sum(nchs)
        # nSamples x nRows x nCols x nChsTotal
        X = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype,requires_grad=True)

        # Expected values
        # nSamples x nRows x nCols x nChsTotal
        ps,pa = nchs
        UnT = mus*torch.eye(pa,dtype=datatype)
        expctdZ = X.clone()
        Ya = X[:,:,:,ps:].view(-1,pa).T
        Za = UnT @ Ya
        expctdZ[:,:,:,ps:] = Za.T.view(nSamples,nrows,ncols,pa)

        # Instantiation of target class
        layer = NsoltIntermediateRotation2dLayer(
            number_of_channels=nchs,
            name='Vn~')
        layer.orthTransUn.mus = mus

        # Actual values
        with torch.no_grad():
            actualZ = layer.forward(X)

        # Evaluation
        self.assertEqual(actualZ.dtype,datatype)
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))
        self.assertFalse(actualZ.requires_grad)

    @parameterized.expand(
        list(itertools.product(nchs,stride,nrows,ncols,mus,datatype))
    )
    def testPredictGrayscaleWithRandomAngles(self,
        nchs, stride, nrows, ncols, mus, datatype):
        rtol,atol=1e-5,1e-8
        gen = OrthonormalMatrixGenerationSystem(dtype=datatype)

        # Parameters
        nSamples = 8
        nChsTotal = sum(nchs)
        # nSamples x nRows x nCols x nChsTotal
        X = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype,requires_grad=True)
        angles = torch.randn(int((nChsTotal-2)*nChsTotal/8),dtype=datatype)

        # Expected values
        # nSamples x nRows x nCols x nChsTotal
        ps,pa = nchs
        UnT = gen(angles,mus).T
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

        # Actual values
        with torch.no_grad():
            actualZ = layer.forward(X)

        # Evaluation
        self.assertEqual(actualZ.dtype,datatype)
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))
        self.assertFalse(actualZ.requires_grad)

    @parameterized.expand(
        list(itertools.product(nchs,stride,nrows,ncols,mus,datatype))
    )
    def testPredictGrayscaleAnalysisMode(self,
        nchs, stride, nrows, ncols, mus, datatype):
        rtol,atol=1e-5,1e-8
        gen = OrthonormalMatrixGenerationSystem(dtype=datatype)

        # Parameters
        nSamples = 8
        nChsTotal = sum(nchs)
        # nSamples x nRows x nCols x nChsTotal
        X = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype,requires_grad=True)
        angles = torch.randn(int((nChsTotal-2)*nChsTotal/8),dtype=datatype)

        # Expected values
        # nSamples x nRows x nCols x nChsTotal
        ps,pa = nchs
        Un = gen(angles,mus)
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
        omgs = OrthonormalMatrixGenerationSystem(dtype=datatype,partial_difference=False)

        # Parameters
        nSamples = 8
        nChsTotal = sum(nchs)
        nAngles = int((nChsTotal-2)*nChsTotal/8)
        angles = torch.zeros(nAngles,dtype=datatype)
        # nSamples x nRows x nCols x nChsTotal
        X = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype,requires_grad=True)
        dLdZ = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype)            

        # Expected values
        ps,pa = nchs
        Un = omgs(angles,mus)
        # dLdX = dZdX x dLdZ
        expctddLdX = dLdZ.clone()
        Ya = dLdZ[:,:,:,ps:].view(nSamples*nrows*ncols,pa).T # pa * n
        Za = Un @ Ya
        expctddLdX[:,:,:,ps:] = Za.T.view(nSamples,nrows,ncols,pa)
        # dLdWi = <dLdZ,(dVdWi)X>
        expctddLdW_U = torch.zeros(nAngles,dtype=datatype)        
        omgs.partial_difference = True
        for iAngle in range(nAngles):
            dUn_T = omgs(angles,mus,index_pd_angle=iAngle).T
            Xa = X[:,:,:,ps:].view(-1,pa).T
            Za = dUn_T @ Xa # pa x n 
            expctddLdW_U[iAngle] = torch.sum(Ya * Za)

        # Instantiation of target class
        layer = NsoltIntermediateRotation2dLayer(
            number_of_channels=nchs,
            name='Vn~')
        layer.orthTransUn.angles.data = angles
        layer.orthTransUn.mus = mus

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

    """
        
    function testBackwardGrayscaleWithRandomAngles(testCase, ...
                nchs, nrows, ncols, mus, datatype)
    
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-4,single(1e-4));
            import saivdr.dictionary.utility.*
            genU = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on');
            
            % Parameters
            nSamples = 8;
            nChsTotal = sum(nchs);
            nAngles = (nChsTotal-2)*nChsTotal/8;
            angles = randn((nChsTotal-2)*nChsTotal/8,1);
                 
            % nChsTotal x nRows x nCols x nSamples
            %X = randn(nrows,ncols,nChsTotal,nSamples,datatype);            
            %dLdZ = randn(nrows,ncols,nChsTotal,nSamples,datatype);
            X = randn(nChsTotal,nrows,ncols,nSamples,datatype);            
            dLdZ = randn(nChsTotal,nrows,ncols,nSamples,datatype);            

            % Expected values
            % nChsTotal x nRows x nCols x nSamples
            ps = nchs(1);
            pa = nchs(2);
            
            % dLdX = dZdX x dLdZ
            Un = genU.step(angles,mus,0);
            adLd_ = dLdZ; % permute(dLdZ,[3 1 2 4]);
            cdLd_low = reshape(adLd_(ps+1:ps+pa,:,:,:),pa,nrows*ncols*nSamples);
            cdLd_low = Un*cdLd_low;
            adLd_(ps+1:ps+pa,:,:,:) = reshape(cdLd_low,pa,nrows,ncols,nSamples);
            expctddLdX = adLd_; %ipermute(adLd_,[3 1 2 4]);           
            
            % dLdWi = <dLdZ,(dVdWi)X>
            expctddLdW = zeros(nAngles,1,datatype);
            for iAngle = 1:nAngles
                dUn_T = transpose(genU.step(angles,mus,iAngle));
                a_ = X; %permute(X,[3 1 2 4]);
                c_low = reshape(a_(ps+1:ps+pa,:,:,:),pa,nrows*ncols*nSamples);
                c_low = dUn_T*c_low;
                a_ = zeros(size(a_),datatype);
                a_(ps+1:ps+pa,:,:,:) = reshape(c_low,pa,nrows,ncols,nSamples);
                dVdW_X = a_; %ipermute(a_,[3 1 2 4]);
                %
                expctddLdW(iAngle) = sum(dLdZ.*dVdW_X,'all');
            end
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltIntermediateRotation2dLayer(...
                'NumberOfChannels',nchs,...
                'Name','Vn~');
            layer.Mus = mus;
            layer.Angles = angles;            
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
        
        function testBackwardGrayscaleAnalysisMode(testCase, ...
                nchs, nrows, ncols, mus, datatype)
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-4,single(1e-4));
            import saivdr.dictionary.utility.*
            genU = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on');
            
            % Parameters
            nSamples = 8;
            nChsTotal = sum(nchs);
            nAngles = (nChsTotal-2)*nChsTotal/8;
            angles = randn((nChsTotal-2)*nChsTotal/8,1);
            
            % nChsTotal x nRows x nCols xnSamples
            %X = randn(nrows,ncols,nChsTotal,nSamples,datatype);
            %dLdZ = randn(nrows,ncols,nChsTotal,nSamples,datatype);
            X = randn(nChsTotal,nrows,ncols,nSamples,datatype);
            dLdZ = randn(nChsTotal,nrows,ncols,nSamples,datatype);            
            
            % Expected values
            % nChsTotal x nRows x nCols x nSamples
            ps = nchs(1);
            pa = nchs(2);
            
            % dLdX = dZdX x dLdZ
            UnT = transpose(genU.step(angles,mus,0));
            adLd_ = dLdZ; %permute(dLdZ,[3 1 2 4]);
            cdLd_low = reshape(adLd_(ps+1:ps+pa,:,:,:),pa,nrows*ncols*nSamples);
            cdLd_low = UnT*cdLd_low;
            adLd_(ps+1:ps+pa,:,:,:) = reshape(cdLd_low,pa,nrows,ncols,nSamples);
            expctddLdX = adLd_; %ipermute(adLd_,[3 1 2 4]);
            
            % dLdWi = <dLdZ,(dVdWi)X>
            expctddLdW = zeros(nAngles,1,datatype);
            for iAngle = 1:nAngles
                dUn = genU.step(angles,mus,iAngle);
                a_ = X; %permute(X,[3 1 2 4]);
                c_low = reshape(a_(ps+1:ps+pa,:,:,:),pa,nrows*ncols*nSamples);
                c_low = dUn*c_low;
                a_ = zeros(size(a_),datatype);
                a_(ps+1:ps+pa,:,:,:) = reshape(c_low,pa,nrows,ncols,nSamples);
                dVdW_X = a_; %ipermute(a_,[3 1 2 4]);
                %
                expctddLdW(iAngle) = sum(dLdZ.*dVdW_X,'all');
            end
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltIntermediateRotation2dLayer(...
                'NumberOfChannels',nchs,...
                'Name','Vn',...
                'Mode','Analysis');
            layer.Mus = mus;
            layer.Angles = angles;
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
