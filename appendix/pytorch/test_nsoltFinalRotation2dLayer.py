import itertools
import unittest
from parameterized import parameterized
import math
import torch
import torch.nn as nn

from nsoltFinalRotation2dLayer import NsoltFinalRotation2dLayer
from nsoltUtility import Direction, OrthonormalMatrixGenerationSystem

nchs = [ [2, 2], [3, 3], [4, 4] ]
stride = [ [1, 1], [1, 2], [2, 2] ]
mus = [ -1, 1 ]
datatype = [ torch.float, torch.double ]
nrows = [ 4, 8, 16 ]
ncols = [ 4, 8, 16 ]

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

        # Parameters
        nSamples = 8
        nDecs = stride[0]*stride[1] # math.prod(stride)
        # nSamples x nRows x nCols x nChs
        X = torch.randn(nSamples,nrows,ncols,sum(nchs),dtype=datatype,requires_grad=True)
        
        # Expected values        
        # nSamples x nRows x nCols x nDecs
        ps, pa = nchs
        W0T = torch.eye(ps,dtype=datatype)
        U0T = torch.eye(pa,dtype=datatype)
        Y = X
        Ys = Y[:,:,:,:ps].view(-1,ps).T
        Ya = Y[:,:,:,ps:].view(-1,pa).T
        Zsa = torch.cat(
                ( W0T[:int(math.ceil(nDecs/2.)),:] @ Ys, 
                  U0T[:int(math.floor(nDecs/2.)),:] @ Ya ),dim=0)
        expctdZ = Zsa.T.view(nSamples,nrows,ncols,nDecs)

        # Instantiation of target class
        layer = NsoltFinalRotation2dLayer(
                number_of_channels=nchs,
                decimation_factor=stride,
                name='V0~'
            )

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
        rtol,atol=1e-4,1e-7
        gen = OrthonormalMatrixGenerationSystem(dtype=datatype)

        # Parameters
        #nchs = [2,2]
        #stride = [2,2]
        #nrows = 4
        #ncols = 6
        nSamples = 8
        nDecs = stride[0]*stride[1] # math.prod(stride)
        nChsTotal = sum(nchs)
        # nSamples x nRows x nCols x nChs
        X = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype)
        angles = torch.randn(int((nChsTotal-2)*nChsTotal/4),dtype=datatype)
    
        # Expected values
        # nSamples x nRows x nCols x nDecs
        ps,pa = nchs
        nAngsW = int(len(angles)/2) 
        angsW,angsU = angles[:nAngsW],angles[nAngsW:]
        W0T,U0T = gen(angsW).T,gen(angsU).T
        Y = X
        Ys = Y[:,:,:,:ps].view(-1,ps).T
        Ya = Y[:,:,:,ps:].view(-1,pa).T
        Zsa = torch.cat(
                ( W0T[:int(math.ceil(nDecs/2.)),:] @ Ys, 
                  U0T[:int(math.floor(nDecs/2.)),:] @ Ya ),dim=0)
        expctdZ = Zsa.T.view(nSamples,nrows,ncols,nDecs)

        # Instantiation of target class
        layer = NsoltFinalRotation2dLayer(
                number_of_channels=nchs,
                decimation_factor=stride,
                name='V0~'
            )
        layer.orthTransW0T.angles.data = angsW
        layer.orthTransW0T.mus = 1
        layer.orthTransU0T.angles.data = angsU
        layer.orthTransU0T.mus = 1

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
    def testPredictGrayscaleWithRandomAnglesNoDcLeackage(self,
        datatype,nchs,stride,nrows,ncols):
        rtol,atol=1e-4,1e-7
        gen = OrthonormalMatrixGenerationSystem(dtype=datatype)

        # Parameters
        #nchs = [4,4]
        #stride = [2,2]
        #nrows = 4
        #ncols = 6
        nSamples = 8
        nDecs = stride[0]*stride[1] # math.prod(stride)
        nChsTotal = sum(nchs)
        # nSamples x nRows x nCols x nChs
        X = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype)
        angles = torch.randn(int((nChsTotal-2)*nChsTotal/4),dtype=datatype)

        # Expected values
        ps, pa = nchs
        nAngsW = int(len(angles)/2) 
        angsW,angsU = angles[:nAngsW],angles[nAngsW:]
        angsWNoDc = angsW.clone()
        angsWNoDc[:ps-1]=torch.zeros(ps-1,dtype=angles.dtype)
        musW,musU = torch.ones(ps,dtype=datatype),torch.ones(pa,dtype=datatype)
        W0T,U0T = gen(angsWNoDc,musW).T,gen(angsU,musU).T        
        Y = X
        Ys = Y[:,:,:,:ps].view(-1,ps).T
        Ya = Y[:,:,:,ps:].view(-1,pa).T
        Zsa = torch.cat(
                ( W0T[:int(math.ceil(nDecs/2.)),:] @ Ys, 
                  U0T[:int(math.floor(nDecs/2.)),:] @ Ya ),dim=0)
        expctdZ = Zsa.T.view(nSamples,nrows,ncols,nDecs)

        # Instantiation of target class
        layer = NsoltFinalRotation2dLayer(
                number_of_channels=nchs,
                decimation_factor=stride,
                no_dc_leakage=True,
                name='V0~'
            )
        layer.orthTransW0T.angles.data = angsW
        layer.orthTransW0T.mus = musW
        layer.orthTransU0T.angles.data = angsU
        layer.orthTransU0T.mus = musU

        # Actual values
        with torch.no_grad():
            actualZ = layer.forward(X)

        # Evaluation
        self.assertEqual(actualZ.dtype,datatype)
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))
        self.assertFalse(actualZ.requires_grad)

    """
        function testBackwardGrayscale(testCase, ...
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
            anglesW = zeros(nAnglesH,1,datatype);            
            anglesU = zeros(nAnglesH,1,datatype);  
            mus_ = 1;
            
            % nDecs x nRows x nCols x nSamples            
            %X = randn(nrows,ncols,sum(nchs),nSamples,datatype); 
            %dLdZ = randn(nrows,ncols,nDecs,nSamples,datatype);            
            X = randn(sum(nchs),nrows,ncols,nSamples,datatype);            
            dLdZ = randn(nDecs,nrows,ncols,nSamples,datatype);
            
            % Expected values
            % nChs x nRows x nCols x nSamples
            ps = nchs(1);
            pa = nchs(2);
            
            % dLdX = dZdX x dLdZ
            W0 = genW.step(anglesW,mus_,0);
            U0 = genU.step(anglesU,mus_,0);
            %expctddLdX = zeros(nrows,ncols,nChsTotal,nSamples,datatype);
            expctddLdX = zeros(nChsTotal,nrows,ncols,nSamples,datatype);
            Y  = zeros(nChsTotal,nrows,ncols,datatype);
            for iSample=1:nSamples
                % Perumation in each block                
                %Ai = permute(dLdZ(:,:,:,iSample),[3 1 2]); 
                Ai = dLdZ(:,:,:,iSample);
                Yi = reshape(Ai,nDecs,nrows,ncols);
                %
                Ys = Yi(1:ceil(nDecs/2),:);
                Ya = Yi(ceil(nDecs/2)+1:end,:);
                Y(1:ps,:,:) = ...
                    reshape(W0(:,1:ceil(nDecs/2))*Ys,ps,nrows,ncols);
                Y(ps+1:ps+pa,:,:) = ...
                    reshape(U0(:,1:floor(nDecs/2))*Ya,pa,nrows,ncols);
                %expctddLdX(:,:,:,iSample) = ipermute(Y,[3 1 2]);                
                expctddLdX(:,:,:,iSample) = Y;
            end
            
            % dLdWi = <dLdZ,(dVdWi)X>
            expctddLdW = zeros(2*nAnglesH,1,datatype);
            dldz_ = dLdZ; %permute(dLdZ,[3 1 2 4]);
            dldz_upp = reshape(dldz_(1:ceil(nDecs/2),:,:,:),ceil(nDecs/2),nrows*ncols*nSamples);
            dldz_low = reshape(dldz_(ceil(nDecs/2)+1:nDecs,:,:,:),floor(nDecs/2),nrows*ncols*nSamples);
            % (dVdWi)X
            for iAngle = 1:nAnglesH
                dW0_T = transpose(genW.step(anglesW,mus_,iAngle));
                dU0_T = transpose(genU.step(anglesU,mus_,iAngle));
                a_ = X; %permute(X,[3 1 2 4]);
                c_upp = reshape(a_(1:ps,:,:,:),ps,nrows*ncols*nSamples);                
                c_low = reshape(a_(ps+1:ps+pa,:,:,:),pa,nrows*ncols*nSamples);
                d_upp = dW0_T(1:ceil(nDecs/2),:)*c_upp;
                d_low = dU0_T(1:floor(nDecs/2),:)*c_low;
                expctddLdW(iAngle) = sum(dldz_upp.*d_upp,'all');
                expctddLdW(nAnglesH+iAngle) = sum(dldz_low.*d_low,'all');
            end
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltFinalRotation2dLayer(...
                'NumberOfChannels',nchs,...
                'DecimationFactor',stride,...
                'Name','V0~');
            layer.Mus = mus_;
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
        
        function testBackwardGayscaleWithRandomAngles(testCase, ...
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
            %X = randn(nrows,ncols,sum(nchs),nSamples,datatype);
            %dLdZ = randn(nrows,ncols,nDecs,nSamples,datatype);
            X = randn(sum(nchs),nrows,ncols,nSamples,datatype);
            dLdZ = randn(nDecs,nrows,ncols,nSamples,datatype);            
            
            % Expected values
            % nChs x nRows x nCols x nSamples
            ps = nchs(1);
            pa = nchs(2);
            
            % dLdX = dZdX x dLdZ
            W0 = genW.step(anglesW,mus_,0);
            U0 = genU.step(anglesU,mus_,0);
            %expctddLdX = zeros(nrows,ncols,nChsTotal,nSamples,datatype);
            expctddLdX = zeros(nChsTotal,nrows,ncols,nSamples,datatype);
            Y  = zeros(nChsTotal,nrows,ncols,datatype);
            for iSample=1:nSamples
                % Perumation in each block
                %Ai = permute(dLdZ(:,:,:,iSample),[3 1 2]);
                Ai = dLdZ(:,:,:,iSample);
                Yi = reshape(Ai,nDecs,nrows,ncols);
                %
                Ys = Yi(1:ceil(nDecs/2),:);
                Ya = Yi(ceil(nDecs/2)+1:end,:);
                Y(1:ps,:,:) = ...
                    reshape(W0(:,1:ceil(nDecs/2))*Ys,ps,nrows,ncols);
                Y(ps+1:ps+pa,:,:) = ...
                    reshape(U0(:,1:floor(nDecs/2))*Ya,pa,nrows,ncols);
                expctddLdX(:,:,:,iSample) = Y; %ipermute(Y,[3 1 2]);
            end
            
            % dLdWi = <dLdZ,(dVdWi)X>
            expctddLdW = zeros(2*nAnglesH,1,datatype);
            dldz_ = dLdZ; %permute(dLdZ,[3 1 2 4]);
            dldz_upp = reshape(dldz_(1:ceil(nDecs/2),:,:,:),ceil(nDecs/2),nrows*ncols*nSamples);
            dldz_low = reshape(dldz_(ceil(nDecs/2)+1:nDecs,:,:,:),floor(nDecs/2),nrows*ncols*nSamples);
            % (dVdWi)X
            for iAngle = 1:nAnglesH
                dW0_T = transpose(genW.step(anglesW,mus_,iAngle));
                dU0_T = transpose(genU.step(anglesU,mus_,iAngle));
                a_ = X; %permute(X,[3 1 2 4]);
                c_upp = reshape(a_(1:ps,:,:,:),ps,nrows*ncols*nSamples);
                c_low = reshape(a_(ps+1:ps+pa,:,:,:),pa,nrows*ncols*nSamples);
                d_upp = dW0_T(1:ceil(nDecs/2),:)*c_upp;
                d_low = dU0_T(1:floor(nDecs/2),:)*c_low;
                expctddLdW(iAngle) = sum(dldz_upp.*d_upp,'all');
                expctddLdW(nAnglesH+iAngle) = sum(dldz_low.*d_low,'all');
            end
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltFinalRotation2dLayer(...
                'NumberOfChannels',nchs,...
                'DecimationFactor',stride,...
                'Name','V0~');
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
        
        function testBackwardWithRandomAnglesNoDcLeackage(testCase, ...
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
            %X = randn(nrows,ncols,sum(nchs),nSamples,datatype);
            %dLdZ = randn(nrows,ncols,nDecs,nSamples,datatype);
            X = randn(sum(nchs),nrows,ncols,nSamples,datatype);
            dLdZ = randn(nDecs,nrows,ncols,nSamples,datatype);            
            
            % Expected values
            % nChs x nRows x nCols x nSamples
            ps = nchs(1);
            pa = nchs(2);
            
            % dLdX = dZdX x dLdZ
            anglesW_NoDc = anglesW;
            anglesW_NoDc(1:ps-1,1)=zeros(ps-1,1);
            musW = mus*ones(ps,1);
            musW(1,1) = 1;
            musU = mus*ones(pa,1);
            W0 = genW.step(anglesW_NoDc,musW,0);
            U0 = genU.step(anglesU,musU,0);
            %expctddLdX = zeros(nrows,ncols,nChsTotal,nSamples,datatype);
            expctddLdX = zeros(nChsTotal,nrows,ncols,nSamples,datatype);
            Y  = zeros(nChsTotal,nrows,ncols,datatype);
            for iSample=1:nSamples
                % Perumation in each block
                %Ai = permute(dLdZ(:,:,:,iSample),[3 1 2]);
                Ai = dLdZ(:,:,:,iSample);
                Yi = reshape(Ai,nDecs,nrows,ncols);
                %
                Ys = Yi(1:ceil(nDecs/2),:);
                Ya = Yi(ceil(nDecs/2)+1:end,:);
                Y(1:ps,:,:) = ...
                    reshape(W0(:,1:ceil(nDecs/2))*Ys,ps,nrows,ncols);
                Y(ps+1:ps+pa,:,:) = ...
                    reshape(U0(:,1:floor(nDecs/2))*Ya,pa,nrows,ncols);
                expctddLdX(:,:,:,iSample) = Y; %ipermute(Y,[3 1 2]);
            end
            
            % dLdWi = <dLdZ,(dVdWi)X>
            expctddLdW = zeros(2*nAnglesH,1,datatype);
            dldz_ = dLdZ; %permute(dLdZ,[3 1 2 4]);
            dldz_upp = reshape(dldz_(1:ceil(nDecs/2),:,:,:),ceil(nDecs/2),nrows*ncols*nSamples);
            dldz_low = reshape(dldz_(ceil(nDecs/2)+1:nDecs,:,:,:),floor(nDecs/2),nrows*ncols*nSamples);
            % (dVdWi)X
            for iAngle = 1:nAnglesH
                dW0_T = transpose(genW.step(anglesW_NoDc,musW,iAngle));
                dU0_T = transpose(genU.step(anglesU,musU,iAngle));
                a_ = X; %permute(X,[3 1 2 4]);
                c_upp = reshape(a_(1:ps,:,:,:),ps,nrows*ncols*nSamples);
                c_low = reshape(a_(ps+1:ps+pa,:,:,:),pa,nrows*ncols*nSamples);
                d_upp = dW0_T(1:ceil(nDecs/2),:)*c_upp;
                d_low = dU0_T(1:floor(nDecs/2),:)*c_low;
                expctddLdW(iAngle) = sum(dldz_upp.*d_upp,'all');
                expctddLdW(nAnglesH+iAngle) = sum(dldz_low.*d_low,'all');
            end
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltFinalRotation2dLayer(...
                'NumberOfChannels',nchs,...
                'DecimationFactor',stride,...
                'NoDcLeakage',true,...
                'Name','V0~');
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


    # Gradient check
    """

if __name__ == '__main__':
    unittest.main()
