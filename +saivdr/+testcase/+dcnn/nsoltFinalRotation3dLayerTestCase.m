classdef nsoltFinalRotation3dLayerTestCase < matlab.unittest.TestCase
    %NSOLTFINALROTATION3DLAYERTESTCASE
    %
    %   コンポーネント別に入力(nComponents):
    %      nRows x nCols x nLays x nChs x nSamples
    %
    %   コンポーネント別に出力(nComponents):
    %       nDecs x nRows x nCols x nLays xnSamples
    %
    % Requirements: MATLAB R2020a
    %
    % Copyright (c) 2020-2021, Shogo MURAMATSU
    %
    % All rights reserved.
    %
    % Contact address: Shogo MURAMATSU,
    %                Faculty of Engineering, Niigata University,
    %                8050 2-no-cho Ikarashi, Nishi-ku,
    %                Niigata, 950-2181, JAPAN
    %
    % http://msiplab.eng.niigata-u.ac.jp/
    
    properties (TestParameter)
        nchs = { [4 4], [5 5] };
        stride = { [1 1 1], [2 2 2], [1 2 4] };
        mus = { -1, 1 };
        datatype = { 'single', 'double' };
        nrows = struct('small', 4,'medium', 8, 'large', 16);
        ncols = struct('small', 4,'medium', 8, 'large', 16);
        nlays = struct('small', 4,'medium', 8, 'large', 16);
    end
    
    methods (TestClassTeardown)
        function finalCheck(~)
            import saivdr.dcnn.*
            layer = nsoltFinalRotation3dLayer(...
                'NumberOfChannels',[5 5],...
                'DecimationFactor',[2 2 2]);
            fprintf("\n --- Check layer for 3-D images ---\n");
            checkLayer(layer,[10 8 8 8],'ObservationDimension',5)
        end
    end
    
    methods (Test)
        
        function testConstructor(testCase, nchs, stride)
            
            % Expected values
            expctdName = 'V0~';
            expctdDescription = "NSOLT final rotation " ...
                + "(ps,pa) = (" ...
                + nchs(1) + "," + nchs(2) + "), "  ...
                + "(mv,mh,md) = (" ...
                + stride(1) + "," + stride(2) + "," + stride(3) + ")";
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltFinalRotation3dLayer(...
                'NumberOfChannels',nchs,...
                'DecimationFactor',stride,...
                'Name',expctdName);
            
            % Actual values
            actualName = layer.Name;
            actualDescription = layer.Description;
            
            % Evaluation
            testCase.verifyEqual(actualName,expctdName);
            testCase.verifyEqual(actualDescription,expctdDescription);
        end
        
        function testPredictGrayscale(testCase, ...
                nchs, stride, nrows, ncols, nlays, datatype)
            
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            
            % Parameters
            nSamples = 8;
            nDecs = prod(stride);
            % nChs x nRows x nCols x nLays x nSamples
            %X = randn(nrows,ncols,nlays,sum(nchs),nSamples,datatype);
            X = randn(sum(nchs),nrows,ncols,nlays,nSamples,datatype);
            % Expected values
            % nDecs x nRows x nCols x nLays x nSamples
            ps = nchs(1);
            pa = nchs(2);
            W0T = eye(ps,datatype);
            U0T = eye(pa,datatype);
            Y = X; %permute(X,[4 1 2 3 5]);
            Ys = reshape(Y(1:ps,:,:,:,:),ps,nrows*ncols*nlays*nSamples);
            Ya = reshape(Y(ps+1:ps+pa,:,:,:,:),pa,nrows*ncols*nlays*nSamples);
            Zsa = [ W0T(1:ceil(nDecs/2),:)*Ys; U0T(1:floor(nDecs/2),:)*Ya ];
            %expctdZ = ipermute(reshape(Zsa,nDecs,nrows,ncols,nlays,nSamples),...
            %    [4 1 2 3 5]);
            expctdZ = reshape(Zsa,nDecs,nrows,ncols,nlays,nSamples);
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltFinalRotation3dLayer(...
                'NumberOfChannels',nchs,...
                'DecimationFactor',stride,...
                'Name','V0~');
            
            % Actual values
            actualZ = layer.predict(X);
            
            % Evaluation
            testCase.verifyInstanceOf(actualZ,datatype);
            testCase.verifyThat(actualZ,...
                IsEqualTo(expctdZ,'Within',tolObj));
            
        end
        
        function testPredictGrayscaleWithRandomAngles(testCase, ...
                nchs, stride, nrows, ncols, nlays, datatype)
            
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            import saivdr.dictionary.utility.*
            genW = OrthonormalMatrixGenerationSystem();
            genU = OrthonormalMatrixGenerationSystem();
            
            % Parameters
            nSamples = 8;
            nDecs = prod(stride);
            nChsTotal = sum(nchs);
            % nChs x nRows x nCols x nLays x nSamples
            %X = randn(nrows,ncols,nlays,sum(nchs),nSamples,datatype);
            X = randn(sum(nchs),nrows,ncols,nlays,nSamples,datatype);
            angles = randn((nChsTotal-2)*nChsTotal/4,1);
            
            % Expected values
            % nDecs x nRows x nCols x nLays x nSamples
            ps = nchs(1);
            pa = nchs(2);
            W0T = transpose(genW.step(angles(1:length(angles)/2),1));
            U0T = transpose(genU.step(angles(length(angles)/2+1:end),1));
            Y = X; %permute(X,[4 1 2 3 5]);
            Ys = reshape(Y(1:ps,:,:,:,:),ps,nrows*ncols*nlays*nSamples);
            Ya = reshape(Y(ps+1:ps+pa,:,:,:,:),pa,nrows*ncols*nlays*nSamples);
            Zsa = [ W0T(1:ceil(nDecs/2),:)*Ys; U0T(1:floor(nDecs/2),:)*Ya ];
            %expctdZ = ipermute(reshape(Zsa,nDecs,nrows,ncols,nlays,nSamples),...
            %    [4 1 2 3 5]);
            expctdZ = reshape(Zsa,nDecs,nrows,ncols,nlays,nSamples);
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltFinalRotation3dLayer(...
                'NumberOfChannels',nchs,...
                'DecimationFactor',stride,...
                'Name','V0~');
            
            % Actual values
            layer.Angles = angles;
            actualZ = layer.predict(X);
            
            % Evaluation
            testCase.verifyInstanceOf(actualZ,datatype);
            testCase.verifyThat(actualZ,...
                IsEqualTo(expctdZ,'Within',tolObj));
            
        end
        
        %{        
        function testPredictGrayscaleWithDlarrayAngles(testCase, ...
                nchs, stride, nrows, ncols, nlays, datatype)
            
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            import saivdr.dictionary.utility.*
            genW = OrthonormalMatrixGenerationSystem();
            genU = OrthonormalMatrixGenerationSystem();
            
            % Parameters
            nSamples = 8;
            nDecs = prod(stride);
            nChsTotal = sum(nchs);
            % nChs x nRows x nCols x nLays x nSamples
            %X = randn(nrows,ncols,nlays,sum(nchs),nSamples,datatype);
            X = randn(sum(nchs),nrows,ncols,nlays,nSamples,datatype);
            angles = randn((nChsTotal-2)*nChsTotal/4,1);
            
            % Expected values
            % nDecs x nRows x nCols x nLays x nSamples
            ps = nchs(1);
            pa = nchs(2);
            W0T = transpose(genW.step(angles(1:length(angles)/2),1));
            U0T = transpose(genU.step(angles(length(angles)/2+1:end),1));
            Y = X; %permute(X,[4 1 2 3 5]);
            Ys = reshape(Y(1:ps,:,:,:,:),ps,nrows*ncols*nlays*nSamples);
            Ya = reshape(Y(ps+1:ps+pa,:,:,:,:),pa,nrows*ncols*nlays*nSamples);
            Zsa = [ W0T(1:ceil(nDecs/2),:)*Ys; U0T(1:floor(nDecs/2),:)*Ya ];
            %expctdZ = ipermute(reshape(Zsa,nDecs,nrows,ncols,nlays,nSamples),...
            %    [4 1 2 3 5]);
            expctdZ = dlarray(reshape(Zsa,nDecs,nrows,ncols,nlays,nSamples));
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltFinalRotation3dLayer(...
                'NumberOfChannels',nchs,...
                'DecimationFactor',stride,...
                'Name','V0~');
            
            % Actual values
            layer.Angles = dlarray(angles);
            actualZ = layer.predict(X);
            
            % Evaluation
            testCase.verifyInstanceOf(actualZ,'dlarray');
            testCase.verifyEqual(actualZ.underlyingType,datatype);
            testCase.verifyThat(actualZ,...
                IsEqualTo(expctdZ,'Within',tolObj));
            
        end
        %}
        
        function testPredictGrayscaleWithRandomAnglesNoDcLeackage(testCase, ...
                nchs, stride, nrows, ncols, nlays, mus, datatype)
            
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            import saivdr.dictionary.utility.*
            genW = OrthonormalMatrixGenerationSystem();
            genU = OrthonormalMatrixGenerationSystem();
            
            % Parameters
            nSamples = 8;
            nDecs = prod(stride);
            nChsTotal = sum(nchs);
            % nChs x nRows x nCols x nLays x nSamples
            %X = randn(nrows,ncols,nlays,sum(nchs),nSamples,datatype);
            X = randn(sum(nchs),nrows,ncols,nlays,nSamples,datatype);
            angles = randn((nChsTotal-2)*nChsTotal/4,1);
            
            % Expected values
            % nDecs x nRows x nCols x nLays x nSamples
            ps = nchs(1);
            pa = nchs(2);
            anglesNoDc = angles;
            anglesNoDc(1:ps-1,1)=zeros(ps-1,1);
            musW = mus*ones(ps,1);
            musW(1,1) = 1;
            musU = mus*ones(pa,1);
            W0T = transpose(genW.step(anglesNoDc(1:length(angles)/2),musW));
            U0T = transpose(genU.step(anglesNoDc(length(angles)/2+1:end),musU));
            Y = X; %permute(X,[4 1 2 3 5]);
            Ys = reshape(Y(1:ps,:,:,:,:),ps,nrows*ncols*nlays*nSamples);
            Ya = reshape(Y(ps+1:ps+pa,:,:,:,:),pa,nrows*ncols*nlays*nSamples);
            Zsa = [ W0T(1:ceil(nDecs/2),:)*Ys; U0T(1:floor(nDecs/2),:)*Ya ];
            %expctdZ = ipermute(reshape(Zsa,nDecs,nrows,ncols,nlays,nSamples),...
            %    [4 1 2 3 5]);
            expctdZ = reshape(Zsa,nDecs,nrows,ncols,nlays,nSamples);
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltFinalRotation3dLayer(...
                'NumberOfChannels',nchs,...
                'DecimationFactor',stride,...
                'NoDcLeakage',true,...
                'Name','V0~');
            
            % Actual values
            layer.Mus = mus;
            layer.Angles = angles;
            actualZ = layer.predict(X);
            
            % Evaluation
            testCase.verifyInstanceOf(actualZ,datatype);
            testCase.verifyThat(actualZ,...
                IsEqualTo(expctdZ,'Within',tolObj));
            
        end
        
        function testBackwardGrayscale(testCase, ...
                nchs, stride, nrows, ncols, nlays, datatype)
            
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-3,single(1e-3));
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
            
            %  nDecs x nRows x nCols x nLays xnSamples
            %X = randn(nrows,ncols,nlays,sum(nchs),nSamples,datatype);
            %dLdZ = randn(nrows,ncols,nlays,nDecs,nSamples,datatype);
            X = randn(sum(nchs),nrows,ncols,nlays,nSamples,datatype);
            dLdZ = randn(nDecs,nrows,ncols,nlays,nSamples,datatype);            
            
            % Expected values
            % nRows x nCols x nChs x nSamples
            ps = nchs(1);
            pa = nchs(2);
            
            % dLdX = dZdX x dLdZ
            W0 = genW.step(anglesW,mus_,0);
            U0 = genU.step(anglesU,mus_,0);
            %expctddLdX = zeros(nrows,ncols,nlays,nChsTotal,nSamples,datatype);
            expctddLdX = zeros(nChsTotal,nrows,ncols,nlays,nSamples,datatype);
            Y  = zeros(nChsTotal,nrows,ncols,nlays,datatype);
            for iSample=1:nSamples
                % Perumation in each block
                %Ai = permute(dLdZ(:,:,:,:,iSample),[4 1 2 3]);
                Ai = dLdZ(:,:,:,:,iSample);
                Yi = reshape(Ai,nDecs,nrows,ncols,nlays);
                %
                Ys = Yi(1:ceil(nDecs/2),:);
                Ya = Yi(ceil(nDecs/2)+1:end,:);
                Y(1:ps,:,:,:) = ...
                    reshape(W0(:,1:ceil(nDecs/2))*Ys,ps,nrows,ncols,nlays);
                Y(ps+1:ps+pa,:,:,:) = ...
                    reshape(U0(:,1:floor(nDecs/2))*Ya,pa,nrows,ncols,nlays);
                %expctddLdX(:,:,:,:,iSample) = ipermute(Y,[4 1 2 3]);
                expctddLdX(:,:,:,:,iSample) = Y;
            end
            
            % dLdWi = <dLdZ,(dVdWi)X>
            expctddLdW = zeros(2*nAnglesH,1,datatype);
            dldz_ = dLdZ; % permute(dLdZ,[4 1 2 3 5]);
            dldz_upp = reshape(dldz_(1:ceil(nDecs/2),:,:,:,:),ceil(nDecs/2),nrows*ncols*nlays*nSamples);
            dldz_low = reshape(dldz_(ceil(nDecs/2)+1:nDecs,:,:,:,:),floor(nDecs/2),nrows*ncols*nlays*nSamples);
            % (dVdWi)X
            for iAngle = 1:nAnglesH
                dW0_T = transpose(genW.step(anglesW,mus_,iAngle));
                dU0_T = transpose(genU.step(anglesU,mus_,iAngle));
                a_ = X; % permute(X,[4 1 2 3 5]);
                c_upp = reshape(a_(1:ps,:,:,:,:,:),ps,nrows*ncols*nlays*nSamples);
                c_low = reshape(a_(ps+1:ps+pa,:,:,:,:,:),pa,nrows*ncols*nlays*nSamples);
                d_upp = dW0_T(1:ceil(nDecs/2),:)*c_upp;
                d_low = dU0_T(1:floor(nDecs/2),:)*c_low;
                expctddLdW(iAngle) = sum(dldz_upp.*d_upp,'all');
                expctddLdW(nAnglesH+iAngle) = sum(dldz_low.*d_low,'all');
            end
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltFinalRotation3dLayer(...
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
                nchs, stride, nrows, ncols, nlays, datatype)
            
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-3,single(1e-3));
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
            
            % nDecs x nRows x nCols x nLays x nSamples
            %X = randn(nrows,ncols,nlays,sum(nchs),nSamples,datatype);
            %dLdZ = randn(nrows,ncols,nlays,nDecs,nSamples,datatype);
            X = randn(sum(nchs),nrows,ncols,nlays,nSamples,datatype);
            dLdZ = randn(nDecs,nrows,ncols,nlays,nSamples,datatype);            
            
            % Expected values
            % nChs x nRows x nCols x nSamples
            ps = nchs(1);
            pa = nchs(2);
            
            % dLdX = dZdX x dLdZ
            W0 = genW.step(anglesW,mus_,0);
            U0 = genU.step(anglesU,mus_,0);
            %expctddLdX = zeros(nrows,ncols,nlays,nChsTotal,nSamples,datatype);
            expctddLdX = zeros(nChsTotal,nrows,ncols,nlays,nSamples,datatype);
            Y  = zeros(nChsTotal,nrows,ncols,nlays,datatype);
            for iSample=1:nSamples
                % Perumation in each block
                %Ai = permute(dLdZ(:,:,:,:,iSample),[4 1 2 3]);
                Ai = dLdZ(:,:,:,:,iSample);
                Yi = reshape(Ai,nDecs,nrows,ncols,nlays);
                %
                Ys = Yi(1:ceil(nDecs/2),:);
                Ya = Yi(ceil(nDecs/2)+1:end,:);
                Y(1:ps,:,:,:) = ...
                    reshape(W0(:,1:ceil(nDecs/2))*Ys,ps,nrows,ncols,nlays);
                Y(ps+1:ps+pa,:,:,:) = ...
                    reshape(U0(:,1:floor(nDecs/2))*Ya,pa,nrows,ncols,nlays);
                expctddLdX(:,:,:,:,iSample) = Y; % ipermute(Y,[4 1 2 3]);
            end
            
            % dLdWi = <dLdZ,(dVdWi)X>
            expctddLdW = zeros(2*nAnglesH,1,datatype);
            dldz_ = dLdZ; %permute(dLdZ,[4 1 2 3 5]);
            dldz_upp = reshape(dldz_(1:ceil(nDecs/2),:,:,:,:),ceil(nDecs/2),nrows*ncols*nlays*nSamples);
            dldz_low = reshape(dldz_(ceil(nDecs/2)+1:nDecs,:,:,:,:),floor(nDecs/2),nrows*ncols*nlays*nSamples);
            % (dVdWi)X
            for iAngle = 1:nAnglesH
                dW0_T = transpose(genW.step(anglesW,mus_,iAngle));
                dU0_T = transpose(genU.step(anglesU,mus_,iAngle));
                a_ = X; % permute(X,[4 1 2 3 5]);
                c_upp = reshape(a_(1:ps,:,:,:,:),ps,nrows*ncols*nlays*nSamples);
                c_low = reshape(a_(ps+1:ps+pa,:,:,:,:),pa,nrows*ncols*nlays*nSamples);
                d_upp = dW0_T(1:ceil(nDecs/2),:)*c_upp;
                d_low = dU0_T(1:floor(nDecs/2),:)*c_low;
                expctddLdW(iAngle) = sum(dldz_upp.*d_upp,'all');
                expctddLdW(nAnglesH+iAngle) = sum(dldz_low.*d_low,'all');
            end
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltFinalRotation3dLayer(...
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
                nchs, stride, nrows, ncols, nlays, mus, datatype)
            
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-3,single(1e-3));
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
            
            % nDecs x nRows x nCols x nLays x nSamples
            %X = randn(nrows,ncols,nlays,sum(nchs),nSamples,datatype);
            %dLdZ = randn(nrows,ncols,nlays,nDecs,nSamples,datatype);
            X = randn(sum(nchs),nrows,ncols,nlays,nSamples,datatype);
            dLdZ = randn(nDecs,nrows,ncols,nlays,nSamples,datatype);            
            
            % Expected values
            % nChs x nRows x nCols x nChs x nSamples
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
            %expctddLdX = zeros(nrows,ncols,nlays,nChsTotal,nSamples,datatype);
            expctddLdX = zeros(nChsTotal,nrows,ncols,nlays,nSamples,datatype);
            Y  = zeros(nChsTotal,nrows,ncols,nlays,datatype);
            for iSample=1:nSamples
                % Perumation in each block
                %Ai = permute(dLdZ(:,:,:,:,iSample),[4 1 2 3]);
                Ai = dLdZ(:,:,:,:,iSample);
                Yi = reshape(Ai,nDecs,nrows,ncols,nlays);
                %
                Ys = Yi(1:ceil(nDecs/2),:);
                Ya = Yi(ceil(nDecs/2)+1:end,:);
                Y(1:ps,:,:,:) = ...
                    reshape(W0(:,1:ceil(nDecs/2))*Ys,ps,nrows,ncols,nlays);
                Y(ps+1:ps+pa,:,:,:) = ...
                    reshape(U0(:,1:floor(nDecs/2))*Ya,pa,nrows,ncols,nlays);
                expctddLdX(:,:,:,:,iSample) = Y; % ipermute(Y,[4 1 2 3]);
            end
            
            % dLdWi = <dLdZ,(dVdWi)X>
            expctddLdW = zeros(2*nAnglesH,1,datatype);
            dldz_ = dLdZ; % permute(dLdZ,[4 1 2 3 5]);
            dldz_upp = reshape(dldz_(1:ceil(nDecs/2),:,:,:,:),ceil(nDecs/2),nrows*ncols*nlays*nSamples);
            dldz_low = reshape(dldz_(ceil(nDecs/2)+1:nDecs,:,:,:,:),floor(nDecs/2),nrows*ncols*nlays*nSamples);
            % (dVdWi)X
            for iAngle = 1:nAnglesH
                dW0_T = transpose(genW.step(anglesW_NoDc,musW,iAngle));
                dU0_T = transpose(genU.step(anglesU,musU,iAngle));
                a_ = X; %permute(X,[4 1 2 3 5]);
                c_upp = reshape(a_(1:ps,:,:,:,:),ps,nrows*ncols*nlays*nSamples);
                c_low = reshape(a_(ps+1:ps+pa,:,:,:,:),pa,nrows*ncols*nlays*nSamples);
                d_upp = dW0_T(1:ceil(nDecs/2),:)*c_upp;
                d_low = dU0_T(1:floor(nDecs/2),:)*c_low;
                expctddLdW(iAngle) = sum(dldz_upp.*d_upp,'all');
                expctddLdW(nAnglesH+iAngle) = sum(dldz_low.*d_low,'all');
            end
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltFinalRotation3dLayer(...
                'NumberOfChannels',nchs,...
                'DecimationFactor',stride,...
                'NoDcLeakage',true,...
                'Name','V0~');
            layer.Mus = mus;
            layer.Angles = [anglesW; anglesU];
            %expctdZ = layer.predict(X);
            
            % Actual values
            [actualdLdX,actualdLdW] = layer.backward(X,[],dLdZ);
            
            % Evaluation
            testCase.verifyInstanceOf(actualdLdX,datatype);
            testCase.verifyInstanceOf(actualdLdW,datatype);
            testCase.verifyThat(actualdLdX,...
                IsEqualTo(expctddLdX,'Within',tolObj));
            testCase.verifyThat(actualdLdW,...
                IsEqualTo(expctddLdW,'Within',tolObj));
        end
        
    end
    
end

