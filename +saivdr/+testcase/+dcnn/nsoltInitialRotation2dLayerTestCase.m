classdef nsoltInitialRotation2dLayerTestCase < matlab.unittest.TestCase
    %NSOLTINITIALROTATION2DLAYERTESTCASE
    %
    %   コンポーネント別に入力(nComponents=1のみサポート):
    %      nDecs x nRows x nCols x nSamples
    %
    %   コンポーネント別に出力(nComponents=1のみサポート):
    %      nChs x nRows x nCols x nSamples
    %
    % Requirements: MATLAB R2020b
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
        nchs = { [2 2], [3 3], [4 4] };
        stride = { [1 1], [1 2], [2 2] };
        mus = { -1, 1 };
        datatype = { 'single', 'double' };
        nrows = struct('small', 4,'medium', 8, 'large', 16);
        ncols = struct('small', 4,'medium', 8, 'large', 16);
    end
    
    methods (TestClassTeardown)
        function finalCheck(~)
            import saivdr.dcnn.*
            layer = nsoltInitialRotation2dLayer(...
                'NumberOfChannels',[3 3],...
                'DecimationFactor',[2 2]);
            fprintf("\n --- Check layer for 2-D images ---\n");
            checkLayer(layer,[4 8 8],...
                'ObservationDimension',4,...
                'CheckCodegenCompatibility',true)
        end
    end
    
    methods (Test)
        
        function testConstructor(testCase, nchs, stride)
            
            % Expected values
            expctdName = 'V0';
            expctdDescription = "NSOLT initial rotation " ...
                + "(ps,pa) = (" ...
                + nchs(1) + "," + nchs(2) + "), "  ...
                + "(mv,mh) = (" ...
                + stride(1) + "," + stride(2) + ")";
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltInitialRotation2dLayer(...
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
                nchs, stride, nrows, ncols, datatype)
            
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            
            % Parameters
            nSamples = 8;
            nDecs = prod(stride);
            nChsTotal = sum(nchs);
            % nDecs x nRows x nCols x nSamples
            %X = randn(nrows,ncols,nDecs,nSamples,datatype);
            X = randn(nDecs,nrows,ncols,nSamples,datatype);
            
            % Expected values
            % nChs x nRows x nCols x nSamples
            ps = nchs(1);
            pa = nchs(2);
            W0 = eye(ps,datatype);
            U0 = eye(pa,datatype);
            %expctdZ = zeros(nrows,ncols,nChsTotal,nSamples,datatype);
            expctdZ = zeros(nChsTotal,nrows,ncols,nSamples,datatype);
            Y  = zeros(nChsTotal,nrows,ncols,datatype);
            for iSample=1:nSamples
                % Perumation in each block
                Ai = X(:,:,:,iSample); %permute(X(:,:,:,iSample),[3 1 2]);
                Yi = reshape(Ai,nDecs,nrows,ncols);
                %
                Ys = Yi(1:ceil(nDecs/2),:);
                Ya = Yi(ceil(nDecs/2)+1:end,:);
                Y(1:ps,:,:) = ...
                    reshape(W0(:,1:ceil(nDecs/2))*Ys,ps,nrows,ncols);
                Y(ps+1:ps+pa,:,:) = ...
                    reshape(U0(:,1:floor(nDecs/2))*Ya,pa,nrows,ncols);
                expctdZ(:,:,:,iSample) = Y; %ipermute(Y,[3 1 2]);
            end
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltInitialRotation2dLayer(...
                'NumberOfChannels',nchs,...
                'DecimationFactor',stride,...
                'Name','V0');
            
            % Actual values
            actualZ = layer.predict(X);
            
            % Evaluation
            testCase.verifyInstanceOf(actualZ,datatype);
            testCase.verifyThat(actualZ,...
                IsEqualTo(expctdZ,'Within',tolObj));
            
        end
        
        function testPredictGrayscaleWithRandomAngles(testCase, ...
                nchs, stride, nrows, ncols, datatype)
            
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
            % nDecs x nRows x nCols x nSamples
            %X = randn(nrows,ncols,nDecs,nSamples,datatype);
            X = randn(nDecs,nrows,ncols,nSamples,datatype);
            angles = randn((nChsTotal-2)*nChsTotal/4,1);
            
            % Expected values
            % nChs x nRows x nCols x nSamples
            ps = nchs(1);
            pa = nchs(2);
            W0 = genW.step(angles(1:length(angles)/2),1);
            U0 = genU.step(angles(length(angles)/2+1:end),1);
            %expctdZ = zeros(nrows,ncols,nChsTotal,nSamples,datatype);
            expctdZ = zeros(nChsTotal,nrows,ncols,nSamples,datatype);
            Y  = zeros(nChsTotal,nrows,ncols,datatype);
            for iSample=1:nSamples
                % Perumation in each block
                Ai = X(:,:,:,iSample); %permute(X(:,:,:,iSample),[3 1 2]);
                Yi = reshape(Ai,nDecs,nrows,ncols);
                %
                Ys = Yi(1:ceil(nDecs/2),:);
                Ya = Yi(ceil(nDecs/2)+1:end,:);
                Y(1:ps,:,:) = ...
                    reshape(W0(:,1:ceil(nDecs/2))*Ys,ps,nrows,ncols);
                Y(ps+1:ps+pa,:,:) = ...
                    reshape(U0(:,1:floor(nDecs/2))*Ya,pa,nrows,ncols);
                expctdZ(:,:,:,iSample) = Y; %ipermute(Y,[3 1 2]);
            end
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltInitialRotation2dLayer(...
                'NumberOfChannels',nchs,...
                'DecimationFactor',stride,...
                'Name','V0');
            
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
                nchs, stride, nrows, ncols, datatype)
            
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
            % nDecs x nRows x nCols x nSamples
            %X = randn(nrows,ncols,nDecs,nSamples,datatype);
            X = randn(nDecs,nrows,ncols,nSamples,datatype);
            angles = randn((nChsTotal-2)*nChsTotal/4,1);
            
            % Expected values
            % nChs x nRows x nCols x nSamples
            ps = nchs(1);
            pa = nchs(2);
            W0 = genW.step(angles(1:length(angles)/2),1);
            U0 = genU.step(angles(length(angles)/2+1:end),1);
            %expctdZ = zeros(nrows,ncols,nChsTotal,nSamples,datatype);
            expctdZ = zeros(nChsTotal,nrows,ncols,nSamples,datatype);
            Y  = zeros(nChsTotal,nrows,ncols,datatype);
            for iSample=1:nSamples
                % Perumation in each block
                Ai = X(:,:,:,iSample); %permute(X(:,:,:,iSample),[3 1 2]);
                Yi = reshape(Ai,nDecs,nrows,ncols);
                %
                Ys = Yi(1:ceil(nDecs/2),:);
                Ya = Yi(ceil(nDecs/2)+1:end,:);
                Y(1:ps,:,:) = ...
                    reshape(W0(:,1:ceil(nDecs/2))*Ys,ps,nrows,ncols);
                Y(ps+1:ps+pa,:,:) = ...
                    reshape(U0(:,1:floor(nDecs/2))*Ya,pa,nrows,ncols);
                expctdZ(:,:,:,iSample) = Y; %ipermute(Y,[3 1 2]);
            end
            expctdZ = dlarray(expctdZ);
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltInitialRotation2dLayer(...
                'NumberOfChannels',nchs,...
                'DecimationFactor',stride,...
                'Name','V0');
            
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
                nchs, stride, nrows, ncols, mus,datatype)
            
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
            % nDecs x nRows x nCols x nDecs x nSamples
            %X = randn(nrows,ncols,nDecs,nSamples,datatype);
            X = randn(nDecs,nrows,ncols,nSamples,datatype);
            angles = randn((nChsTotal-2)*nChsTotal/4,1);
            
            % Expected values
            % nChs x nRows x nCols x nSamples
            ps = nchs(1);
            pa = nchs(2);
            anglesNoDc = angles;
            anglesNoDc(1:ps-1,1)=zeros(ps-1,1);
            musW = mus*ones(ps,1);
            musW(1,1) = 1;
            musU = mus*ones(pa,1);
            W0 = genW.step(anglesNoDc(1:length(angles)/2),musW);
            U0 = genU.step(anglesNoDc(length(angles)/2+1:end),musU);
            %expctdZ = zeros(nrows,ncols,nChsTotal,nSamples,datatype);
            expctdZ = zeros(nChsTotal,nrows,ncols,nSamples,datatype);
            Y  = zeros(nChsTotal,nrows,ncols,datatype);
            for iSample=1:nSamples
                % Perumation in each block
                Ai = X(:,:,:,iSample); %permute(X(:,:,:,iSample),[3 1 2]);
                Yi = reshape(Ai,nDecs,nrows,ncols);
                %
                Ys = Yi(1:ceil(nDecs/2),:);
                Ya = Yi(ceil(nDecs/2)+1:end,:);
                Y(1:ps,:,:) = ...
                    reshape(W0(:,1:ceil(nDecs/2))*Ys,ps,nrows,ncols);
                Y(ps+1:ps+pa,:,:) = ...
                    reshape(U0(:,1:floor(nDecs/2))*Ya,pa,nrows,ncols);
                expctdZ(:,:,:,iSample) = Y; %ipermute(Y,[3 1 2]);
            end
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltInitialRotation2dLayer(...
                'NumberOfChannels',nchs,...
                'DecimationFactor',stride,...
                'NoDcLeakage',true,...
                'Name','V0');
            
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
            Y = dLdZ; % permute(dLdZ,[3 1 2 4]);
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
        
    end
    
end

