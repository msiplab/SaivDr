classdef nsoltInitialRotationLayerTestCase < matlab.unittest.TestCase
    %NSOLTINITIALROTATIONLAYERTESTCASE
    %
    %   コンポーネント別に入力(nComponents=1のみサポート):
    %      nRows x nCols x nDecs x nSamples
    %
    %   コンポーネント別に出力(nComponents=1のみサポート):
    %      nRows x nCols x nChs x nSamples
    %
    % Requirements: MATLAB R2020a
    %
    % Copyright (c) 2020, Shogo MURAMATSU
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
        nchs = { [3 3], [4 4] };
        stride = { [2 2] };
        datatype = { 'single', 'double' };
        nrows = struct('small', 4,'medium', 8, 'large', 16);
        ncols = struct('small', 4,'medium', 8, 'large', 16);
    end
    
    methods (TestClassTeardown)
        function finalCheck(~)
            import saivdr.dcnn.*
            layer = nsoltInitialRotationLayer(...
                'NumberOfChannels',[3 3],...
                'DecimationFactor',[2 2]);
            fprintf("\n --- Check layer for 2-D images ---\n");
            checkLayer(layer,[8 8 4],'ObservationDimension',4)
        end
    end
    
    methods (Test)
        
        function testConstructor(testCase, nchs, stride)
            
            % Expected values
            expctdName = 'V0';
            expctdDescription = "NSOLT initial rotation ( " ...
                + "(ps,pa) = (" ...
                + nchs(1) + "," + nchs(2) + "), "  ...
                + "(mv,mh) = (" ...
                + stride(1) + "," + stride(2) + ")" ...
                + " )";
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltInitialRotationLayer(...
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
            % nRows x nCols x nDecs x nSamples            
            X = randn(nrows,ncols,nDecs,nSamples,datatype);
            
            % Expected values
            % nRows x nCols x nChs x nSamples
            ps = nchs(1);
            pa = nchs(2);
            W0 = eye(ps,datatype);
            U0 = eye(pa,datatype);
            expctdZ = zeros(nrows,ncols,nChsTotal,nSamples,datatype);
            Y  = zeros(nChsTotal,nrows,ncols,datatype);
            for iSample=1:nSamples
                % Perumation in each block                
                Ai = permute(X(:,:,:,iSample),[3 1 2]); 
                Yi = reshape(Ai,nDecs,nrows,ncols);
                %
                Ys = Yi(1:nDecs/2,:);
                Ya = Yi(nDecs/2+1:end,:);
                Y(1:ps,:,:) = ...
                    reshape(W0(:,1:nDecs/2)*Ys,ps,nrows,ncols);
                Y(ps+1:ps+pa,:,:) = ...
                    reshape(U0(:,1:nDecs/2)*Ya,pa,nrows,ncols);
                expctdZ(:,:,:,iSample) = ipermute(Y,[3 1 2]);                
            end
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltInitialRotationLayer(...
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
            % nRows x nCols x nDecs x nSamples            
            X = randn(nrows,ncols,nDecs,nSamples,datatype);
            angles = randn((nChsTotal-2)*nChsTotal/4,1);
            
            % Expected values
            % nRows x nCols x nChs x nSamples
            ps = nchs(1);
            pa = nchs(2);
            W0 = genW.step(angles(1:length(angles)/2),1);
            U0 = genU.step(angles(length(angles)/2+1:end),1);
            expctdZ = zeros(nrows,ncols,nChsTotal,nSamples,datatype);
            Y  = zeros(nChsTotal,nrows,ncols,datatype);
            for iSample=1:nSamples
                % Perumation in each block                
                Ai = permute(X(:,:,:,iSample),[3 1 2]); 
                Yi = reshape(Ai,nDecs,nrows,ncols);
                %
                Ys = Yi(1:nDecs/2,:);
                Ya = Yi(nDecs/2+1:end,:);
                Y(1:ps,:,:) = ...
                    reshape(W0(:,1:nDecs/2)*Ys,ps,nrows,ncols);
                Y(ps+1:ps+pa,:,:) = ...
                    reshape(U0(:,1:nDecs/2)*Ya,pa,nrows,ncols);
                expctdZ(:,:,:,iSample) = ipermute(Y,[3 1 2]);
            end
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltInitialRotationLayer(...
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


    end
    
end

