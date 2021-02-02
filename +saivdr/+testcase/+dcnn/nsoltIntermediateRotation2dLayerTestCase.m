classdef nsoltIntermediateRotation2dLayerTestCase < matlab.unittest.TestCase
    %NSOLTINTERMEDIATEROTATION2DLAYERTESTCASE 
    %   
    %   コンポーネント別に入力(nComponents):
    %      nChsTotal x nRows x nCols x nSamples
    %
    %   コンポーネント別に出力(nComponents):
    %      nChsTotal x nRows x nCols x nSamples
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
        nchs = { [3 3], [4 4] };
        datatype = { 'single', 'double' };
        mus = { -1, 1 };
        nrows = struct('small', 4,'medium', 8, 'large', 16);
        ncols = struct('small', 4,'medium', 8, 'large', 16);
    end

    methods (TestClassTeardown)
        function finalCheck(~)
            import saivdr.dcnn.*
            layer = nsoltIntermediateRotation2dLayer(...
                'NumberOfChannels',[3 3]);
            fprintf("\n --- Check layer for 2-D images ---\n");
            checkLayer(layer,[6 8 8],...
                'ObservationDimension',4,...
                'CheckCodegenCompatibility',true)      
        end
    end
    
    methods (Test)
        
        function testConstructor(testCase, nchs)
            
            % Expected values
            expctdName = 'Vn~';
            expctdMode = 'Synthesis';
            expctdDescription = "Synthesis NSOLT intermediate rotation " ...
                + "(ps,pa) = (" ...
                + nchs(1) + "," + nchs(2) + ")";
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltIntermediateRotation2dLayer(...
                'NumberOfChannels',nchs,...
                'Name',expctdName);
            
            % Actual values
            actualName = layer.Name;
            actualMode = layer.Mode;
            actualDescription = layer.Description;
            
            % Evaluation
            testCase.verifyEqual(actualName,expctdName);
            testCase.verifyEqual(actualMode,expctdMode);
            testCase.verifyEqual(actualDescription,expctdDescription);
        end
        
        function testPredictGrayscale(testCase, ...
                nchs, nrows, ncols, mus, datatype)
            
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            
            % Parameters
            nSamples = 8;
            nChsTotal = sum(nchs);
            % nChsTotal x nRows x nCols xnSamples
            %X = randn(nrows,ncols,nChsTotal,nSamples,datatype);
            X = randn(nChsTotal,nrows,ncols,nSamples,datatype);
            
            % Expected values
            % nChsTotal x nRows x nCols x nSamples
            ps = nchs(1);
            pa = nchs(2);
            UnT = mus*eye(pa,datatype);
            Y = X; %permute(X,[3 1 2 4]);
            Ya = reshape(Y(ps+1:ps+pa,:,:,:),pa,nrows*ncols*nSamples);
            Za = UnT*Ya;
            Y(ps+1:ps+pa,:,:,:) = reshape(Za,pa,nrows,ncols,nSamples);
            expctdZ = Y; %ipermute(Y,[3 1 2 4]);
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltIntermediateRotation2dLayer(...
                'NumberOfChannels',nchs,...
                'Name','Vn~');
            
            % Actual values
            layer.Mus = mus;
            actualZ = layer.predict(X);
            
            % Evaluation
            testCase.verifyInstanceOf(actualZ,datatype);
            testCase.verifyThat(actualZ,...
                IsEqualTo(expctdZ,'Within',tolObj));
            
        end
        
        function testPredictGrayscaleWithRandomAngles(testCase, ...
                nchs, nrows, ncols, mus, datatype)
            
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            import saivdr.dictionary.utility.*
            genU = OrthonormalMatrixGenerationSystem();
            
            % Parameters
            nSamples = 8;
            nChsTotal = sum(nchs);
            % nChsTotal x nRows x nCols x nSamples
            %X = randn(nrows,ncols,nChsTotal,nSamples,datatype);
            X = randn(nChsTotal,nrows,ncols,nSamples,datatype);
            angles = randn((nChsTotal-2)*nChsTotal/8,1);
            
            % Expected values
            % nChsTotal x nRows x nCols x nSamples
            ps = nchs(1);
            pa = nchs(2);
            UnT = transpose(genU.step(angles,mus));
            Y = X; %permute(X,[3 1 2 4]);
            Ya = reshape(Y(ps+1:ps+pa,:,:,:),pa,nrows*ncols*nSamples);
            Za = UnT*Ya;
            Y(ps+1:ps+pa,:,:,:) = reshape(Za,pa,nrows,ncols,nSamples);
            expctdZ = Y; %ipermute(Y,[3 1 2 4]);
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltIntermediateRotation2dLayer(...
                'NumberOfChannels',nchs,...
                'Name','Vn~');
            
            % Actual values
            layer.Mus = mus;
            layer.Angles = angles;
            actualZ = layer.predict(X);
            
            % Evaluation
            testCase.verifyInstanceOf(actualZ,datatype);
            testCase.verifyThat(actualZ,...
                IsEqualTo(expctdZ,'Within',tolObj));
            
        end
        
        function testPredictGrayscaleAnalysisMode(testCase, ...
                nchs, nrows, ncols, mus, datatype)
            
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            import saivdr.dictionary.utility.*
            genU = OrthonormalMatrixGenerationSystem();
            
            % Parameters
            nSamples = 8;
            nChsTotal = sum(nchs);
            % nChsTotal x nRows x nCols x nSamples
            %X = randn(nrows,ncols,nChsTotal,nSamples,datatype);
            X = randn(nChsTotal,nrows,ncols,nSamples,datatype);
            angles = randn((nChsTotal-2)*nChsTotal/8,1);
            
            % Expected values
            % nChsTotal x nRows x nCols x nSamples
            ps = nchs(1);
            pa = nchs(2);
            Un = genU.step(angles,mus);
            Y = X; % permute(X,[3 1 2 4]);
            Ya = reshape(Y(ps+1:ps+pa,:,:,:),pa,nrows*ncols*nSamples);
            Za = Un*Ya;
            Y(ps+1:ps+pa,:,:,:) = reshape(Za,pa,nrows,ncols,nSamples);
            expctdZ = Y; %ipermute(Y,[3 1 2 4]);
            expctdDescription = "Analysis NSOLT intermediate rotation " ...
                + "(ps,pa) = (" ...
                + nchs(1) + "," + nchs(2) + ")";
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltIntermediateRotation2dLayer(...
                'NumberOfChannels',nchs,...
                'Name','Vn',...
                'Mode','Analysis');
            
            % Actual values
            layer.Mus = mus;
            layer.Angles = angles;
            actualZ = layer.predict(X);
            actualDescription = layer.Description;
            
            % Evaluation
            testCase.verifyInstanceOf(actualZ,datatype);
            testCase.verifyThat(actualZ,...
                IsEqualTo(expctdZ,'Within',tolObj));
            testCase.verifyEqual(actualDescription,expctdDescription);            
            
        end
        
        function testBackwardGrayscale(testCase, ...
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
            angles = zeros(nAngles,1,datatype);
            
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
            adLd_ = dLdZ; %permute(dLdZ,[3 1 2 4]);
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
        
    end
    
end