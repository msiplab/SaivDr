classdef nsoltIntermediateRotation3dLayerTestCase < matlab.unittest.TestCase
    %NSOLTINTERMEDIATEROTATION3DLAYERTESTCASE 
    %   
    %   コンポーネント別に入力(nComponents):
    %      nRows x nCols x nLays x nChsTotal x nSamples
    %
    %   コンポーネント別に出力(nComponents):
    %      nRows x nCols x nLays x nChsTotal x nSamples
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
        nchs = { [4 4], [5 5] };
        datatype = { 'single', 'double' };
        mus = { -1, 1 };
        nrows = struct('small', 4,'medium', 8, 'large', 16);
        ncols = struct('small', 4,'medium', 8, 'large', 16);
        nlays = struct('small', 4,'medium', 8, 'large', 16);
    end

    methods (TestClassTeardown)
        function finalCheck(~)
            import saivdr.dcnn.*
            layer = nsoltIntermediateRotation3dLayer(...
                'NumberOfChannels',[5 5]);
            fprintf("\n --- Check layer for 3-D images ---\n");
            checkLayer(layer,[8 8 8 10],'ObservationDimension',5)      
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
            layer = nsoltIntermediateRotation3dLayer(...
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
                nchs, nrows, ncols, nlays, mus, datatype)
            
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            
            % Parameters
            nSamples = 8;
            nChsTotal = sum(nchs);
            % nRows x nCols x nLays x nChsTotal x nSamples
            X = randn(nrows,ncols,nlays, nChsTotal,nSamples,datatype);
            
            % Expected values
            % nRows x nCols x nLays x nChsTotal x nSamples
            ps = nchs(1);
            pa = nchs(2);
            UnT = mus*eye(pa,datatype);
            Y = permute(X,[4 1 2 3 5]);
            Ya = reshape(Y(ps+1:ps+pa,:,:,:,:),pa,nrows*ncols*nlays*nSamples);
            Za = UnT*Ya;
            Y(ps+1:ps+pa,:,:,:,:) = reshape(Za,pa,nrows,ncols,nlays,nSamples);
            expctdZ = ipermute(Y,[4 1 2 3 5]);
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltIntermediateRotation3dLayer(...
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
                nchs, nrows, ncols, nlays, mus, datatype)
            
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            import saivdr.dictionary.utility.*
            genU = OrthonormalMatrixGenerationSystem();
            
            % Parameters
            nSamples = 8;
            nChsTotal = sum(nchs);
            % nRows x nCols x nLays x nChs x nSamples
            X = randn(nrows,ncols,nlays,nChsTotal,nSamples,datatype);
            angles = randn((nChsTotal-2)*nChsTotal/8,1);
            
            % Expected values
            % nRows x nCols x nLays x nChsTotal x nSamples
            ps = nchs(1);
            pa = nchs(2);
            UnT = transpose(genU.step(angles,mus));
            Y = permute(X,[4 1 2 3 5]);
            Ya = reshape(Y(ps+1:ps+pa,:,:,:,:),pa,nrows*ncols*nlays*nSamples);
            Za = UnT*Ya;
            Y(ps+1:ps+pa,:,:,:,:) = reshape(Za,pa,nrows,ncols,nlays,nSamples);
            expctdZ = ipermute(Y,[4 1 2 3 5]);
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltIntermediateRotation3dLayer(...
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
                nchs, nrows, ncols, nlays, mus, datatype)
            
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-5,single(1e-5));
            import saivdr.dictionary.utility.*
            genU = OrthonormalMatrixGenerationSystem();
            
            % Parameters
            nSamples = 8;
            nChsTotal = sum(nchs);
            % nRows x nCols x nLays x nChs x nSamples
            X = randn(nrows,ncols,nlays,nChsTotal,nSamples,datatype);
            angles = randn((nChsTotal-2)*nChsTotal/8,1);
            
            % Expected values
            % nRows x nCols x nLays x nChsTotal x nSamples
            ps = nchs(1);
            pa = nchs(2);
            Un = genU.step(angles,mus);
            Y = permute(X,[4 1 2 3 5]);
            Ya = reshape(Y(ps+1:ps+pa,:,:,:,:),pa,nrows*ncols*nlays*nSamples);
            Za = Un*Ya;
            Y(ps+1:ps+pa,:,:,:,:) = reshape(Za,pa,nrows,ncols,nlays,nSamples);
            expctdZ = ipermute(Y,[4 1 2 3 5]);
            expctdDescription = "Analysis NSOLT intermediate rotation " ...
                + "(ps,pa) = (" ...
                + nchs(1) + "," + nchs(2) + ")";
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltIntermediateRotation3dLayer(...
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
                
        function testForwardBackwardGrayscale(testCase, ...
                nchs, nrows, ncols, nlays, mus, datatype)
            
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
            
            % nRows x nCols x nLays x nChsTotal x nSamples
            X = randn(nrows,ncols,nlays,nChsTotal,nSamples,datatype);            
            dLdZ = randn(nrows,ncols,nlays,nChsTotal,nSamples,datatype);

            % Expected values
            expctdMem = X;
            % nRows x nCols x nChsTotal x nSamples
            ps = nchs(1);
            pa = nchs(2);
            
            % dLdX = dZdX x dLdZ
            Un = genU.step(angles,mus,0);
            adLd_ = permute(dLdZ,[4 1 2 3 5]);
            cdLd_low = reshape(adLd_(ps+1:ps+pa,:,:,:,:),pa,nrows*ncols*nlays*nSamples);
            cdLd_low = Un*cdLd_low;
            adLd_(ps+1:ps+pa,:,:,:,:) = reshape(cdLd_low,pa,nrows,ncols,nlays,nSamples);
            expctddLdX = ipermute(adLd_,[4 1 2 3 5]);           
            
            % dLdWi = <dLdZ,(dVdWi)X>
            expctddLdW = zeros(nAngles,1,datatype);
            for iAngle = 1:nAngles
                dUn_T = transpose(genU.step(angles,mus,iAngle));
                a_ = permute(X,[4 1 2 3 5]);
                c_low = reshape(a_(ps+1:ps+pa,:,:,:,:),pa,nrows*ncols*nlays*nSamples);
                c_low = dUn_T*c_low;
                a_ = zeros(size(a_),datatype);
                a_(ps+1:ps+pa,:,:,:,:) = reshape(c_low,pa,nrows,ncols,nlays,nSamples);
                dVdW_X = ipermute(a_,[4 1 2 3 5]);
                %
                expctddLdW(iAngle) = sum(dLdZ.*dVdW_X,'all');
            end
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltIntermediateRotation3dLayer(...
                'NumberOfChannels',nchs,...
                'Name','Vn~');
            layer.Mus = mus;
            expctdZ = layer.predict(X);
            
            % Actual values
            [actualZ,actualMem] = layer.forward(X);
            [actualdLdX,actualdLdW] = layer.backward([],[],dLdZ,actualMem);
            
            % Evaluation
            testCase.verifyInstanceOf(actualMem,datatype);            
            testCase.verifyInstanceOf(actualZ,datatype);                        
            testCase.verifyInstanceOf(actualdLdX,datatype);
            testCase.verifyInstanceOf(actualdLdW,datatype);            
            testCase.verifyThat(actualMem,...
                IsEqualTo(expctdMem,'Within',tolObj));
            testCase.verifyThat(actualZ,...
                IsEqualTo(expctdZ,'Within',tolObj));
            testCase.verifyThat(actualdLdX,...
                IsEqualTo(expctddLdX,'Within',tolObj));            
            testCase.verifyThat(actualdLdW,...
                IsEqualTo(expctddLdW,'Within',tolObj));                        
        end
        
        function testForwardBackwardGrayscaleWithRandomAngles(testCase, ...
                nchs, nrows, ncols, nlays, mus, datatype)
    
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
            
            % nRows x nCols x nLays x nChsTotal x nSamples
            X = randn(nrows,ncols,nlays,nChsTotal,nSamples,datatype);            
            dLdZ = randn(nrows,ncols,nlays,nChsTotal,nSamples,datatype);

            % Expected values
            expctdMem = X;
            % nRows x nCols x nLays x nChsTotal x nSamples
            ps = nchs(1);
            pa = nchs(2);
            
            % dLdX = dZdX x dLdZ
            Un = genU.step(angles,mus,0);
            adLd_ = permute(dLdZ,[4 1 2 3 5]);
            cdLd_low = reshape(adLd_(ps+1:ps+pa,:,:,:,:),pa,nrows*ncols*nlays*nSamples);
            cdLd_low = Un*cdLd_low;
            adLd_(ps+1:ps+pa,:,:,:,:) = reshape(cdLd_low,pa,nrows,ncols,nlays,nSamples);
            expctddLdX = ipermute(adLd_,[4 1 2 3 5]);           
            
            % dLdWi = <dLdZ,(dVdWi)X>
            expctddLdW = zeros(nAngles,1,datatype);
            for iAngle = 1:nAngles
                dUn_T = transpose(genU.step(angles,mus,iAngle));
                a_ = permute(X,[4 1 2 3 5]);
                c_low = reshape(a_(ps+1:ps+pa,:,:,:),pa,nrows*ncols*nlays*nSamples);
                c_low = dUn_T*c_low;
                a_ = zeros(size(a_),datatype);
                a_(ps+1:ps+pa,:,:,:,:) = reshape(c_low,pa,nrows,ncols,nlays,nSamples);
                dVdW_X = ipermute(a_,[4 1 2 3 5]);
                %
                expctddLdW(iAngle) = sum(dLdZ.*dVdW_X,'all');
            end
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltIntermediateRotation3dLayer(...
                'NumberOfChannels',nchs,...
                'Name','Vn~');
            layer.Mus = mus;
            layer.Angles = angles;            
            expctdZ = layer.predict(X);
            
            % Actual values
            [actualZ,actualMem] = layer.forward(X);
            [actualdLdX,actualdLdW] = layer.backward([],[],dLdZ,actualMem);
            
            % Evaluation
            testCase.verifyInstanceOf(actualMem,datatype);            
            testCase.verifyInstanceOf(actualZ,datatype);                        
            testCase.verifyInstanceOf(actualdLdX,datatype);
            testCase.verifyInstanceOf(actualdLdW,datatype);            
            testCase.verifyThat(actualMem,...
                IsEqualTo(expctdMem,'Within',tolObj));
            testCase.verifyThat(actualZ,...
                IsEqualTo(expctdZ,'Within',tolObj));
            testCase.verifyThat(actualdLdX,...
                IsEqualTo(expctddLdX,'Within',tolObj));            
            testCase.verifyThat(actualdLdW,...
                IsEqualTo(expctddLdW,'Within',tolObj));                              
        end
        
        function testForwardBackwardGrayscaleAnalysisMode(testCase, ...
                nchs, nrows, ncols, nlays, mus, datatype)
            
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
            
            % nRows x nCols x nLays,nChsTotal x nSamples
            X = randn(nrows,ncols,nlays,nChsTotal,nSamples,datatype);
            dLdZ = randn(nrows,ncols,nlays,nChsTotal,nSamples,datatype);
            
            % Expected values
            expctdMem = X;
            % nRows x nCols x nLays x nChsTotal x nSamples
            ps = nchs(1);
            pa = nchs(2);
            
            % dLdX = dZdX x dLdZ
            UnT = transpose(genU.step(angles,mus,0));
            adLd_ = permute(dLdZ,[4 1 2 3 5]);
            cdLd_low = reshape(adLd_(ps+1:ps+pa,:,:,:,:),pa,nrows*ncols*nlays*nSamples);
            cdLd_low = UnT*cdLd_low;
            adLd_(ps+1:ps+pa,:,:,:,:) = reshape(cdLd_low,pa,nrows,ncols,nlays,nSamples);
            expctddLdX = ipermute(adLd_,[4 1 2 3 5]);
            
            % dLdWi = <dLdZ,(dVdWi)X>
            expctddLdW = zeros(nAngles,1,datatype);
            for iAngle = 1:nAngles
                dUn = genU.step(angles,mus,iAngle);
                a_ = permute(X,[4 1 2 3 5]);
                c_low = reshape(a_(ps+1:ps+pa,:,:,:,:),pa,nrows*ncols*nlays*nSamples);
                c_low = dUn*c_low;
                a_ = zeros(size(a_),datatype);
                a_(ps+1:ps+pa,:,:,:,:) = reshape(c_low,pa,nrows,ncols,nlays,nSamples);
                dVdW_X = ipermute(a_,[4 1 2 3 5]);
                %
                expctddLdW(iAngle) = sum(dLdZ.*dVdW_X,'all');
            end
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltIntermediateRotation3dLayer(...
                'NumberOfChannels',nchs,...
                'Name','Vn',...
                'Mode','Analysis');
            layer.Mus = mus;
            layer.Angles = angles;
            expctdZ = layer.predict(X);
            
            % Actual values
            [actualZ,actualMem] = layer.forward(X);
            [actualdLdX,actualdLdW] = layer.backward([],[],dLdZ,actualMem);
            
            % Evaluation
            testCase.verifyInstanceOf(actualMem,datatype);
            testCase.verifyInstanceOf(actualZ,datatype);
            testCase.verifyInstanceOf(actualdLdX,datatype);
            testCase.verifyInstanceOf(actualdLdW,datatype);
            testCase.verifyThat(actualMem,...
                IsEqualTo(expctdMem,'Within',tolObj));
            testCase.verifyThat(actualZ,...
                IsEqualTo(expctdZ,'Within',tolObj));
            testCase.verifyThat(actualdLdX,...
                IsEqualTo(expctddLdX,'Within',tolObj));
            testCase.verifyThat(actualdLdW,...
                IsEqualTo(expctddLdW,'Within',tolObj));
        end
    end
    
end

