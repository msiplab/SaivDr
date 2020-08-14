classdef nsoltChannelSeparation3dLayerTestCase < matlab.unittest.TestCase
    %NSOLTCHANNELSEPARATION3DLAYERTESTCASE
    %
    %   １コンポーネント入力(nComponents=1のみサポート):
    %      nRows x nCols x nChsTotal x nSamples
    %
    %   ２コンポーネント出力(nComponents=2のみサポート):
    %      nRows x nCols x 1 x nSamples
    %      nRows x nCols x (nChsTotal-1) x nSamples    
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
        nrows = struct('small', 4,'medium', 8, 'large', 16);
        ncols = struct('small', 4,'medium', 8, 'large', 16);
        nlays = struct('small', 4,'medium', 8, 'large', 16);        
    end
    
    methods (TestClassTeardown)
        function finalCheck(~)
            import saivdr.dcnn.*
            layer = nsoltChannelSeparation3dLayer();
            fprintf("\n --- Check layer for 3-D images ---\n");
            checkLayer(layer,[8 8 8 10],'ObservationDimension',5)
        end
    end
    
    methods (Test)
        
        function testConstructor(testCase)
            
            % Expected values
            expctdName = 'Sp';
            expctdDescription = "Channel separation";
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltChannelSeparation3dLayer('Name',expctdName);
            
            % Actual values
            actualName = layer.Name;
            actualDescription = layer.Description;
            
            % Evaluation
            testCase.verifyEqual(actualName,expctdName);    
            testCase.verifyEqual(actualDescription,expctdDescription);
        end
        
        function testPredict(testCase,nchs,nrows,ncols,nlays,datatype)
            
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            
            % Parameters
            nSamples = 8;
            nChsTotal = sum(nchs);
            % nRows x nCols x nLays x nChsTotal x nSamples
            X = randn(nrows,ncols,nlays,nChsTotal,nSamples,datatype);
            
            % Expected values
            % nRows x nCols x nLays x 1 x nSamples
            expctdZ1 = X(:,:,:,1,:);
            % nRows x nCols x nLays x (nChsTotal-1) x nSamples 
            expctdZ2 = X(:,:,:,2:end,:);
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltChannelSeparation3dLayer('Name','Sp');
            
            % Actual values
            [actualZ1,actualZ2] = layer.predict(X);
            
            % Evaluation
            testCase.verifyInstanceOf(actualZ1,datatype);
            testCase.verifyInstanceOf(actualZ2,datatype);            
            testCase.verifyThat(actualZ1,...
                IsEqualTo(expctdZ1,'Within',tolObj));
            testCase.verifyThat(actualZ2,...
                IsEqualTo(expctdZ2,'Within',tolObj));            
            
        end

    end
    
end

