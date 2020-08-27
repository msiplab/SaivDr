classdef nsoltChannelSeparation2dLayerTestCase < matlab.unittest.TestCase
    %NSOLTCHANNELSEPARATION2DLAYERTESTCASE
    %
    %   １コンポーネント入力(nComponents=1のみサポート):
    %      nChsTotal x nRows x nCols x nSamples
    %
    %   ２コンポーネント出力(nComponents=2のみサポート):
    %      1 x nRows x nCols x nSamples
    %      (nChsTotal-1) x nRows x nCols x nSamples    
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
        datatype = { 'single', 'double' };
        nrows = struct('small', 4,'medium', 8, 'large', 16);
        ncols = struct('small', 4,'medium', 8, 'large', 16);
    end
    
    methods (TestClassTeardown)
        function finalCheck(~)
            import saivdr.dcnn.*
            layer = nsoltChannelSeparation2dLayer();
            fprintf("\n --- Check layer for 2-D images ---\n");
            checkLayer(layer,[6 8 8],'ObservationDimension',4)
        end
    end
    
    methods (Test)
        
        function testConstructor(testCase)
            
            % Expected values
            expctdName = 'Sp';
            expctdDescription = "Channel separation";
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltChannelSeparation2dLayer('Name',expctdName);
            
            % Actual values
            actualName = layer.Name;
            actualDescription = layer.Description;
            
            % Evaluation
            testCase.verifyEqual(actualName,expctdName);    
            testCase.verifyEqual(actualDescription,expctdDescription);
        end
        
        function testPredict(testCase,nchs,nrows,ncols,datatype)
            
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            
            % Parameters
            nSamples = 8;
            nChsTotal = sum(nchs);
            % nChsTotal x nRows x nCols x nSamples
            %X = randn(nrows,ncols,nChsTotal,nSamples,datatype);
            X = randn(nChsTotal,nrows,ncols,nSamples,datatype);
            
            % Expected values
            % 1 x nRows x nCols x nSamples
            %expctdZ1 = X(:,:,1,:);
            expctdZ1 = X(1,:,:,:);
            % (nChsTotal-1) x nRows x nCols x nSamples 
            %expctdZ2 = X(:,:,2:end,:);
            expctdZ2 = X(2:end,:,:,:);
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltChannelSeparation2dLayer('Name','Sp');
            
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

        function testBackward(testCase,nchs,nrows,ncols,datatype)
            
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            
            % Parameters
            nSamples = 8;
            nChsTotal = sum(nchs);
            % 1 x nRows x nCols x nSamples
            %dLdZ1 = randn(nrows,ncols,1,nSamples,datatype);
            dLdZ1 = randn(1,nrows,ncols,nSamples,datatype);
            % (nChsTotal-1) x nRows x nCols x nSamples 
            %dLdZ2 = randn(nrows,ncols,nChsTotal-1,nSamples,datatype);
            dLdZ2 = randn(nChsTotal-1,nrows,ncols,nSamples,datatype);
            
            % Expected values
            % nChsTotal x nRows x nCols x nSamples
            %expctddLdX = cat(3,dLdZ1,dLdZ2);
            expctddLdX = cat(1,dLdZ1,dLdZ2);
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltChannelSeparation2dLayer('Name','Sp');
            
            % Actual values
            actualdLdX = layer.backward([],[],[],dLdZ1,dLdZ2,[]);
            
            % Evaluation
            testCase.verifyInstanceOf(actualdLdX,datatype);
            testCase.verifyThat(actualdLdX,...
                IsEqualTo(expctddLdX,'Within',tolObj));
            
        end
    end
    
end

