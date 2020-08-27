classdef nsoltChannelConcatenation2dLayerTestCase < matlab.unittest.TestCase
    %NSOLTCHANNELCONCATENATION2DLAYERTESTCASE
    %
    %   ２コンポーネント入力(nComponents=2のみサポート):
    %      nRows x nCols x 1 x nSamples
    %      nRows x nCols x (nChsTotal-1) x nSamples    
    %
    %   １コンポーネント出力(nComponents=1のみサポート):
    %      nRows x nCols x nChsTotal x nSamples
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
            layer = nsoltChannelConcatenation2dLayer();
            fprintf("\n --- Check layer for 2-D images ---\n");
            checkLayer(layer,{[8 8 1], [8 8 5]},'ObservationDimension',4)
        end
    end
    
    methods (Test)
        
        function testConstructor(testCase)
            
            % Expected values
            expctdName = 'Cn';
            expctdDescription = "Channel concatenation";
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltChannelConcatenation2dLayer('Name',expctdName);
            
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
            % nRows x nCols x 1 x nSamples
            X1 = randn(nrows,ncols,1,nSamples,datatype);
            % nRows x nCols x (nChsTotal-1) x nSamples 
            X2 = randn(nrows,ncols,nChsTotal-1,nSamples,datatype);
            
            % Expected values
            % nChsTotal x nRows x nCols x nSamples
            expctdZ = permute(cat(3,X1,X2),[3 1 2 4]);
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltChannelConcatenation2dLayer('Name','Cn');
            
            % Actual values
            actualZ = layer.predict(X1,X2);
            
            % Evaluation
            testCase.verifyInstanceOf(actualZ,datatype);
            testCase.verifyThat(actualZ,...
                IsEqualTo(expctdZ,'Within',tolObj));
            
        end
                
        function testBackward(testCase,nchs,nrows,ncols,datatype)
            
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            
            % Parameters
            nSamples = 8;
            nChsTotal = sum(nchs);
            % nChsTotal x nRows x nCols x nSamples
            %dLdZ = randn(nrows,ncols,nChsTotal,nSamples,datatype);
            dLdZ = randn(nChsTotal,nrows,ncols,nSamples,datatype);
            
            % Expected values
            % nRows x nCols x 1 x nSamples
            expctddLdX1 = permute(dLdZ(1,:,:,:),[2 3 1 4]);
            % nRows x nCols x (nChsTotal-1) x nSamples 
            expctddLdX2 = permute(dLdZ(2:end,:,:,:),[2 3 1 4]);
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltChannelConcatenation2dLayer('Name','Cn');
            
            % Actual values
            [actualdLdX1,actualdLdX2] = layer.backward([],[],[],dLdZ,[]);
            
            % Evaluation
            testCase.verifyInstanceOf(actualdLdX1,datatype);
            testCase.verifyInstanceOf(actualdLdX2,datatype);            
            testCase.verifyThat(actualdLdX1,...
                IsEqualTo(expctddLdX1,'Within',tolObj));
            testCase.verifyThat(actualdLdX2,...
                IsEqualTo(expctddLdX2,'Within',tolObj));            
            
        end
        
    end
    
end

