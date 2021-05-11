classdef nsoltChannelConcatenation3dLayerTestCase < matlab.unittest.TestCase
    %NSOLTCHANNELCONCATENATION3DLAYERTESTCASE
    %
    %   ２コンポーネント入力(nComponents=2のみサポート):
    %      nRows x nCols x (nChsTotal-1) x nSamples
    %      nRows x nCols x 1 x nSamples
    %
    %   １コンポーネント出力(nComponents=1のみサポート):
    %      nChsTotal x nRows x nCols x nSamples
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
        nrows = struct('small', 1,'medium', 4, 'large', 16);
        ncols = struct('small', 1,'medium', 4, 'large', 16);
        nlays = struct('small', 1,'medium', 4, 'large', 16);
        batch = { 1, 8 };
    end
    
    methods (TestClassTeardown)
        function finalCheck(~)
            import saivdr.dcnn.*
            layer = nsoltChannelConcatenation3dLayer();
            fprintf("\n --- Check layer for 3-D images ---\n");
            checkLayer(layer,{[8 8 8 9], [8 8 8 1]},'ObservationDimension',5)
        end
    end
    
    methods (Test)
        
        function testConstructor(testCase)
            
            % Expected values
            expctdName = 'Cn';
            expctdDescription = "Channel concatenation";
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltChannelConcatenation3dLayer('Name',expctdName);
            
            % Actual values
            actualName = layer.Name;
            actualDescription = layer.Description;
            
            % Evaluation
            testCase.verifyEqual(actualName,expctdName);
            testCase.verifyEqual(actualDescription,expctdDescription);
        end
        
        function testPredict(testCase,nchs,nrows,ncols,nlays,batch,datatype)
            
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            
            % Parameters
            nSamples = batch;
            nChsTotal = sum(nchs);
            % nRows x nCols x nLays x (nChsTotal-1) x nSamples
            Xac = randn(nrows,ncols,nlays,nChsTotal-1,nSamples,datatype);
            % nRows x nCols x nLays x 1 x nSamples
            Xdc = randn(nrows,ncols,nlays,1,nSamples,datatype);
            
            % Expected values
            % nChsTotal x nRows x nCols x nSamples
            %expctdZ = cat(4,Xdc,Xac);
            expctdZ = permute(cat(4,Xdc,Xac),[4 1 2 3 5]);
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltChannelConcatenation3dLayer('Name','Cn');
            
            % Actual values
            actualZ = layer.predict(Xac,Xdc);
            
            % Evaluation
            testCase.verifyInstanceOf(actualZ,datatype);
            testCase.verifyThat(actualZ,...
                IsEqualTo(expctdZ,'Within',tolObj));
            
        end
        
        function testBackward(testCase,nchs,nrows,ncols,nlays,batch,datatype)
            
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            
            % Parameters
            nSamples = batch;
            nChsTotal = sum(nchs);
            % nChsTotal x nRows x nCols x nLays x nSamples
            %dLdZ = randn(nrows,ncols,nlays,nChsTotal,nSamples,datatype);
            dLdZ = randn(nChsTotal,nrows,ncols,nlays,nSamples,datatype);
            
            % Expected values
            % nRows x nCols x nLays x (nChsTotal-1) x nSamples
            expctddLdXac = permute(dLdZ(2:end,:,:,:,:),[2 3 4 1 5]);
            % nRows x nCols x nLays x 1 x nSamples
            expctddLdXdc = permute(dLdZ(1,:,:,:,:),[2 3 4 1 5]);

            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltChannelConcatenation3dLayer('Name','Cn');
            
            % Actual values
            [actualdLdXac,actualdLdXdc] = layer.backward([],[],[],dLdZ,[]);
            
            % Evaluation
            testCase.verifyInstanceOf(actualdLdXdc,datatype);
            testCase.verifyInstanceOf(actualdLdXac,datatype);
            testCase.verifyThat(actualdLdXdc,...
                IsEqualTo(expctddLdXdc,'Within',tolObj));
            testCase.verifyThat(actualdLdXac,...
                IsEqualTo(expctddLdXac,'Within',tolObj));
            
        end
        
    end
    
end

