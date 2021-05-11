classdef nsoltChannelSeparation3dLayerTestCase < matlab.unittest.TestCase
    %NSOLTCHANNELSEPARATION3DLAYERTESTCASE
    %
    %   １コンポーネント入力(nComponents=1のみサポート):
    %      nChsTotal x nRows x nCols x nSamples
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
        nrows = struct('small', 1,'medium', 4, 'large', 16);
        ncols = struct('small', 1,'medium', 4, 'large', 16);
        nlays = struct('small', 1,'medium', 4, 'large', 16);     
        batch = { 1, 8 };        
    end
    
    methods (TestClassTeardown)
        function finalCheck(~)
            import saivdr.dcnn.*
            layer = nsoltChannelSeparation3dLayer();
            fprintf("\n --- Check layer for 3-D images ---\n");
            checkLayer(layer,[10 8 8 8],'ObservationDimension',5)
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
        
        function testPredict(testCase,nchs,nrows,ncols,nlays,batch,datatype)
            
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            
            % Parameters
            nSamples = batch;
            nChsTotal = sum(nchs);
            % nChsTotal x nRows x nCols x nLays x nSamples
            %X = randn(nrows,ncols,nlays,nChsTotal,nSamples,datatype);
            X = randn(nChsTotal,nrows,ncols,nlays,nSamples,datatype);
            
            % Expected values
            % nRows x nCols x nLays x (nChsTotal-1) x nSamples 
            %expctdZ2 = X(:,:,:,2:end,:);
            expctdZac = permute(X(2:end,:,:,:,:),[2 3 4 1 5]);
            % nRows x nCols x nLays x (nChsTotal-1) x nSamples
            %expctdZ1 = X(:,:,:,1,:);
            expctdZdc = permute(X(1,:,:,:,:),[2 3 4 1 5]);
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltChannelSeparation3dLayer('Name','Sp');
            
            % Actual values
            [actualZac,actualZdc] = layer.predict(X);
            
            % Evaluation
            testCase.verifyInstanceOf(actualZdc,datatype);
            testCase.verifyInstanceOf(actualZac,datatype);            
            testCase.verifyThat(actualZdc,...
                IsEqualTo(expctdZdc,'Within',tolObj));
            testCase.verifyThat(actualZac,...
                IsEqualTo(expctdZac,'Within',tolObj));            
            
        end

                
        function testBackward(testCase,nchs,nrows,ncols,nlays,batch,datatype)
            
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            
            % Parameters
            nSamples = batch;
            nChsTotal = sum(nchs);
            % (nChsTotal-1) x nRows x nCols x nLays x nSamples 
            dLdZac = randn(nrows,ncols,nlays,nChsTotal-1,nSamples,datatype);
            % 1 x nRows x nCols x nLays x nSamples
            dLdZdc = randn(nrows,ncols,nlays,1,nSamples,datatype);

            
            % Expected values
            %  nChsTotal x nRows x nCols xnSamples
            %expctddLdX = cat(4,dLdZdc,dLdZac);
            expctddLdX = ipermute(cat(4,dLdZdc,dLdZac),[2 3 4 1 5]);
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltChannelSeparation3dLayer('Name','Sp');
            
            % Actual values
            actualdLdX = layer.backward([],[],[],dLdZac,dLdZdc,[]);
            
            % Evaluation
            testCase.verifyInstanceOf(actualdLdX,datatype);
            testCase.verifyThat(actualdLdX,...
                IsEqualTo(expctddLdX,'Within',tolObj));
            
        end

        
    end
    
end

