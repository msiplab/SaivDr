classdef nsoltBlockDct2LayerTestCase < matlab.unittest.TestCase
    %NSOLTBLOCKDCT2LAYERTESTCASE
    %
    %   ベクトル配列をブロック配列を入力(nComponents=1のみサポート):
    %      (Stride(1)xnRows) x (Stride(2)xnCols) x nComponents x nSamples
    %
    %   コンポーネント別に出力(nComponents=1のみサポート):
    %      nRows x nCols x nDecs x nSamples
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
        stride = { [2 2], [4 4] };
        datatype = { 'single', 'double' };
        height = struct('small', 8,'medium', 16, 'large', 32);
        width = struct('small', 8,'medium', 16, 'large', 32);
    end
    
    methods (TestClassTeardown)
        function finalCheck(~)
            import saivdr.dcnn.*
            layer = nsoltBlockDct2Layer(...
                'DecimationFactor',[2 2]);
            fprintf("\n --- Check layer for 2-D images ---\n");
            checkLayer(layer,[8 8 4],'ObservationDimension',4)
        end
    end
    
    methods (Test)
        
        function testConstructor(testCase, stride)
            
            % Expected values
            expctdName = 'E0';
            expctdDescription = "Block DCT of size " ...
                + stride(1) + "x"' + stride(2);
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltBlockDct2Layer(...
                'DecimationFactor',stride,...
                'Name',expctdName);
            
            % Actual values
            actualName = layer.Name;
            actualDescription = layer.Description;
            
            % Evaluation
            testCase.verifyEqual(actualName,expctdName);
            testCase.verifyEqual(actualDescription,expctdDescription);
        end
        
        function testPredictGrayScale(testCase, ...
                stride, height, width, datatype)
                
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            
            % Parameters
            nSamples = 8;
            nComponents = 1;
            X = rand(height,width,nComponents,nSamples, datatype);
            
            % Expected values
            nrows = height/stride(1);
            ncols = width/stride(2);
            ndecs = prod(stride);
            expctdZ = zeros(nrows,ncols,ndecs,nSamples,datatype);
            for iSample = 1:nSamples
                % Block DCT
                Y = blockproc(X(:,:,nComponents,iSample),...
                    stride,@(x) dct2(x.data));
                % Rearrange the DCT Coefs.
                A = blockproc(Y,...
                    stride,@testCase.permuteDctCoefs_);
                expctdZ(:,:,:,iSample) = ...
                    permute(reshape(A,ndecs,nrows,ncols),[2 3 1]);                
            end
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltBlockDct2Layer(...
                'DecimationFactor',stride,...
                'Name','E0');
            
            % Actual values
            actualZ = layer.predict(X);
            
            % Evaluation
            testCase.verifyInstanceOf(actualZ,datatype);
            testCase.verifyThat(actualZ,...
                IsEqualTo(expctdZ,'Within',tolObj));
            
        end
       
    end
    
    methods (Static, Access = private)
        
        function value = permuteDctCoefs_(x)
            coefs = x.data;
            cee = coefs(1:2:end,1:2:end);
            coo = coefs(2:2:end,2:2:end);
            coe = coefs(2:2:end,1:2:end);
            ceo = coefs(1:2:end,2:2:end);
            value = [ cee(:) ; coo(:) ; coe(:) ; ceo(:) ];
        end
        
    end
    
end

