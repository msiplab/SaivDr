classdef nsoltBlockDct2dLayerTestCase < matlab.unittest.TestCase
    %NSOLTBLOCKDCT2DLAYERTESTCASE
    %
    %   ベクトル配列をブロック配列を入力:
    %      (Stride(1)xnRows) x (Stride(2)xnCols) x nComponents x nSamples
    %
    %   コンポーネント別に出力:
    %      nDecs x nRows x nCols x nSamples
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
        stride = { [1 1], [2 2], [2 4], [4 1], [4 4] };
        datatype = { 'single', 'double' };
        height = struct('small', 8,'medium', 16, 'large', 32);
        width = struct('small', 8,'medium', 16, 'large', 32);
    end
    
    methods (TestClassTeardown)
        function finalCheck(~)
            import saivdr.dcnn.*
            fprintf("\n --- Check layer for 2-D images ---\n");
            % Grayscale
            layer = nsoltBlockDct2dLayer(...
                'DecimationFactor',[2 2]);
            checkLayer(layer,[8 8 1],'ObservationDimension',4,...
                'CheckCodegenCompatibility',true)
            % RGB color
            layer = nsoltBlockDct2dLayer(...
                'NumberOfComponents',3,...
                'DecimationFactor',[2 2]);            
            checkLayer(layer,[8 8 3],'ObservationDimension',4,...
                'CheckCodegenCompatibility',true)
        end
    end
    
    methods (Test)
        
        function testConstructor(testCase, stride)
            
            % Expected values
            expctdName = 'E0';
            expctdDescription = "Block DCT of size " ...
                + stride(1) + "x" + stride(2);
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltBlockDct2dLayer(...
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
            import saivdr.dictionary.utility.Direction
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            
            % Parameters
            nSamples = 8;
            nComponents = 1;
            X = rand(height,width,nComponents,nSamples, datatype);
            
            % Expected values
            nrows = ceil(height/stride(Direction.VERTICAL));
            ncols = ceil(width/stride(Direction.HORIZONTAL));
            ndecs = prod(stride);
            %expctdZ = zeros(nrows,ncols,ndecs,nSamples,datatype);
            expctdZ = zeros(ndecs,nrows,ncols,nSamples,datatype);
            for iSample = 1:nSamples
                % Block DCT
                Y = blockproc(X(:,:,nComponents,iSample),...
                    stride,@(x) dct2(x.data));
                % Rearrange the DCT Coefs.
                A = blockproc(Y,...
                    stride,@testCase.permuteDctCoefs_);
                expctdZ(:,:,:,iSample) = ...
                    ...permute(reshape(A,ndecs,nrows,ncols),[2 3 1]);
                    reshape(A,ndecs,nrows,ncols);
            end
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltBlockDct2dLayer(...
                'DecimationFactor',stride,...
                'Name','E0');
            
            % Actual values
            actualZ = layer.predict(X);
            
            % Evaluation
            testCase.verifyInstanceOf(actualZ,datatype);
            testCase.verifyThat(actualZ,...
                IsEqualTo(expctdZ,'Within',tolObj));
            
        end
        
                
        function testForwardGrayScale(testCase, ...
                stride, height, width, datatype)
            import saivdr.dictionary.utility.Direction
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            
            % Parameters
            nSamples = 8;
            nComponents = 1;
            X = rand(height,width,nComponents,nSamples, datatype);
            
            % Expected values
            nrows = ceil(height/stride(Direction.VERTICAL));
            ncols = ceil(width/stride(Direction.HORIZONTAL));
            ndecs = prod(stride);
            %expctdZ = zeros(nrows,ncols,ndecs,nSamples,datatype);
            expctdZ = zeros(ndecs,nrows,ncols,nSamples,datatype);
            for iSample = 1:nSamples
                % Block DCT
                Y = blockproc(X(:,:,nComponents,iSample),...
                    stride,@(x) dct2(x.data));
                % Rearrange the DCT Coefs.
                A = blockproc(Y,...
                    stride,@testCase.permuteDctCoefs_);
                expctdZ(:,:,:,iSample) = ...
                    ...permute(reshape(A,ndecs,nrows,ncols),[2 3 1]);
                    reshape(A,ndecs,nrows,ncols);
            end
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltBlockDct2dLayer(...
                'DecimationFactor',stride,...
                'Name','E0');
            
            % Actual values
            actualZ = layer.forward(X);
            
            % Evaluation
            testCase.verifyInstanceOf(actualZ,datatype);
            testCase.verifyThat(actualZ,...
                IsEqualTo(expctdZ,'Within',tolObj));
            
        end
        
        
        function testPredictRgbColor(testCase, ...
                stride, height, width, datatype)
            import saivdr.dictionary.utility.Direction
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            
            % Parameters
            nSamples = 8;
            nComponents = 3; % RGB
            X = rand(height,width,nComponents,nSamples, datatype);
            
            % Expected values
            nrows = height/stride(Direction.VERTICAL);
            ncols = width/stride(Direction.HORIZONTAL);
            ndecs = prod(stride);
            %expctdZr = zeros(nrows,ncols,ndecs,nSamples,datatype);
            %expctdZg = zeros(nrows,ncols,ndecs,nSamples,datatype);
            %expctdZb = zeros(nrows,ncols,ndecs,nSamples,datatype);
            expctdZr = zeros(ndecs,nrows,ncols,nSamples,datatype);
            expctdZg = zeros(ndecs,nrows,ncols,nSamples,datatype);
            expctdZb = zeros(ndecs,nrows,ncols,nSamples,datatype);            
            for iSample = 1:nSamples
                % Block DCT
                Yr = blockproc(X(:,:,1,iSample),...
                    stride,@(x) dct2(x.data));
                Yg = blockproc(X(:,:,2,iSample),...
                    stride,@(x) dct2(x.data));
                Yb = blockproc(X(:,:,3,iSample),...
                    stride,@(x) dct2(x.data));
                % Rearrange the DCT Coefs.
                Ar = blockproc(Yr,...
                    stride,@testCase.permuteDctCoefs_);
                Ag = blockproc(Yg,...
                    stride,@testCase.permuteDctCoefs_);
                Ab = blockproc(Yb,...
                    stride,@testCase.permuteDctCoefs_);
                expctdZr(:,:,:,iSample) = ...
                    ...permute(reshape(Ar,ndecs,nrows,ncols),[2 3 1]);
                    reshape(Ar,ndecs,nrows,ncols);
                expctdZg(:,:,:,iSample) = ...
                    ...permute(reshape(Ag,ndecs,nrows,ncols),[2 3 1]);
                    reshape(Ag,ndecs,nrows,ncols);
                expctdZb(:,:,:,iSample) = ...
                    ...permute(reshape(Ab,ndecs,nrows,ncols),[2 3 1]);
                    reshape(Ab,ndecs,nrows,ncols);
            end
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltBlockDct2dLayer(...
                'DecimationFactor',stride,...
                'NumberOfComponents',nComponents,...
                'Name','E0');
            
            % Actual values
            [actualZr,actualZg,actualZb] = layer.predict(X);
            
            % Evaluation
            testCase.verifyInstanceOf(actualZr,datatype);
            testCase.verifyInstanceOf(actualZg,datatype);
            testCase.verifyInstanceOf(actualZb,datatype);
            testCase.verifyThat(actualZr,...
                IsEqualTo(expctdZr,'Within',tolObj));
            testCase.verifyThat(actualZg,...
                IsEqualTo(expctdZg,'Within',tolObj));
            testCase.verifyThat(actualZb,...
                IsEqualTo(expctdZb,'Within',tolObj));
            
        end
        
        function testForwardRgbColor(testCase, ...
                stride, height, width, datatype)
            import saivdr.dictionary.utility.Direction
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            
            % Parameters
            nSamples = 8;
            nComponents = 3; % RGB
            X = rand(height,width,nComponents,nSamples, datatype);
            
            % Expected values
            nrows = height/stride(Direction.VERTICAL);
            ncols = width/stride(Direction.HORIZONTAL);
            ndecs = prod(stride);
            %expctdZr = zeros(nrows,ncols,ndecs,nSamples,datatype);
            %expctdZg = zeros(nrows,ncols,ndecs,nSamples,datatype);
            %expctdZb = zeros(nrows,ncols,ndecs,nSamples,datatype);
            expctdZr = zeros(ndecs,nrows,ncols,nSamples,datatype);
            expctdZg = zeros(ndecs,nrows,ncols,nSamples,datatype);
            expctdZb = zeros(ndecs,nrows,ncols,nSamples,datatype);            
            for iSample = 1:nSamples
                % Block DCT
                Yr = blockproc(X(:,:,1,iSample),...
                    stride,@(x) dct2(x.data));
                Yg = blockproc(X(:,:,2,iSample),...
                    stride,@(x) dct2(x.data));
                Yb = blockproc(X(:,:,3,iSample),...
                    stride,@(x) dct2(x.data));
                % Rearrange the DCT Coefs.
                Ar = blockproc(Yr,...
                    stride,@testCase.permuteDctCoefs_);
                Ag = blockproc(Yg,...
                    stride,@testCase.permuteDctCoefs_);
                Ab = blockproc(Yb,...
                    stride,@testCase.permuteDctCoefs_);
                expctdZr(:,:,:,iSample) = ...
                    ...permute(reshape(Ar,ndecs,nrows,ncols),[2 3 1]);
                    reshape(Ar,ndecs,nrows,ncols);
                expctdZg(:,:,:,iSample) = ...
                    ...permute(reshape(Ag,ndecs,nrows,ncols),[2 3 1]);
                    reshape(Ag,ndecs,nrows,ncols);
                expctdZb(:,:,:,iSample) = ...
                    ...permute(reshape(Ab,ndecs,nrows,ncols),[2 3 1]);
                    reshape(Ab,ndecs,nrows,ncols);
            end
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltBlockDct2dLayer(...
                'DecimationFactor',stride,...
                'NumberOfComponents',nComponents,...
                'Name','E0');
            
            % Actual values
            [actualZr,actualZg,actualZb] = layer.forward(X);
            
            % Evaluation
            testCase.verifyInstanceOf(actualZr,datatype);
            testCase.verifyInstanceOf(actualZg,datatype);
            testCase.verifyInstanceOf(actualZb,datatype);
            testCase.verifyThat(actualZr,...
                IsEqualTo(expctdZr,'Within',tolObj));
            testCase.verifyThat(actualZg,...
                IsEqualTo(expctdZg,'Within',tolObj));
            testCase.verifyThat(actualZb,...
                IsEqualTo(expctdZb,'Within',tolObj));
            
        end
        
        
        function testBackwardGrayScale(testCase, ...
                stride, height, width, datatype)
            import saivdr.dictionary.utility.Direction
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            
            % Parameters
            nSamples = 8;
            nrows = height/stride(Direction.VERTICAL);
            ncols = width/stride(Direction.HORIZONTAL);
            nDecs = prod(stride);
            nComponents = 1;
            %dLdZ = rand(nrows,ncols,nDecs,nSamples,datatype);
            dLdZ = rand(nDecs,nrows,ncols,nSamples,datatype);
            
            % Expected values
            expctddLdX = zeros(height,width,datatype);
            for iSample = 1:nSamples
                %A = reshape(permute(dLdZ(:,:,:,iSample),[3 1 2]),...
                %    nDecs*nrows,ncols);
                A = reshape(dLdZ(:,:,:,iSample),nDecs*nrows,ncols);                
                Y = blockproc(A,[nDecs 1],...
                    @(x) testCase.permuteIdctCoefs_(x.data,stride));
                expctddLdX(:,:,nComponents,iSample) = ...
                    blockproc(Y,...
                    stride,...
                    @(x) idct2(x.data));
            end
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltBlockDct2dLayer(...
                'DecimationFactor',stride,...
                'Name','E0');
            
            % Actual values
            actualdLdX = layer.backward([],[],dLdZ,[]);
            
            % Evaluation
            testCase.verifyInstanceOf(actualdLdX,datatype);
            testCase.verifyThat(actualdLdX,...
                IsEqualTo(expctddLdX,'Within',tolObj));
            
        end
        
        function testBackwardRgbColor(testCase, ...
                stride, height, width, datatype)
            import saivdr.dictionary.utility.Direction
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            
            % Parameters
            nSamples = 8;
            nrows = ceil(height/stride(Direction.VERTICAL));
            ncols = ceil(width/stride(Direction.HORIZONTAL));
            nDecs = prod(stride);
            nComponents = 3; % RGB
            %dLdZr = rand(nrows,ncols,nDecs,nSamples,datatype);
            %dLdZg = rand(nrows,ncols,nDecs,nSamples,datatype);
            %dLdZb = rand(nrows,ncols,nDecs,nSamples,datatype);
            dLdZr = rand(nDecs,nrows,ncols,nSamples,datatype);
            dLdZg = rand(nDecs,nrows,ncols,nSamples,datatype);
            dLdZb = rand(nDecs,nrows,ncols,nSamples,datatype);            
            
            % Expected values
            expctddLdX = zeros(height,width,nComponents,datatype);
            for iSample = 1:nSamples
                %Ar = reshape(permute(dLdZr(:,:,:,iSample),[3 1 2]),...
                %    nDecs*nrows,ncols);
                %Ag = reshape(permute(dLdZg(:,:,:,iSample),[3 1 2]),...
                %    nDecs*nrows,ncols);
                %Ab = reshape(permute(dLdZb(:,:,:,iSample),[3 1 2]),...
                %    nDecs*nrows,ncols);
                Ar = reshape(dLdZr(:,:,:,iSample),nDecs*nrows,ncols);
                Ag = reshape(dLdZg(:,:,:,iSample),nDecs*nrows,ncols);
                Ab = reshape(dLdZb(:,:,:,iSample),nDecs*nrows,ncols);                
                Yr = blockproc(Ar,[nDecs 1],...
                    @(x) testCase.permuteIdctCoefs_(x.data,stride));
                Yg = blockproc(Ag,[nDecs 1],...
                    @(x) testCase.permuteIdctCoefs_(x.data,stride));
                Yb = blockproc(Ab,[nDecs 1],...
                    @(x) testCase.permuteIdctCoefs_(x.data,stride));
                expctddLdX(:,:,1,iSample) = ...
                    blockproc(Yr,...
                    stride,...
                    @(x) idct2(x.data));
                expctddLdX(:,:,2,iSample) = ...
                    blockproc(Yg,...
                    stride,...
                    @(x) idct2(x.data));
                expctddLdX(:,:,3,iSample) = ...
                    blockproc(Yb,...
                    stride,...
                    @(x) idct2(x.data));
            end
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltBlockDct2dLayer(...
                'DecimationFactor',stride,...
                'NumberOfComponents',nComponents,...
                'Name','E0');
            
            % Actual values
            actualdLdX = layer.backward([],[],[],[],dLdZr,dLdZg,dLdZb,[]);
            
            % Evaluation
            testCase.verifyInstanceOf(actualdLdX,datatype);
            testCase.verifyThat(actualdLdX,...
                IsEqualTo(expctddLdX,'Within',tolObj));
            
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
        function value = permuteIdctCoefs_(x,blockSize)
            import saivdr.dictionary.utility.Direction
            coefs = x;
            decY_ = blockSize(Direction.VERTICAL);
            decX_ = blockSize(Direction.HORIZONTAL);
            nQDecsee = ceil(decY_/2)*ceil(decX_/2);
            nQDecsoo = floor(decY_/2)*floor(decX_/2);
            nQDecsoe = floor(decY_/2)*ceil(decX_/2);
            cee = coefs(         1:  nQDecsee);
            coo = coefs(nQDecsee+1:nQDecsee+nQDecsoo);
            coe = coefs(nQDecsee+nQDecsoo+1:nQDecsee+nQDecsoo+nQDecsoe);
            ceo = coefs(nQDecsee+nQDecsoo+nQDecsoe+1:end);
            value = zeros(decY_,decX_,'like',x);
            value(1:2:decY_,1:2:decX_) = reshape(cee,ceil(decY_/2),ceil(decX_/2));
            value(2:2:decY_,2:2:decX_) = reshape(coo,floor(decY_/2),floor(decX_/2));
            value(2:2:decY_,1:2:decX_) = reshape(coe,floor(decY_/2),ceil(decX_/2));
            value(1:2:decY_,2:2:decX_) = reshape(ceo,ceil(decY_/2),floor(decX_/2));
        end
    end
    
end

