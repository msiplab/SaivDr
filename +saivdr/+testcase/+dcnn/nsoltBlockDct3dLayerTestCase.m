classdef nsoltBlockDct3dLayerTestCase < matlab.unittest.TestCase
    %NSOLTBLOCKDCT3DLAYERTESTCASE
    %
    %   ベクトル配列をブロック配列を入力(nComponents=1のみサポート):
    %      (Stride(1)xnRows) x (Stride(2)xnCols) x nComponents x nSamples
    %
    %   コンポーネント別に出力(nComponents=1のみサポート):
    %      nDecs x nRows x nCols x nLays x nSamples
    %
    % Requirements: MATLAB R2020a
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
        stride = { [1 1 1], [2 2 2], [1 2 4] };
        datatype = { 'single', 'double' };
        height = struct('small', 8,'medium', 16, 'large', 32);
        width = struct('small', 8,'medium', 16, 'large', 32);
        depth = struct('small', 8,'medium', 16, 'large', 32);
    end
    
    methods (TestClassTeardown)
        function finalCheck(~)
            import saivdr.dcnn.*
            layer = nsoltBlockDct3dLayer(...
                'DecimationFactor',[2 2 2]);
            fprintf("\n --- Check layer for 3-D images ---\n");
            checkLayer(layer,[8 8 8 1],'ObservationDimension',5)
        end
    end
    
    methods (Test)
        
        function testConstructor(testCase, stride)
            
            % Expected values
            expctdName = 'E0';
            expctdDescription = "Block DCT of size " ...
                + stride(1) + "x" + stride(2) + "x" + stride(3);
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltBlockDct3dLayer(...
                'DecimationFactor',stride,...
                'Name',expctdName);
            
            % Actual values
            actualName = layer.Name;
            actualDescription = layer.Description;
            
            % Evaluation
            testCase.verifyEqual(actualName,expctdName);
            testCase.verifyEqual(actualDescription,expctdDescription);
        end
        
        function testPredict(testCase, ...
                stride, height, width, depth, datatype)
            import saivdr.dictionary.utility.Direction
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            
            % Parameters
            nSamples = 8;
            nComponents = 1;
            X = rand(height,width,depth, nComponents,nSamples, datatype);
            
            % Expected values
            nrows = height/stride(Direction.VERTICAL);
            ncols = width/stride(Direction.HORIZONTAL);
            nlays = depth/stride(Direction.DEPTH);
            ndecs = prod(stride);
            %expctdZ = zeros(nrows,ncols,nlays,ndecs,nSamples,datatype);
            expctdZ = zeros(ndecs,nrows,ncols,nlays,nSamples,datatype);
            E0 = testCase.getMatrixE0_(stride);
            for iSample = 1:nSamples
                % Block DCT
                Y = testCase.vol2col_(X(:,:,:,1,iSample),stride,...
                    [nrows,ncols,nlays]);
                A = E0*Y;
                % Rearrange the DCT Coefs.
                expctdZ(:,:,:,:,iSample) = ...
                    ...permute(reshape(A,ndecs,nrows,ncols,nlays),[2 3 4 1]);
                    reshape(A,ndecs,nrows,ncols,nlays);
            end
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltBlockDct3dLayer(...
                'DecimationFactor',stride,...
                'Name','E0');
            
            % Actual values
            actualZ = layer.predict(X);
            
            % Evaluation
            testCase.verifyInstanceOf(actualZ,datatype);
            testCase.verifyThat(actualZ,...
                IsEqualTo(expctdZ,'Within',tolObj));
            
        end
        
        function testForward(testCase, ...
                stride, height, width, depth, datatype)
            import saivdr.dictionary.utility.Direction
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            
            % Parameters
            nSamples = 8;
            nComponents = 1;
            X = rand(height,width,depth, nComponents,nSamples, datatype);
            
            % Expected values
            nrows = height/stride(Direction.VERTICAL);
            ncols = width/stride(Direction.HORIZONTAL);
            nlays = depth/stride(Direction.DEPTH);            
            ndecs = prod(stride);
            %expctdZ = zeros(nrows,ncols,nlays,ndecs,nSamples,datatype);
            expctdZ = zeros(ndecs,nrows,ncols,nlays,nSamples,datatype);
            E0 = testCase.getMatrixE0_(stride);
            for iSample = 1:nSamples
                % Block DCT
                Y = testCase.vol2col_(X(:,:,:,1,iSample),stride,...
                    [nrows,ncols,nlays]);
                A = E0*Y;
                % Rearrange the DCT Coefs.
                expctdZ(:,:,:,:,iSample) = ...
                    ...permute(reshape(A,ndecs,nrows,ncols,nlays),[2 3 4 1]);
                    reshape(A,ndecs,nrows,ncols,nlays);
            end
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltBlockDct3dLayer(...
                'DecimationFactor',stride,...
                'Name','E0');
            
            % Actual values
            actualZ = layer.forward(X);
            
            % Evaluation
            testCase.verifyInstanceOf(actualZ,datatype);
            testCase.verifyThat(actualZ,...
                IsEqualTo(expctdZ,'Within',tolObj));
            
        end
            
            
        function testBackward(testCase, ...
                stride, height, width, depth, datatype)
            import saivdr.dictionary.utility.Direction
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            
            % Parameters
            nSamples = 8;
            nrows = height/stride(Direction.VERTICAL);
            ncols = width/stride(Direction.HORIZONTAL);
            nlays = depth/stride(Direction.DEPTH);
            nDecs = prod(stride);
            %dLdZ = rand(nrows,ncols,nlays,nDecs,nSamples,datatype);
            dLdZ = rand(nDecs,nrows,ncols,nlays,nSamples,datatype);
            
            % Expected values
            expctddLdX = zeros(height,width,depth,datatype);
            E0_T = transpose(testCase.getMatrixE0_(stride));
            for iSample = 1:nSamples
                % Rearrange the DCT coefs
                %A = reshape(permute(dLdZ(:,:,:,:,iSample),[4 1 2 3]),...
                %    nDecs,nrows*ncols*nlays);
                A = reshape(dLdZ(:,:,:,:,iSample),nDecs,nrows*ncols*nlays);
                % Block IDCT
                Y = E0_T*A;
                expctddLdX(:,:,:,1,iSample) = testCase.col2vol_(Y,stride,...
                    [nrows,ncols,nlays]);
            end
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltBlockDct3dLayer(...
                'DecimationFactor',stride,...
                'Name','E0');
            
            % Actual values
            actualdLdX = layer.backward([],[],dLdZ,[]);
            
            % Evaluation
            testCase.verifyInstanceOf(actualdLdX,datatype);
            testCase.verifyThat(actualdLdX,...
                IsEqualTo(expctddLdX,'Within',tolObj));
            
        end
            
        
    end
    
    methods (Static, Access = private)
        
        function x = col2vol_(y,decFactor,nBlocks)
            import saivdr.dictionary.utility.Direction
            decY = decFactor(Direction.VERTICAL);
            decX = decFactor(Direction.HORIZONTAL);
            decZ = decFactor(Direction.DEPTH);
            nRows_ = nBlocks(Direction.VERTICAL);
            nCols_ = nBlocks(Direction.HORIZONTAL);
            nLays_ = nBlocks(Direction.DEPTH);
            
            idx = 0;
            x = zeros(decY*nRows_,decX*nCols_,decZ*nLays_);
            for iLay = 1:nLays_
                idxZ = iLay*decZ;
                for iCol = 1:nCols_
                    idxX = iCol*decX;
                    for iRow = 1:nRows_
                        idxY = iRow*decY;
                        idx = idx + 1;
                        blockData = y(:,idx);
                        x(idxY-decY+1:idxY,...
                            idxX-decX+1:idxX,...
                            idxZ-decZ+1:idxZ) = ...
                            reshape(blockData,decY,decX,decZ);
                    end
                end
            end
            
        end
        
        function y = vol2col_(x,decFactor,nBlocks)
            import saivdr.dictionary.utility.Direction
            decY = decFactor(Direction.VERTICAL);
            decX = decFactor(Direction.HORIZONTAL);
            decZ = decFactor(Direction.DEPTH);
            nRows_ = nBlocks(Direction.VERTICAL);
            nCols_ = nBlocks(Direction.HORIZONTAL);
            nLays_ = nBlocks(Direction.DEPTH);
            
            idx = 0;
            y = zeros(decY*decX*decZ,nRows_*nCols_*nLays_);
            for iLay = 1:nLays_
                idxZ = iLay*decZ;
                for iCol = 1:nCols_
                    idxX = iCol*decX;
                    for iRow = 1:nRows_
                        idxY = iRow*decY;
                        idx = idx + 1;
                        blockData = x(...
                            idxY-decY+1:idxY,...
                            idxX-decX+1:idxX,...
                            idxZ-decZ+1:idxZ);
                        y(:,idx) = blockData(:);
                    end
                end
            end
            
        end
        
        function value = getMatrixE0_(decFactor)
            import saivdr.dictionary.utility.Direction
            decY_ = decFactor(Direction.VERTICAL);
            decX_ = decFactor(Direction.HORIZONTAL);
            decZ_ = decFactor(Direction.DEPTH);
            nElmBi = decY_*decX_*decZ_;
            coefs = zeros(nElmBi);
            iElm = 1;
            % E0.'= [ Beee Beoo Booe Boeo Beeo Beoe Booo Boee ] % Byxz
            % Beee
            for iRow = 1:2:decY_ % y-e
                for iCol = 1:2:decX_ % x-e
                    dctCoefYX = zeros(decY_,decX_);
                    dctCoefYX(iRow,iCol) = 1;
                    basisYX = idct2(dctCoefYX);
                    for iDep = 1:2:decZ_ % z-e
                        dctCoefZ = zeros(decZ_,1);
                        dctCoefZ(iDep) = 1;
                        basisZ  = permute(idct(dctCoefZ),[2 3 1]);
                        basisVd = convn(basisZ,basisYX);
                        coefs(iElm,:) = basisVd(:).';
                        iElm = iElm + 1;
                    end
                end
            end
            % Beoo
            for iRow = 1:2:decY_ % y-e
                for iCol = 2:2:decX_ % x-o
                    dctCoefYX = zeros(decY_,decX_);
                    dctCoefYX(iRow,iCol) = 1;
                    basisYX = idct2(dctCoefYX);
                    for iDep = 2:2:decZ_ % z-o
                        dctCoefZ = zeros(decZ_,1);
                        dctCoefZ(iDep) = 1;
                        basisZ  = permute(idct(dctCoefZ),[2 3 1]);
                        basisVd = convn(basisZ,basisYX);
                        coefs(iElm,:) = basisVd(:).';
                        iElm = iElm + 1;
                    end
                end
            end
            %Booe
            for iRow = 2:2:decY_ % y-o
                for iCol = 2:2:decX_ % x-o
                    dctCoefYX = zeros(decY_,decX_);
                    dctCoefYX(iRow,iCol) = 1;
                    basisYX = idct2(dctCoefYX);
                    for iDep = 1:2:decZ_ % z-e
                        dctCoefZ = zeros(decZ_,1);
                        dctCoefZ(iDep) = 1;
                        basisZ  = permute(idct(dctCoefZ),[2 3 1]);
                        basisVd = convn(basisZ,basisYX);
                        coefs(iElm,:) = basisVd(:).';
                        iElm = iElm + 1;
                    end
                end
            end
            %Boeo
            for iRow = 2:2:decY_ % y-o
                for iCol = 1:2:decX_ % x-e
                    dctCoefYX = zeros(decY_,decX_);
                    dctCoefYX(iRow,iCol) = 1;
                    basisYX = idct2(dctCoefYX);
                    for iDep = 2:2:decZ_ % z-o
                        dctCoefZ = zeros(decZ_,1);
                        dctCoefZ(iDep) = 1;
                        basisZ  = permute(idct(dctCoefZ),[2 3 1]);
                        basisVd = convn(basisZ,basisYX);
                        coefs(iElm,:) = basisVd(:).';
                        iElm = iElm + 1;
                    end
                end
            end
            %Beeo
            for iRow = 1:2:decY_ % y-e
                for iCol = 1:2:decX_ % x-e
                    dctCoefYX = zeros(decY_,decX_);
                    dctCoefYX(iRow,iCol) = 1;
                    basisYX = idct2(dctCoefYX);
                    for iDep = 2:2:decZ_ % z-o
                        dctCoefZ = zeros(decZ_,1);
                        dctCoefZ(iDep) = 1;
                        basisZ  = permute(idct(dctCoefZ),[2 3 1]);
                        basisVd = convn(basisZ,basisYX);
                        coefs(iElm,:) = basisVd(:).';
                        iElm = iElm + 1;
                    end
                end
            end
            %Beoe
            for iRow = 1:2:decY_ % y-e
                for iCol = 2:2:decX_ % x-o
                    dctCoefYX = zeros(decY_,decX_);
                    dctCoefYX(iRow,iCol) = 1;
                    basisYX = idct2(dctCoefYX);
                    for iDep = 1:2:decZ_ % z-e
                        dctCoefZ = zeros(decZ_,1);
                        dctCoefZ(iDep) = 1;
                        basisZ  = permute(idct(dctCoefZ),[2 3 1]);
                        basisVd = convn(basisZ,basisYX);
                        coefs(iElm,:) = basisVd(:).';
                        iElm = iElm + 1;
                    end
                end
            end
            %Booo
            for iRow = 2:2:decY_ % y-o
                for iCol = 2:2:decX_ % x-o
                    dctCoefYX = zeros(decY_,decX_);
                    dctCoefYX(iRow,iCol) = 1;
                    basisYX = idct2(dctCoefYX);
                    for iDep = 2:2:decZ_ % z-o
                        dctCoefZ = zeros(decZ_,1);
                        dctCoefZ(iDep) = 1;
                        basisZ  = permute(idct(dctCoefZ),[2 3 1]);
                        basisVd = convn(basisZ,basisYX);
                        coefs(iElm,:) = basisVd(:).';
                        iElm = iElm + 1;
                    end
                end
            end
            %Boee
            for iRow = 2:2:decY_ % y-o
                for iCol = 1:2:decX_ % x-e
                    dctCoefYX = zeros(decY_,decX_);
                    dctCoefYX(iRow,iCol) = 1;
                    basisYX = idct2(dctCoefYX);
                    for iDep = 1:2:decZ_ % z-e
                        dctCoefZ = zeros(decZ_,1);
                        dctCoefZ(iDep) = 1;
                        basisZ  = permute(idct(dctCoefZ),[2 3 1]);
                        basisVd = convn(basisZ,basisYX);
                        coefs(iElm,:) = basisVd(:).';
                        iElm = iElm + 1;
                    end
                end
            end
            %
            value = coefs;
        end
        
        
    end
    
end

