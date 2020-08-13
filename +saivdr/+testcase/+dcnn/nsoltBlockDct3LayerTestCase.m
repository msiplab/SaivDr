classdef nsoltBlockDct3LayerTestCase < matlab.unittest.TestCase
    %NSOLTBLOCKDCT3LAYERTESTCASE
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
        stride = { [2 2 2], [1 2 4] };
        datatype = { 'single', 'double' };
        height = struct('small', 8,'medium', 16, 'large', 32);
        width = struct('small', 8,'medium', 16, 'large', 32);
        depth = struct('small', 8,'medium', 16, 'large', 32);
    end
    
    methods (TestClassTeardown)
        function finalCheck(~)
            import saivdr.dcnn.*
            layer = nsoltBlockDct3Layer(...
                'DecimationFactor',[2 2 2]);
            fprintf("\n --- Check layer for 3-D images ---\n");
            checkLayer(layer,[8 8 8 8],'ObservationDimension',5)
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
            layer = nsoltBlockDct3Layer(...
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
            
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            
            % Parameters
            nSamples = 8;
            nComponents = 1;
            X = rand(height,width,depth, nComponents,nSamples, datatype);
            
            % Expected values
            nrows = height/stride(1);
            ncols = width/stride(2);
            nlays = depth/stride(3);
            ndecs = prod(stride);
            expctdZ = zeros(nrows,ncols,nlays,ndecs,nSamples,datatype);
            for iSample = 1:nSamples
                % Block DCT
                Y = testCase.vol2col_(X(:,:,:,1,iSample),stride,...
                    [nrows,ncols,nlays]);
                E0 = testCase.getMatrixE0_(stride);
                A = E0*Y;
                % Rearrange the DCT Coefs.
                expctdZ(:,:,:,:,iSample) = ...
                    permute(reshape(A,ndecs,nrows,ncols,nlays),[2 3 4 1]);
            end
            
            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltBlockDct3Layer(...
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
        
        function y = vol2col_(x,decFactor,nBlocks)
            import saivdr.dictionary.utility.Direction
            decY = decFactor(1);
            decX = decFactor(2);
            decZ = decFactor(3);
            nRows_ = nBlocks(1);
            nCols_ = nBlocks(2);
            nLays_ = nBlocks(3);
            
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
            decY_ = decFactor(1);
            decX_ = decFactor(2);
            decZ_ = decFactor(3);
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

