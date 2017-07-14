classdef OvsdLpPuFb3dTypeICostEvaluator < ... %#codegen
        saivdr.dictionary.nsoltx.design.AbstOvsdLpPuFbCostEvaluator
    %OVSDLPPUFB3DTYPEICOSTEVALUATOR Cost evaluator for Type-I NSOLT
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2015-2017, Shogo MURAMATSU
    %
    % All rights reserved.
    %
    % Contact address: Shogo MURAMATSU,
    %                Faculty of Engineering, Niigata University,
    %                8050 2-no-cho Ikarashi, Nishi-ku,
    %                Niigata, 950-2181, JAPAN
    %
    % http://msiplab.eng.niigata-u.ac.jp/
    %
    
    properties (Access = protected, Constant = true)
        DATA_DIMENSION = 3
    end
    
    methods
        
        % Constractor
        function obj = OvsdLpPuFb3dTypeICostEvaluator(varargin)
            obj = obj@saivdr.dictionary.nsoltx.design.AbstOvsdLpPuFbCostEvaluator(...
                varargin{:});
        end
        
    end
    
    methods (Access=protected)
        
        function dim = getDataDimension(obj)
            dim = obj.DATA_DIMENSION;
        end
        
%         function s = saveObjectImpl(obj)
%             s = saveObjectImpl@saivdr.dictionary.nsoltx.design.AbstOvsdLpPuFbCostEvaluator(obj);
%         end
%         
%         function loadObjectImpl(obj,s,wasLocked)
%             loadObjectImpl@saivdr.dictionary.nsoltx.design.AbstOvsdLpPuFbCostEvaluator(obj,s,wasLocked);
%         end     
        
%         function validatePropertiesImpl(~)
%         end        
        
        function setupImpl(obj,~,~,scales)
            
            nch = [ obj.NumberOfSymmetricChannels ...
                obj.NumberOfAntisymmetricChannels ];
            nChs = sum(nch);
            
            % Check nLeves
            nLevels = (size(scales,1)-1)/(nChs-1);
            if nLevels ~= 1
                error('Number of tree levels should be one.');
            end
            
            % Atom concatenator
            if exist('fcn_NsoltAtomConcatenator3dCodeGen_mex','file')==3
                obj.atomCncFcn = @fcn_NsoltAtomConcatenator3dCodeGen_mex;
            else
                import saivdr.dictionary.nsoltx.mexsrcs.fcn_NsoltAtomConcatenator3dCodeGen;
                obj.atomCncFcn = @fcn_NsoltAtomConcatenator3dCodeGen;
            end
            % Gradient evaluator
            if exist('fcn_GradEvalSteps3dCodeGen_mex','file')==3
                obj.gradFcn = @fcn_GradEvalSteps3dCodeGen_mex;
            else
                import saivdr.dictionary.nsoltx.mexsrcs.fcn_GradEvalSteps3dCodeGen
                obj.gradFcn = @fcn_GradEvalSteps3dCodeGen;
            end
        end
                
        function [cost,grad] = stepImpl(obj,srcImg,coefs,scales)
            pmMtx = step(obj.LpPuFb,[],[]);
            %
            [recImg,intrCoefs] = synthesize_(obj, coefs, scales, pmMtx);
            difImg = srcImg-recImg;
            cost = sum(difImg(:).^2);
            %
            angs = get(obj.LpPuFb,'Angles');
            mus  = get(obj.LpPuFb,'Mus');
            isnodc = isa(obj.LpPuFb,...
                'saivdr.dictionary.nsoltx.OvsdLpPuFb3dTypeIVm1System');
            grad = gradient_(obj, difImg, intrCoefs, pmMtx, scales, ...
                angs, mus, isnodc);
        end
                
    end
   
    methods (Access = private)
        
        function grad = gradient_(obj, difImg, arrayCoefsB, pmMtx, scales,...
                angs, mus, isnodc)
            import saivdr.dictionary.utility.Direction
            %
            pmCoefs = get(pmMtx,'Coefficients');
            %
            ps  = obj.NumberOfSymmetricChannels;
            pa  = obj.NumberOfAntisymmetricChannels;
            nChs  = ps + pa;
            %
            decY_  = obj.decimationFactor(Direction.VERTICAL);
            decX_  = obj.decimationFactor(Direction.HORIZONTAL);
            decZ_  = obj.decimationFactor(Direction.DEPTH);
            %
            nRows_ = uint32(scales(1,1));
            nCols_ = uint32(scales(1,2));
            nLays_ = uint32(scales(1,3));
            %
            subScale  = [ nRows_ nCols_ nLays_ ];
            blockSize = [ decY_ decX_ decZ_ ];
            
            % Prepare array
            arrayCoefsC = zeros(nChs,nRows_*nCols_*nLays_);
            
            % Block DCT for difImg (Step 2)
           if decY_ == 1 && decX_ == 1 && decZ_ == 1
                coefs = obj.vol2col_(difImg,blockSize,subScale);
                arrayCoefsC(1,:) = coefs(1,:);
            elseif decY_ == 2 && decX_ == 2 && decZ_ == 2
                difImg1 = difImg(1:2:end,1:2:end,1:2:end);
                difImg2 = difImg(2:2:end,1:2:end,1:2:end);
                difImg3 = difImg(1:2:end,2:2:end,1:2:end);
                difImg4 = difImg(2:2:end,2:2:end,1:2:end);
                difImg5 = difImg(1:2:end,1:2:end,2:2:end);
                difImg6 = difImg(2:2:end,1:2:end,2:2:end);
                difImg7 = difImg(1:2:end,2:2:end,2:2:end);
                difImg8 = difImg(2:2:end,2:2:end,2:2:end);
                %
                difImg1 = difImg1(:).';
                difImg2 = difImg2(:).';
                difImg3 = difImg3(:).';
                difImg4 = difImg4(:).';
                difImg5 = difImg5(:).';
                difImg6 = difImg6(:).';
                difImg7 = difImg7(:).';
                difImg8 = difImg8(:).';
                %
                difImg1p2 = difImg1 + difImg2;
                difImg1m2 = difImg1 - difImg2;
                difImg3p4 = difImg3 + difImg4;
                difImg3m4 = difImg3 - difImg4;                
                difImg5p6 = difImg5 + difImg6;
                difImg5m6 = difImg5 - difImg6;                                
                difImg7p8 = difImg7 + difImg8;
                difImg7m8 = difImg7 - difImg8;        
                %
                difImg1p2p3p4 = difImg1p2 + difImg3p4;
                difImg1p2m3m4 = difImg1p2 - difImg3p4;
                difImg1m2p3m4 = difImg1m2 + difImg3m4;
                difImg1m2m3p4 = difImg1m2 - difImg3m4;                
                difImg5p6p7p8 = difImg5p6 + difImg7p8;
                difImg5p6m7m8 = difImg5p6 - difImg7p8;
                difImg5m6p7m8 = difImg5m6 + difImg7m8;
                difImg5m6m7p8 = difImg5m6 - difImg7m8;                                
                %
                arrayCoefsC(1,:)    = difImg1p2p3p4 + difImg5p6p7p8;
                arrayCoefsC(2,:)    = difImg1p2m3m4 - difImg5p6m7m8;
                arrayCoefsC(3,:)    = difImg1m2m3p4 + difImg5m6m7p8;
                arrayCoefsC(4,:)    = difImg1m2p3m4 - difImg5m6p7m8;
                arrayCoefsC(ps+1,:) = difImg1p2p3p4 - difImg5p6p7p8;
                arrayCoefsC(ps+2,:) = difImg1p2m3m4 + difImg5p6m7m8;
                arrayCoefsC(ps+3,:) = difImg1m2m3p4 - difImg5m6m7p8;
                arrayCoefsC(ps+4,:) = difImg1m2p3m4 + difImg5m6p7m8;
                %
                arrayCoefsC = arrayCoefsC/(2*sqrt(2));
            else 
                nDec_=decY_*decX_*decZ_;
                mc = ceil(nDec_/2);
                mf = floor(nDec_/2);
                coefs = obj.vol2col_(difImg,blockSize,subScale);
                E0 = obj.getMatrixE0_(blockSize);
                coefs = E0*coefs;
                arrayCoefsC(1:mc,:) = coefs(1:mc,:);
                arrayCoefsC(ps+1:ps+mf,:) = coefs(mc+1:end,:);
            end
            
            % Gradient calculation steps                  
            fpe = strcmp(obj.BoundaryOperation,'Circular');
            ord = uint32(obj.polyPhaseOrder);
            grad = obj.gradFcn(...
                arrayCoefsB, arrayCoefsC, subScale, pmCoefs, ...
                angs, mus, [ps pa], ord, fpe, isnodc);
        end
        
        function [recImg,arrayCoefs] = synthesize_(obj,coefs,scales,pmMtx)
            import saivdr.dictionary.utility.Direction
            %
            pmCoefs = get(pmMtx,'Coefficients');
            nChs = obj.NumberOfSymmetricChannels ...
                + obj.NumberOfAntisymmetricChannels;
            %
            iSubband = 1;
            eIdx = prod(scales(iSubband,:));
            %
            height = scales(1,1);
            width  = scales(1,2);
            depth  = scales(1,3);
            arrayCoefs = zeros(nChs,height*width*depth);
            arrayCoefs(1,:) = coefs(1:eIdx);
            for iCh = 2:nChs
                iSubband = iSubband + 1;
                sIdx = eIdx + 1;
                eIdx = sIdx + prod(scales(iSubband,:))-1;
                arrayCoefs(iCh,:) = coefs(sIdx:eIdx);
            end
            %
            ps = obj.NumberOfSymmetricChannels;
            nRows_ = uint32(height);
            nCols_ = uint32(width);
            nLays_ = uint32(depth);
            decY_  = obj.decimationFactor(Direction.VERTICAL);
            decX_  = obj.decimationFactor(Direction.HORIZONTAL);
            decZ_  = obj.decimationFactor(Direction.DEPTH);
            nDec   = decY_*decX_*decZ_;
            %
            blockSize = [ decY_ decX_ decZ_ ];
            %
            if isinteger(arrayCoefs)
                arrayCoefs = double(arrayCoefs);
            end
            
            % Atom concatenation
            subScale  = [ nRows_ nCols_ nLays_ ];
            ord  = uint32(obj.polyPhaseOrder);
            fpe = strcmp(obj.BoundaryOperation,'Circular');
            arrayCoefs = obj.atomCncFcn(arrayCoefs,subScale,pmCoefs,...
                [ps ps],ord,fpe);
            
            % Block IDCT
            if decY_ == 1 && decX_ == 1 && decZ_ == 1
                coefs = zeros(nDec,nRows_*nCols_*nLays_);
                coefs(1,:) = arrayCoefs(1,:);
                recImg = obj.col2vol_(coefs,blockSize,subScale);
            elseif decY_ == 2 && decX_ == 2 && decZ_ == 2
                recImg = zeros(2*subScale);
                subCoef1 = arrayCoefs(1,:);
                subCoef2 = arrayCoefs(2,:);
                subCoef3 = arrayCoefs(3,:);
                subCoef4 = arrayCoefs(4,:);
                subCoef5 = arrayCoefs(ps+1,:);
                subCoef6 = arrayCoefs(ps+2,:);
                subCoef7 = arrayCoefs(ps+3,:);
                subCoef8 = arrayCoefs(ps+4,:);
                %
                subCoef1p2 = subCoef1 + subCoef2;
                subCoef1m2 = subCoef1 - subCoef2;
                subCoef3p4 = subCoef3 + subCoef4;
                subCoef3m4 = subCoef3 - subCoef4;
                subCoef5p6 = subCoef5 + subCoef6;
                subCoef5m6 = subCoef5 - subCoef6;
                subCoef7p8 = subCoef7 + subCoef8;
                subCoef7m8 = subCoef7 - subCoef8;
                %
                subCoef1p2p3p4 = subCoef1p2 + subCoef3p4;
                subCoef1p2m3m4 = subCoef1p2 - subCoef3p4;
                subCoef1m2p3m4 = subCoef1m2 + subCoef3m4;
                subCoef1m2m3p4 = subCoef1m2 - subCoef3m4; 
                subCoef5p6p7p8 = subCoef5p6 + subCoef7p8;
                subCoef5p6m7m8 = subCoef5p6 - subCoef7p8;
                subCoef5m6p7m8 = subCoef5m6 + subCoef7m8;
                subCoef5m6m7p8 = subCoef5m6 - subCoef7m8;                                
                %
                recImg(1:2:end,1:2:end,1:2:end) = ...
                    reshape( subCoef1p2p3p4 + subCoef5p6p7p8, subScale );
                recImg(2:2:end,1:2:end,1:2:end) = ...
                    reshape( subCoef1p2m3m4 + subCoef5p6m7m8, subScale );
                recImg(1:2:end,2:2:end,1:2:end) = ...
                    reshape( subCoef1m2m3p4 + subCoef5m6m7p8, subScale );
                recImg(2:2:end,2:2:end,1:2:end) = ...
                    reshape( subCoef1m2p3m4 + subCoef5m6p7m8, subScale );
                recImg(1:2:end,1:2:end,2:2:end) = ...
                    reshape( subCoef1m2p3m4 - subCoef5m6p7m8, subScale );
                recImg(2:2:end,1:2:end,2:2:end) = ...
                    reshape( subCoef1m2m3p4 - subCoef5m6m7p8, subScale );
                recImg(1:2:end,2:2:end,2:2:end) = ...
                    reshape( subCoef1p2m3m4 - subCoef5p6m7m8, subScale );
                recImg(2:2:end,2:2:end,2:2:end) = ...
                    reshape( subCoef1p2p3p4 - subCoef5p6p7p8, subScale );
                %
                recImg = recImg/(2*sqrt(2));                    
            else 
                mc = ceil(nDec/2);
                mf = floor(nDec/2);
                coefs = zeros(nDec,nRows_*nCols_*nLays_);
                coefs(1:mc,:) = arrayCoefs(1:mc,:);
                coefs(mc+1:end,:) = arrayCoefs(ps+1:ps+mf,:);                 
                E0 = obj.getMatrixE0_(blockSize);
                coefs = E0.'*coefs;
                recImg=obj.col2vol_(coefs,blockSize,subScale);
            end
        end 
    end
    
    methods (Access = private, Static = true)
        
        function x = col2vol_(y,blockSize,subScale)
            import saivdr.dictionary.utility.Direction
            decY = blockSize(Direction.VERTICAL);
            decX = blockSize(Direction.HORIZONTAL);
            decZ = blockSize(Direction.DEPTH);
            nRows_ = subScale(Direction.VERTICAL);
            nCols_ = subScale(Direction.HORIZONTAL);
            nLays_ = subScale(Direction.DEPTH);
            
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
        
        function y = vol2col_(x,blockSize,subScale)
            import saivdr.dictionary.utility.Direction
            decY = blockSize(Direction.VERTICAL);
            decX = blockSize(Direction.HORIZONTAL);
            decZ = blockSize(Direction.DEPTH);
            nRows_ = subScale(Direction.VERTICAL);
            nCols_ = subScale(Direction.HORIZONTAL);
            nLays_ = subScale(Direction.DEPTH);
            
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
        
        function value = getMatrixE0_(blockSize)
            import saivdr.dictionary.utility.Direction
            decY_ = blockSize(Direction.VERTICAL);
            decX_ = blockSize(Direction.HORIZONTAL);
            decZ_ = blockSize(Direction.DEPTH);
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

