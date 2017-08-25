classdef CnsoltAnalysis3dSystem < ...
        saivdr.dictionary.AbstAnalysisSystem %#~codegen
    %NSOLTANALYSISSYSTEM Abstract class of NSOLT analysis system
    %
    % SVN identifier:
    % $Id: NsoltAnalysis3dSystem.m 683 2015-05-29 08:22:13Z sho $
    %
    % Requirements: MATLAB R2013b
    %
    % Copyright (c) 2014-2015, Shogo MURAMATSU
    %
    % All rights reserved.
    %
    % Contact address: Shogo MURAMATSU,
    %                Faculty of Engineering, Niigata University,
    %                8050 2-no-cho Ikarashi, Nishi-ku,
    %                Niigata, 950-2181, JAPAN
    %
    % LinedIn: http://www.linkedin.com/pub/shogo-muramatsu/4b/b08/627    
    %
    
    properties (Access = protected, Constant = true)
        DATA_DIMENSION = 3
    end
    
    properties (Nontunable)
        LpPuFb3d
        BoundaryOperation = 'Termination'
    end

    properties (Nontunable, PositiveInteger)    
        NumberOfChannels     = 8
        NumberOfHalfChannels = 4
    end
    
    properties (Nontunable, Logical)
        IsCloneLpPuFb3d = true;
    end
    
    properties (Hidden, Transient)
        BoundaryOperationSet = ...
            matlab.system.StringSet({'Termination','Circular'});
    end
    
    properties (Access = private, Nontunable)
        nAllCoefs
        nAllChs
        decimationFactor
        polyPhaseOrder
    end

    properties (Access = private)
        fcnAtomExt
        allScales
        allCoefs
    end
    
    properties (Access = private, PositiveInteger)
        nRows
        nCols
        nLays
    end
    
    properties (Access = private, Logical)
        isMexFcn = false
    end
    
    methods
        
        % Constructor
        function obj = CnsoltAnalysis3dSystem(varargin)
            setProperties(obj,nargin,varargin{:});
            %
            if isempty(obj.LpPuFb3d)
                import saivdr.dictionary.cnsoltx.CnsoltFactory
                obj.LpPuFb3d = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                    'NumberOfChannels',obj.NumberOfChannels,...
                    'NumberOfVanishingMoments',1,...
                    'OutputMode','ParameterMatrixSet');
            end
            %
            if obj.IsCloneLpPuFb3d
                obj.LpPuFb3d = clone(obj.LpPuFb3d); 
            end
            %
            if ~strcmp(get(obj.LpPuFb3d,'OutputMode'),'ParameterMatrixSet')
                release(obj.LpPuFb3d);
                set(obj.LpPuFb3d,'OutputMode','ParameterMatrixSet');            
            end
            %
            obj.decimationFactor = get(obj.LpPuFb3d,'DecimationFactor');
            obj.polyPhaseOrder   = get(obj.LpPuFb3d,'PolyPhaseOrder');
            nch = get(obj.LpPuFb3d,'NumberOfChannels');
            obj.NumberOfChannels = nch;
        end
        
    end
    
    methods (Access=protected)
        
        function s = saveObjectImpl(obj)
            % Call the base class method
            s = saveObjectImpl@saivdr.dictionary.AbstAnalysisSystem(obj);
            % Save the child System objects            
            s.LpPuFb3d = matlab.System.saveObject(obj.LpPuFb3d);
            
            % Save the protected & private properties
            s.fcnAtomExt = obj.fcnAtomExt;            
            s.nAllCoefs  = obj.nAllCoefs;
            s.nAllChs    = obj.nAllChs;
            s.decimationFactor = obj.decimationFactor;
            s.polyPhaseOrder   = obj.polyPhaseOrder;
            s.allScales  = obj.allScales;
            s.allCoefs   = obj.allCoefs;
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            % Load protected and private properties
            obj.fcnAtomExt = s.fcnAtomExt;
            obj.nAllCoefs  = s.nAllCoefs;
            obj.nAllChs    = s.nAllChs;
            obj.decimationFactor = s.decimationFactor;
            obj.polyPhaseOrder   = s.polyPhaseOrder;
            obj.allScales   = s.allScales;
            obj.allCoefs    = s.allCoefs;
            
            % Call base class method to load public properties
            loadObjectImpl@saivdr.dictionary.AbstAnalysisSystem(obj,s,wasLocked);
            % Load the child System objects            
            obj.LpPuFb3d = matlab.System.loadObject(s.LpPuFb3d);
        end
        
        function setupImpl(obj,srcImg,nLevels)
            dec = obj.decimationFactor;
            nch = obj.NumberOfChannels;
            %
            nChs  = sum(nch);
            nDec = prod(dec);
            %
            if nDec == 1
                obj.nAllCoefs = numel(srcImg)*(...
                    (nChs-1)*(nLevels/nDec) + 1/nDec^nLevels);
            else
                obj.nAllCoefs = numel(srcImg)*(...
                    (nChs-1)*(nDec^nLevels-1)/(nDec^nLevels*(nDec-1))  ...
                    + 1/nDec^nLevels);
            end
            obj.nAllChs = nLevels*(nChs-1)+1;            
            obj.allCoefs  = zeros(1,obj.nAllCoefs);
            obj.allScales = zeros(obj.nAllChs,obj.DATA_DIMENSION);
            
            % Prepare MEX function
            if exist('fcn_CnsoltAtomExtender3dCodeGen_mex','file')==3
                obj.fcnAtomExt = @fcn_CnsoltAtomExtender3dCodeGen_mex;
            else
                import saivdr.dictionary.cnsoltx.mexsrcs.fcn_CnsoltAtomExtender3dCodeGen
                obj.fcnAtomExt = @fcn_CnsoltAtomExtender3dCodeGen;
            end
%             if ~obj.isMexFcn
%                 import saivdr.dictionary.cnsoltx.mexsrcs.fcn_autobuild_catomext3d
%                 [mexFcn, obj.isMexFcn] = ...
%                     fcn_autobuild_catomext3d(nch);
%             end
%             if ~isempty(mexFcn)
%                 obj.fcnAtomExt = @(coefs,scale,pmcoefs,ord,fpe) ...
%                     mexFcn(coefs,scale,pmcoefs,...
%                     nch,ord,fpe);
%             else
%                 import saivdr.dictionary.cnsoltx.mexsrcs.fcn_CnsoltAtomExtender3d
%                 clear fcn_CnsoltAtomExtender3d
%                 obj.fcnAtomExt = @(coefs,scale,pmcoefs,ord,fpe) ...
%                     fcn_CnsoltAtomExtender3d(coefs,scale,pmcoefs,...
%                     nch,ord,fpe);
%             end

        end
        
        function [ coefs, scales ] = stepImpl(obj, srcImg, nLevels)
            pmMtx = step(obj.LpPuFb3d,[],[]);
            pmMtxCoefs = get(pmMtx,'Coefficients');
            symmetry = get(obj.LpPuFb3d,'Symmetry');
            [ coefs, scales ] = analyze_(obj, srcImg, nLevels, pmMtxCoefs, symmetry);
        end
        
    end
    
    methods (Access = private)
        
        function [ coefs, scales ] = ...
                analyze_(obj, srcImg, nLevels, pmCoefs, symmetry)
            import saivdr.dictionary.utility.Direction            
            %
            nChs = obj.NumberOfChannels;
            decY = obj.decimationFactor(Direction.VERTICAL);
            decX = obj.decimationFactor(Direction.HORIZONTAL);   
            decZ = obj.decimationFactor(Direction.DEPTH);   
            %
            iSubband = obj.nAllChs;
            eIdx     = obj.nAllCoefs;
            %
            subImg = srcImg;
            for iLevel = 1:nLevels
                height = size(subImg,1);
                width  = size(subImg,2);
                depth  = size(subImg,3);
                obj.nRows = uint32(height/decY);
                obj.nCols = uint32(width/decX);
                obj.nLays = uint32(depth/decZ);                
                arrayCoefs = subAnalyze_(obj,subImg,pmCoefs,symmetry);
                for iCh = nChs:-1:2
                    subbandCoefs = arrayCoefs(iCh,:);
                    obj.allScales(iSubband,:) = [ obj.nRows obj.nCols obj.nLays];
                    sIdx = eIdx - (obj.nRows*obj.nCols*obj.nLays) + 1;
                    obj.allCoefs(sIdx:eIdx) = subbandCoefs(:).';
                    iSubband = iSubband-1;
                    eIdx = sIdx - 1;
                end
                subImg = reshape(arrayCoefs(1,:),obj.nRows,obj.nCols,obj.nLays);
            end
            obj.allScales(1,:) = [ obj.nRows obj.nCols obj.nLays];
            obj.allCoefs(1:eIdx) = subImg(:).';
            %
            scales = obj.allScales;
            coefs  = obj.allCoefs;
        end
        
        function arrayCoefs = subAnalyze_(obj,subImg,pmCoefs,symmetry)
            import saivdr.dictionary.utility.Direction
           %
            nChs = obj.NumberOfChannels;
            nRows_ = obj.nRows;
            nCols_ = obj.nCols;
            nLays_ = obj.nLays;
            decY_  = obj.decimationFactor(Direction.VERTICAL);
            decX_  = obj.decimationFactor(Direction.HORIZONTAL);
            decZ_  = obj.decimationFactor(Direction.DEPTH);
            %
            if isinteger(subImg)
                subImg = im2double(subImg);
            end
            
            % Prepare array
            arrayCoefs = complex(zeros(nChs,nRows_*nCols_*nLays_));
            
            % Block DCT
            if decY_ == 1 && decX_ == 1 && decZ_ == 1
                coefs = vol2col_(obj,subImg);
                arrayCoefs(1,:) = coefs(1,:);
%             elseif decY_ == 2 && decX_ == 2 && decZ_ == 2
%                 subImg1 = subImg(1:2:end,1:2:end,1:2:end);
%                 subImg2 = subImg(2:2:end,1:2:end,1:2:end);
%                 subImg3 = subImg(1:2:end,2:2:end,1:2:end);
%                 subImg4 = subImg(2:2:end,2:2:end,1:2:end);
%                 subImg5 = subImg(1:2:end,1:2:end,2:2:end);
%                 subImg6 = subImg(2:2:end,1:2:end,2:2:end);
%                 subImg7 = subImg(1:2:end,2:2:end,2:2:end);
%                 subImg8 = subImg(2:2:end,2:2:end,2:2:end);
%                 %
%                 subImg1 = subImg1(:).';
%                 subImg2 = subImg2(:).';
%                 subImg3 = subImg3(:).';
%                 subImg4 = subImg4(:).';
%                 subImg5 = subImg5(:).';
%                 subImg6 = subImg6(:).';
%                 subImg7 = subImg7(:).';
%                 subImg8 = subImg8(:).';
%                 %
%                 subImg1p2 = subImg1 + subImg2;
%                 subImg1m2 = subImg1 - subImg2;
%                 subImg3p4 = subImg3 + subImg4;
%                 subImg3m4 = subImg3 - subImg4;                
%                 subImg5p6 = subImg5 + subImg6;
%                 subImg5m6 = subImg5 - subImg6;                                
%                 subImg7p8 = subImg7 + subImg8;
%                 subImg7m8 = subImg7 - subImg8;        
%                 %
%                 subImg1p2p3p4 = subImg1p2 + subImg3p4;
%                 subImg1p2m3m4 = subImg1p2 - subImg3p4;
%                 subImg1m2p3m4 = subImg1m2 + subImg3m4;
%                 subImg1m2m3p4 = subImg1m2 - subImg3m4;                
%                 subImg5p6p7p8 = subImg5p6 + subImg7p8;
%                 subImg5p6m7m8 = subImg5p6 - subImg7p8;
%                 subImg5m6p7m8 = subImg5m6 + subImg7m8;
%                 subImg5m6m7p8 = subImg5m6 - subImg7m8;                                
%                 %
%                 arrayCoefs(1,:)    = subImg1p2p3p4 + subImg5p6p7p8;
%                 arrayCoefs(2,:)    = subImg1p2m3m4 - subImg5p6m7m8;
%                 arrayCoefs(3,:)    = subImg1m2m3p4 + subImg5m6m7p8;
%                 arrayCoefs(4,:)    = subImg1m2p3m4 - subImg5m6p7m8;
%                 arrayCoefs(ps+1,:) = subImg1p2p3p4 - subImg5p6p7p8;
%                 arrayCoefs(ps+2,:) = subImg1p2m3m4 + subImg5p6m7m8;
%                 arrayCoefs(ps+3,:) = subImg1m2m3p4 - subImg5m6m7p8;
%                 arrayCoefs(ps+4,:) = subImg1m2p3m4 + subImg5m6p7m8;
%                 %
%                 arrayCoefs = arrayCoefs/(2*sqrt(2));
            else 
                nDec_=decY_*decX_*decZ_;
                coefs = vol2col_(obj,subImg);
                E0 = getMatrixE0_(obj);
                coefs = conj(E0)*coefs;
                arrayCoefs(1:nDec_,:) = coefs;
            end
            
            % Atom extension
            S = diag(exp(1i*symmetry));
            subScale = [ obj.nRows obj.nCols obj.nLays];
            nch = obj.NumberOfChannels;
            ord   = uint32(obj.polyPhaseOrder);            
            fpe = strcmp(obj.BoundaryOperation,'Circular');
             arrayCoefs = S*obj.fcnAtomExt(arrayCoefs,subScale,pmCoefs,...
                 nch,ord,fpe);
        end        

        function y = vol2col_(obj,x)
            import saivdr.dictionary.utility.Direction
            decY = obj.decimationFactor(Direction.VERTICAL);
            decX = obj.decimationFactor(Direction.HORIZONTAL);
            decZ = obj.decimationFactor(Direction.DEPTH);
            nRows_ = obj.nRows;
            nCols_ = obj.nCols;
            nLays_ = obj.nLays;
            
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
        
        function value = getMatrixE0_(obj)
            import saivdr.dictionary.utility.Direction
            import saivdr.utility.HermitianSymmetricDFT
            decY_ = obj.decimationFactor(Direction.VERTICAL);
            decX_ = obj.decimationFactor(Direction.HORIZONTAL);
            decZ_ = obj.decimationFactor(Direction.DEPTH);
            nElmBi = decY_*decX_*decZ_;
            coefs = complex(zeros(nElmBi));
            iElm = 1;
            for iRow = 1:decY_
                for iCol = 1:decX_
                    hsdftCoefYX = complex(zeros(decY_,decX_));
                    hsdftCoefYX(iRow,iCol) = 1;
                    basisYX = HermitianSymmetricDFT.ihsdft2(hsdftCoefYX);
                    for iDep = 1:decZ_
                        hsdftCoefZ = zeros(decZ_,1);
                        hsdftCoefZ(iDep) = 1;
                        %basisZ  = permute(idct(hsdftCoefZ),[2 3 1]);
                        basisZ = permute(HermitianSymmetricDFT.ihsdft(hsdftCoefZ),[2 3 1]);
                        basisVd = convn(basisZ,basisYX);
                        coefs(iElm,:) = basisVd(:).';
                        iElm = iElm + 1;
                    end
                end
            end
            %
            value = flip(coefs,2);
        end
        
%         function value = getMatrixE0_(obj)
%             import saivdr.dictionary.utility.Direction
%             decY_ = obj.decimationFactor(Direction.VERTICAL);
%             decX_ = obj.decimationFactor(Direction.HORIZONTAL);
%             decZ_ = obj.decimationFactor(Direction.DEPTH);
%             nElmBi = decY_*decX_*decZ_;
%             coefs = zeros(nElmBi);
%             iElm = 1;
%             % E0.'= [ Beee Beoo Booe Boeo Beeo Beoe Booo Boee ] % Byxz
%             % Beee
%             for iRow = 1:2:decY_ % y-e
%                 for iCol = 1:2:decX_ % x-e
%                     dctCoefYX = zeros(decY_,decX_);
%                     dctCoefYX(iRow,iCol) = 1;
%                     basisYX = idct2(dctCoefYX);
%                     for iDep = 1:2:decZ_ % z-e
%                         dctCoefZ = zeros(decZ_,1);
%                         dctCoefZ(iDep) = 1;
%                         basisZ  = permute(idct(dctCoefZ),[2 3 1]);
%                         basisVd = convn(basisZ,basisYX);
%                         coefs(iElm,:) = basisVd(:).';
%                         iElm = iElm + 1;
%                     end
%                 end
%             end
%             % Beoo
%             for iRow = 1:2:decY_ % y-e
%                 for iCol = 2:2:decX_ % x-o
%                     dctCoefYX = zeros(decY_,decX_);
%                     dctCoefYX(iRow,iCol) = 1;
%                     basisYX = idct2(dctCoefYX);
%                     for iDep = 2:2:decZ_ % z-o
%                         dctCoefZ = zeros(decZ_,1);
%                         dctCoefZ(iDep) = 1;
%                         basisZ  = permute(idct(dctCoefZ),[2 3 1]);
%                         basisVd = convn(basisZ,basisYX);
%                         coefs(iElm,:) = basisVd(:).';
%                         iElm = iElm + 1;
%                     end
%                 end
%             end
%             %Booe
%             for iRow = 2:2:decY_ % y-o
%                 for iCol = 2:2:decX_ % x-o
%                     dctCoefYX = zeros(decY_,decX_);
%                     dctCoefYX(iRow,iCol) = 1;
%                     basisYX = idct2(dctCoefYX);
%                     for iDep = 1:2:decZ_ % z-e
%                         dctCoefZ = zeros(decZ_,1);
%                         dctCoefZ(iDep) = 1;
%                         basisZ  = permute(idct(dctCoefZ),[2 3 1]);
%                         basisVd = convn(basisZ,basisYX);
%                         coefs(iElm,:) = basisVd(:).';
%                         iElm = iElm + 1;
%                     end
%                 end
%             end
%             %Boeo
%             for iRow = 2:2:decY_ % y-o
%                 for iCol = 1:2:decX_ % x-e
%                     dctCoefYX = zeros(decY_,decX_);
%                     dctCoefYX(iRow,iCol) = 1;
%                     basisYX = idct2(dctCoefYX);
%                     for iDep = 2:2:decZ_ % z-o
%                         dctCoefZ = zeros(decZ_,1);
%                         dctCoefZ(iDep) = 1;
%                         basisZ  = permute(idct(dctCoefZ),[2 3 1]);
%                         basisVd = convn(basisZ,basisYX);
%                         coefs(iElm,:) = basisVd(:).';
%                         iElm = iElm + 1;
%                     end
%                 end
%             end
%             %Beeo
%             for iRow = 1:2:decY_ % y-e
%                 for iCol = 1:2:decX_ % x-e
%                     dctCoefYX = zeros(decY_,decX_);
%                     dctCoefYX(iRow,iCol) = 1;
%                     basisYX = idct2(dctCoefYX);
%                     for iDep = 2:2:decZ_ % z-o
%                         dctCoefZ = zeros(decZ_,1);
%                         dctCoefZ(iDep) = 1;
%                         basisZ  = permute(idct(dctCoefZ),[2 3 1]);
%                         basisVd = convn(basisZ,basisYX);
%                         coefs(iElm,:) = basisVd(:).';
%                         iElm = iElm + 1;
%                     end
%                 end
%             end
%             %Beoe
%             for iRow = 1:2:decY_ % y-e
%                 for iCol = 2:2:decX_ % x-o
%                     dctCoefYX = zeros(decY_,decX_);
%                     dctCoefYX(iRow,iCol) = 1;
%                     basisYX = idct2(dctCoefYX);
%                     for iDep = 1:2:decZ_ % z-e
%                         dctCoefZ = zeros(decZ_,1);
%                         dctCoefZ(iDep) = 1;
%                         basisZ  = permute(idct(dctCoefZ),[2 3 1]);
%                         basisVd = convn(basisZ,basisYX);
%                         coefs(iElm,:) = basisVd(:).';
%                         iElm = iElm + 1;
%                     end
%                 end
%             end
%             %Booo
%             for iRow = 2:2:decY_ % y-o
%                 for iCol = 2:2:decX_ % x-o
%                     dctCoefYX = zeros(decY_,decX_);
%                     dctCoefYX(iRow,iCol) = 1;
%                     basisYX = idct2(dctCoefYX);
%                     for iDep = 2:2:decZ_ % z-o
%                         dctCoefZ = zeros(decZ_,1);
%                         dctCoefZ(iDep) = 1;
%                         basisZ  = permute(idct(dctCoefZ),[2 3 1]);
%                         basisVd = convn(basisZ,basisYX);
%                         coefs(iElm,:) = basisVd(:).';
%                         iElm = iElm + 1;
%                     end
%                 end
%             end
%             %Boee
%             for iRow = 2:2:decY_ % y-o
%                 for iCol = 1:2:decX_ % x-e
%                     dctCoefYX = zeros(decY_,decX_);
%                     dctCoefYX(iRow,iCol) = 1;
%                     basisYX = idct2(dctCoefYX);
%                     for iDep = 1:2:decZ_ % z-e
%                         dctCoefZ = zeros(decZ_,1);
%                         dctCoefZ(iDep) = 1;
%                         basisZ  = permute(idct(dctCoefZ),[2 3 1]);
%                         basisVd = convn(basisZ,basisYX);
%                         coefs(iElm,:) = basisVd(:).';
%                         iElm = iElm + 1;
%                     end
%                 end
%             end
%             %
%             value = coefs;
%         end
        
    end
    
end
