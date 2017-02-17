classdef CnsoltSynthesis3dSystem  < ...
        saivdr.dictionary.AbstSynthesisSystem %#~codegen
    %NsoltSynthesis3dSystem Synthesis system of Type-I NSOLT
    %
    % SVN identifier:
    % $Id: NsoltSynthesis3dSystem.m 683 2015-05-29 08:22:13Z sho $
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
        IsDifferentiation = false
    end

    properties (PositiveInteger)    
        IndexOfDifferentiationAngle = 1
    end    
    
    properties (Hidden, Transient)
        BoundaryOperationSet = ...
            matlab.system.StringSet({'Termination','Circular'});
    end
    
    properties (Access = private, Nontunable)
        decimationFactor
        polyPhaseOrder
    end    

    properties (Access = private)
        atomCncFcn
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
        
        % Constractor
        function obj = CnsoltSynthesis3dSystem(varargin)
            setProperties(obj,nargin,varargin{:})
            %
            if isempty(obj.LpPuFb3d)
                import saivdr.dictionary.cnsoltx.CnsoltFactory
                obj.LpPuFb3d = CnsoltFactory.createCplxOvsdLpPuFb3dSystem(...
                    'NumberOfChannels',obj.NumberOfChannels, ...
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
            %
            obj.FrameBound = 1;            
        end
        
    end
    
    methods (Access=protected)
        
        function s = saveObjectImpl(obj)
            % Call the base class method
            s = saveObjectImpl@saivdr.dictionary.AbstSynthesisSystem(obj);
            % Save the child System objects            
            s.LpPuFb3d = matlab.System.saveObject(obj.LpPuFb3d);
            
            % Save the protected & private properties
            s.atomCncFcn       = obj.atomCncFcn;            
            s.decimationFactor = obj.decimationFactor;
            s.polyPhaseOrder   = obj.polyPhaseOrder;
            %s.nRows            = obj.nRows;
            %s.nCols            = obj.nCols;
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            % Load protected and private properties            
            obj.atomCncFcn       = s.atomCncFcn;
            obj.decimationFactor = s.decimationFactor;
            obj.polyPhaseOrder   = s.polyPhaseOrder;
            %obj.nRows            = s.nRows;
            %obj.nCols            = s.nCols;            
            % Call base class method to load public properties            
            loadObjectImpl@saivdr.dictionary.AbstSynthesisSystem(obj,s,wasLocked);
            % Load the child System objects            
            obj.LpPuFb3d = matlab.System.loadObject(s.LpPuFb3d);            
        end
        
        function validatePropertiesImpl(~)
        end
        
        function validateInputsImpl(~,~,scales)
            id = 'SaivDr:InvalidArgumentException';
            if size(scales,2) ~= 3
                error('%s \n\t The number of column of scales should be three.',...
                    id);
            end
        end        
        
        function setupImpl(obj, ~, ~)
            nch = obj.NumberOfChannels;
            
            % Prepare MEX function
            if ~obj.isMexFcn
                import saivdr.dictionary.cnsoltx.mexsrcs.fcn_autobuild_catomcnc3d
                [mexFcn, obj.isMexFcn] = ...
                    fcn_autobuild_catomcnc3d(nch);
            end
            if ~isempty(mexFcn)
                obj.atomCncFcn = @(coefs,scale,pmcoefs,ord,fpe) ...
                    mexFcn(coefs,scale,pmcoefs,...
                    nch,ord,fpe);
            else
                import saivdr.dictionary.cnsoltx.mexsrcs.fcn_CnsoltAtomConcatenator3d
                clear fcn_CnsoltAtomConcatenator3d
                obj.atomCncFcn = @(coefs,scale,pmcoefs,ord,fpe) ...
                    fcn_CnsoltAtomConcatenator3d(coefs,scale,pmcoefs,...
                    nch,ord,fpe);
            end
            
        end
        
        function recImg = stepImpl(obj, coefs, scales)
            pmMtx = step(obj.LpPuFb3d,[],[]);
            pmMtxCoefs = get(pmMtx,'Coefficients');
            symmetry = get(obj.LpPuFb2d,'Symmetry');
            recImg = synthesize_(obj, coefs, scales, pmMtxCoefs,symmetry);
        end
        
    end
    
    methods (Access = private)
        
        function recImg = synthesize_(obj,coefs,scales,pmCoefs,symmetry)
            import saivdr.dictionary.utility.Direction
            %
            nChs = obj.NumberOfChannels;
            nLevels = (size(scales,1)-1)/(nChs-1);
            %
            iSubband = 1;
            eIdx = prod(scales(iSubband,:));
            %
            height = scales(1,1);
            width  = scales(1,2);
            depth  = scales(1,3);
            arrayCoefs = complex(zeros(nChs,height*width*depth));
            arrayCoefs(1,:) = coefs(1:eIdx);
            for iLevel = 1:nLevels
                obj.nRows = uint32(height);
                obj.nCols = uint32(width);
                obj.nLays = uint32(depth);
                for iCh = 2:nChs
                    iSubband = iSubband + 1;
                    sIdx = eIdx + 1;
                    eIdx = sIdx + prod(scales(iSubband,:))-1;
                    arrayCoefs(iCh,:) = coefs(sIdx:eIdx);
                end
                subImg = subSynthesize_(obj,arrayCoefs,pmCoefs,symmetry);
                if iLevel < nLevels
                    height = size(subImg,1);
                    width  = size(subImg,2);                    
                    depth  = size(subImg,3);                    
                    arrayCoefs = zeros(nChs,height*width*depth);
                    arrayCoefs(1,:) = subImg(:).';
                end
            end
            recImg = subImg;
        end
        
        function subImg = subSynthesize_(obj,arrayCoefs,pmCoefs,symmetry)
            import saivdr.dictionary.utility.Direction
            %
            ps = obj.NumberOfSymmetricChannels;
            nRows_ = obj.nRows;
            nCols_ = obj.nCols;
            nLays_ = obj.nLays;
            decY_ = obj.decimationFactor(Direction.VERTICAL);
            decX_ = obj.decimationFactor(Direction.HORIZONTAL);
            decZ_ = obj.decimationFactor(Direction.DEPTH);
            nDec = decY_*decX_*decZ_;
            %
            if isinteger(arrayCoefs)
                arrayCoefs = double(arrayCoefs);
            end
            
            % Atom concatenation
            S = diag(exp(-1i*symmetry));
            subScale  = [ nRows_ nCols_ nLays_ ];
            ord  = uint32(obj.polyPhaseOrder);            
            fpe = strcmp(obj.BoundaryOperation,'Circular');         
            arrayCoefs = obj.atomCncFcn(S*arrayCoefs,subScale,pmCoefs,...
                ord,fpe);
            
            % Block IDCT
            if decY_ == 1 && decX_ == 1 && decZ_ == 1
                coefs = complex(zeros(nDec,nRows_*nCols_*nLays_));
                coefs(1,:) = arrayCoefs(1,:);
                subImg = col2vol_(obj,coefs);
%             elseif decY_ == 2 && decX_ == 2 && decZ_ == 2
%                 subImg = zeros(2*subScale);
%                 subCoef1 = arrayCoefs(1,:);
%                 subCoef2 = arrayCoefs(2,:);
%                 subCoef3 = arrayCoefs(3,:);
%                 subCoef4 = arrayCoefs(4,:);
%                 subCoef5 = arrayCoefs(ps+1,:);
%                 subCoef6 = arrayCoefs(ps+2,:);
%                 subCoef7 = arrayCoefs(ps+3,:);
%                 subCoef8 = arrayCoefs(ps+4,:);
%                 %
%                 subCoef1p2 = subCoef1 + subCoef2;
%                 subCoef1m2 = subCoef1 - subCoef2;
%                 subCoef3p4 = subCoef3 + subCoef4;
%                 subCoef3m4 = subCoef3 - subCoef4;
%                 subCoef5p6 = subCoef5 + subCoef6;
%                 subCoef5m6 = subCoef5 - subCoef6;
%                 subCoef7p8 = subCoef7 + subCoef8;
%                 subCoef7m8 = subCoef7 - subCoef8;
%                 %
%                 subCoef1p2p3p4 = subCoef1p2 + subCoef3p4;
%                 subCoef1p2m3m4 = subCoef1p2 - subCoef3p4;
%                 subCoef1m2p3m4 = subCoef1m2 + subCoef3m4;
%                 subCoef1m2m3p4 = subCoef1m2 - subCoef3m4; 
%                 subCoef5p6p7p8 = subCoef5p6 + subCoef7p8;
%                 subCoef5p6m7m8 = subCoef5p6 - subCoef7p8;
%                 subCoef5m6p7m8 = subCoef5m6 + subCoef7m8;
%                 subCoef5m6m7p8 = subCoef5m6 - subCoef7m8;                                
%                 %
%                 subImg(1:2:end,1:2:end,1:2:end) = ...
%                     reshape( subCoef1p2p3p4 + subCoef5p6p7p8, subScale );
%                 subImg(2:2:end,1:2:end,1:2:end) = ...
%                     reshape( subCoef1p2m3m4 + subCoef5p6m7m8, subScale );
%                 subImg(1:2:end,2:2:end,1:2:end) = ...
%                     reshape( subCoef1m2m3p4 + subCoef5m6m7p8, subScale );
%                 subImg(2:2:end,2:2:end,1:2:end) = ...
%                     reshape( subCoef1m2p3m4 + subCoef5m6p7m8, subScale );
%                 subImg(1:2:end,1:2:end,2:2:end) = ...
%                     reshape( subCoef1m2p3m4 - subCoef5m6p7m8, subScale );
%                 subImg(2:2:end,1:2:end,2:2:end) = ...
%                     reshape( subCoef1m2m3p4 - subCoef5m6m7p8, subScale );
%                 subImg(1:2:end,2:2:end,2:2:end) = ...
%                     reshape( subCoef1p2m3m4 - subCoef5p6m7m8, subScale );
%                 subImg(2:2:end,2:2:end,2:2:end) = ...
%                     reshape( subCoef1p2p3p4 - subCoef5p6p7p8, subScale );
%                 %
%                 subImg = subImg/(2*sqrt(2));                    
            else 
                coefs = arrayCoefs(1:nDec,:);                
                E0 = getMatrixE0_(obj);
                coefs = E0.'*coefs;
                subImg=col2vol_(obj,coefs);
            end
        end
        
        function x = col2vol_(obj,y)
            import saivdr.dictionary.utility.Direction
            decY = obj.decimationFactor(Direction.VERTICAL);
            decX = obj.decimationFactor(Direction.HORIZONTAL);
            decZ = obj.decimationFactor(Direction.DEPTH);
            nRows_ = obj.nRows;
            nCols_ = obj.nCols;
            nLays_ = obj.nLays;
            
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
        
        function value = getMatrixE0_(obj)
            import saivdr.dictionary.utility.Direction
            import saivdr.utility.HermitianSymmetricDFT
            decY_ = obj.DecimationFactor(Direction.VERTICAL);
            decX_ = obj.DecimationFactor(Direction.HORIZONTAL);
            decZ_ = obj.DecimationFactor(Direction.DEPTH);
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

