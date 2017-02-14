classdef CnsoltSynthesis2dSystem  < ...
        saivdr.dictionary.AbstSynthesisSystem %#~codegen
    %NsoltSynthesis2dSystem Synthesis system of Type-I NSOLT
    %
    % SVN identifier:
    % $Id: NsoltSynthesis2dSystem.m 683 2015-05-29 08:22:13Z sho $
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
        DATA_DIMENSION = 2
    end
    
    properties (Nontunable)
        LpPuFb2d
        BoundaryOperation = 'Termination'        
    end

    properties (Nontunable, PositiveInteger)    
        NumberOfChannels     = 2
        NumberOfHalfChannels = 2
    end
    
    properties (Nontunable, Logical)
        IsCloneLpPuFb2d = true;
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
    end 
    
    properties (Access = private, Logical)
        isMexFcn = false
    end
    
    methods
        
        % Constractor
        function obj = CnsoltSynthesis2dSystem(varargin)
            setProperties(obj,nargin,varargin{:})
            %
            if isempty(obj.LpPuFb2d)
                import saivdr.dictionary.cnsoltx.CnsoltFactory
                obj.LpPuFb2d = CnsoltFactory.createCplxOvsdLpPuFb2dSystem(...
                    'NumberOfChannels',obj.NumberOfChannels, ...
                    'NumberOfVanishingMoments',1,...
                    'OutputMode','ParameterMatrixSet');
            end            
            %
            if obj.IsCloneLpPuFb2d
                obj.LpPuFb2d = clone(obj.LpPuFb2d); 
            end
            %       
            if ~strcmp(get(obj.LpPuFb2d,'OutputMode'),'ParameterMatrixSet')
                release(obj.LpPuFb2d);
                set(obj.LpPuFb2d,'OutputMode','ParameterMatrixSet');
            end
            %
            obj.decimationFactor = get(obj.LpPuFb2d,'DecimationFactor');
            obj.polyPhaseOrder   = get(obj.LpPuFb2d,'PolyPhaseOrder');
            nch = get(obj.LpPuFb2d,'NumberOfChannels');
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
            s.LpPuFb2d = matlab.System.saveObject(obj.LpPuFb2d);
            
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
            obj.LpPuFb2d = matlab.System.loadObject(s.LpPuFb2d);            
        end
        
        function validatePropertiesImpl(~)
        end
        
        function setupImpl(obj, ~, ~)
            nch = obj.NumberOfChannels;
            
            % Prepare MEX function
            %TODO: MEXコード化を完了したら下の2行を削除する
            obj.isMexFcn = 1;
            mexFcn = [];
            
            if ~obj.isMexFcn
                import saivdr.dictionary.cnsoltx.mexsrcs.fcn_autobuild_catomcnc2d
                [mexFcn, obj.isMexFcn] = ...
                    fcn_autobuild_catomcnc2d(nch);
            end
            if ~isempty(mexFcn)
                obj.atomCncFcn = @(coefs,scale,pmcoefs,ord,fpe) ...
                    mexFcn(coefs,scale,pmcoefs,...
                    nch,ord,fpe);
            else
                import saivdr.dictionary.cnsoltx.mexsrcs.fcn_CnsoltAtomConcatenator2d
                clear fcn_CnsoltAtomConcatenator2d
                obj.atomCncFcn = @(coefs,scale,pmcoefs,ord,fpe) ...
                    fcn_CnsoltAtomConcatenator2d(coefs,scale,pmcoefs,...
                    nch,ord,fpe);
            end
            
        end
        
        function recImg = stepImpl(obj, coefs, scales)
            pmMtx = step(obj.LpPuFb2d,[],[]);
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
            arrayCoefs = zeros(nChs,height*width);
            arrayCoefs(1,:) = coefs(1:eIdx);
            for iLevel = 1:nLevels
                obj.nRows = uint32(height);
                obj.nCols = uint32(width);
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
                    arrayCoefs = zeros(nChs,height*width);
                    arrayCoefs(1,:) = subImg(:).';
                end
            end
            recImg = subImg;
        end
        
        function subImg = subSynthesize_(obj,arrayCoefs,pmCoefs,symmetry)
            import saivdr.dictionary.utility.Direction
            nRows_ = obj.nRows;
            nCols_ = obj.nCols;
            decY_ = obj.decimationFactor(Direction.VERTICAL);
            decX_ = obj.decimationFactor(Direction.HORIZONTAL);
            nDec = decY_*decX_;
            %
            blockSize = [ decY_ decX_ ];
            %
            if isinteger(arrayCoefs)
                arrayCoefs = double(arrayCoefs);
            end
            
            % Atom concatenation
            S = diag(exp(-1i*symmetry));
            subScale  = [ nRows_ nCols_ ];
            ord  = uint32(obj.polyPhaseOrder);            
            fpe = strcmp(obj.BoundaryOperation,'Circular');
            arrayCoefs = obj.atomCncFcn(S*arrayCoefs,subScale,pmCoefs,...
                ord,fpe);
            
            % Block IDFT
            if decY_ == 1 && decX_ == 1
                scale = double(subScale);
                coefs = zeros(nDec,nRows_*nCols_);
                coefs(1,:) = arrayCoefs(1,:);
                subImg = col2im(coefs,blockSize,scale,'distinct');
%             elseif decY_ == 2 && decX_ == 2
%                 subImg = zeros(2*subScale);
%                 subCoef1 = arrayCoefs(1,:);
%                 subCoef2 = arrayCoefs(2,:);
%                 subCoef3 = arrayCoefs(ps+1,:);
%                 subCoef4 = arrayCoefs(ps+2,:);
%                 %
%                 subImg(1:2:end,1:2:end) = ...
%                     reshape(subCoef1+subCoef2+subCoef3+subCoef4,subScale);
%                 subImg(2:2:end,1:2:end)  = ...
%                     reshape(subCoef1-subCoef2-subCoef3+subCoef4,subScale);
%                 subImg(1:2:end,2:2:end)  = ...
%                     reshape(subCoef1-subCoef2+subCoef3-subCoef4,subScale);
%                 subImg(2:2:end,2:2:end)  = ...
%                     reshape(subCoef1+subCoef2-subCoef3-subCoef4,subScale);
%                 %
%                 subImg = subImg/2;
            else
                coefs = arrayCoefs(1:nDec,:);
                scale = double(subScale) .* obj.decimationFactor;
                dctCoefs = col2im(coefs,blockSize,scale,'distinct');
                subImg = blockproc(dctCoefs,blockSize,...
                    @(x) saivdr.utility.HermitianSymmetricDFT.ihsdft2(x.data));
            end
        end
        
    end

    
end

