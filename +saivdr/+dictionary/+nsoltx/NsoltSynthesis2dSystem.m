classdef NsoltSynthesis2dSystem  < ...
        saivdr.dictionary.AbstSynthesisSystem %#~codegen
    %NsoltSynthesis2dSystem Synthesis system of Type-I NSOLT
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2014-2020, Shogo MURAMATSU
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
        DATA_DIMENSION = 2
    end
    
    properties (Nontunable)
        LpPuFb2d
        BoundaryOperation = 'Termination'        
    end

    properties (Nontunable, PositiveInteger)    
        NumberOfSymmetricChannels     = 2
        NumberOfAntisymmetricChannels = 2
    end
    
    properties (Nontunable, Logical)
        IsCloneLpPuFb = true;
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
        fcnAtomCnc
    end
    
    properties (Access = private, PositiveInteger)
        nRows 
        nCols 
    end 
    
    methods
        
        % Constractor
        function obj = NsoltSynthesis2dSystem(varargin)
            setProperties(obj,nargin,varargin{:})
            %
            if isempty(obj.LpPuFb2d)
                import saivdr.dictionary.nsoltx.NsoltFactory
                obj.LpPuFb2d = NsoltFactory.createOvsdLpPuFb2dSystem(...
                    'NumberOfChannels', ...
                    [ obj.NumberOfSymmetricChannels ...
                      obj.NumberOfAntisymmetricChannels ], ...
                    'NumberOfVanishingMoments',1,...
                    'OutputMode','ParameterMatrixSet');
            end            
            %
            if obj.IsCloneLpPuFb
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
            obj.NumberOfSymmetricChannels = nch(1);
            obj.NumberOfAntisymmetricChannels = nch(2);
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
            s.decimationFactor = obj.decimationFactor;
            s.polyPhaseOrder   = obj.polyPhaseOrder;
            s.fcnAtomCnc       = obj.fcnAtomCnc;                          
            %s.nRows            = obj.nRows;            
            %s.nCols            = obj.nCols;
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            % Load protected and private properties            
            obj.decimationFactor = s.decimationFactor;
            obj.polyPhaseOrder   = s.polyPhaseOrder;
            obj.fcnAtomCnc       = s.fcnAtomCnc;
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
            if exist('fcn_NsoltAtomConcatenator2dCodeGen_mex','file')==3
                obj.fcnAtomCnc = @fcn_NsoltAtomConcatenator2dCodeGen_mex;
            else
                import saivdr.dictionary.nsoltx.mexsrcs.fcn_NsoltAtomConcatenator2dCodeGen
                obj.fcnAtomCnc = @fcn_NsoltAtomConcatenator2dCodeGen;
            end
        end
        
        function recImg = stepImpl(obj, coefs, scales)
            pmMtx = step(obj.LpPuFb2d,[],[]);
            pmMtxCoefs = get(pmMtx,'Coefficients');
            recImg = synthesize_(obj, coefs, scales, pmMtxCoefs);
        end
        
    end
    
    methods (Access = private)
        
        function recImg = synthesize_(obj,coefs,scales,pmCoefs)
            import saivdr.dictionary.utility.Direction
            %
            nChs = obj.NumberOfSymmetricChannels ...
                + obj.NumberOfAntisymmetricChannels;
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
                subImg = subSynthesize_(obj,arrayCoefs,pmCoefs);
                if iLevel < nLevels
                    height = size(subImg,1);
                    width  = size(subImg,2);                    
                    arrayCoefs = zeros(nChs,height*width);
                    arrayCoefs(1,:) = subImg(:).';
                end
            end
            recImg = subImg;
        end
        
        function subImg = subSynthesize_(obj,arrayCoefs,pmCoefs)
            import saivdr.dictionary.utility.Direction
            %
            ps = obj.NumberOfSymmetricChannels;
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
            subScale  = [ nRows_ nCols_ ];
            %arrayCoefs = obj.atomCncFcn(arrayCoefs,subScale,pmCoefs,...
            %    ord,fpe);
            nch = [ obj.NumberOfSymmetricChannels ...
                obj.NumberOfAntisymmetricChannels ];
            ord = uint32(obj.polyPhaseOrder);
            fpe = strcmp(obj.BoundaryOperation,'Circular');
            arrayCoefs = obj.fcnAtomCnc(...
                arrayCoefs, subScale, pmCoefs, nch, ord, fpe);
            
            % Block IDCT
            if decY_ == 1 && decX_ == 1
                scale = double(subScale);
                coefs = zeros(nDec,nRows_*nCols_);
                coefs(1,:) = arrayCoefs(1,:);
                subImg = col2im(coefs,blockSize,scale,'distinct');
            elseif decY_ == 2 && decX_ == 2
                subImg = zeros(2*subScale);
                subCoef1 = arrayCoefs(1,:);
                subCoef2 = arrayCoefs(2,:);
                subCoef3 = arrayCoefs(ps+1,:);
                subCoef4 = arrayCoefs(ps+2,:);
                %
                subImg(1:2:end,1:2:end) = ...
                    reshape(subCoef1+subCoef2+subCoef3+subCoef4,subScale);
                subImg(2:2:end,1:2:end)  = ...
                    reshape(subCoef1-subCoef2-subCoef3+subCoef4,subScale);
                subImg(1:2:end,2:2:end)  = ...
                    reshape(subCoef1-subCoef2+subCoef3-subCoef4,subScale);
                subImg(2:2:end,2:2:end)  = ...
                    reshape(subCoef1+subCoef2-subCoef3-subCoef4,subScale);
                %
                subImg = subImg/2;
            else 
                mc = ceil(decX_*decY_/2);
                mf = floor(decX_*decY_/2);
                coefs = zeros(nDec,size(arrayCoefs,2));
                coefs(1:mc,:) = arrayCoefs(1:mc,:);
                coefs(mc+1:end,:) = arrayCoefs(ps+1:ps+mf,:);
                scale = double(subScale) .* obj.decimationFactor;
                dctCoefs = col2im(coefs,blockSize,scale,'distinct');
                dctCoefs = blockproc(dctCoefs,blockSize,...
                    @obj.permuteIdctCoefs_);
                subImg = blockproc(dctCoefs,blockSize,...
                    @obj.idct2_);
            end
        end
        
    end
    
    methods (Access = private, Static = true)
        
        function value = idct2_(x)
            value = idct2(x.data);
        end
        
        function value = permuteIdctCoefs_(x)
            coefs = x.data;
            decY_ = x.blockSize(1);
            decX_ = x.blockSize(2);
            nQDecsee = ceil(decY_/2)*ceil(decX_/2);
            nQDecsoo = floor(decY_/2)*floor(decX_/2);
            nQDecsoe = floor(decY_/2)*ceil(decX_/2);
            cee = coefs(         1:  nQDecsee);
            coo = coefs(nQDecsee+1:nQDecsee+nQDecsoo);
            coe = coefs(nQDecsee+nQDecsoo+1:nQDecsee+nQDecsoo+nQDecsoe);
            ceo = coefs(nQDecsee+nQDecsoo+nQDecsoe+1:end);
            value = zeros(decY_,decX_);
            value(1:2:decY_,1:2:decX_) = reshape(cee,ceil(decY_/2),ceil(decX_/2));
            value(2:2:decY_,2:2:decX_) = reshape(coo,floor(decY_/2),floor(decX_/2));
            value(2:2:decY_,1:2:decX_) = reshape(coe,floor(decY_/2),ceil(decX_/2));
            value(1:2:decY_,2:2:decX_) = reshape(ceo,ceil(decY_/2),floor(decX_/2));
        end
        
    end
    
end
