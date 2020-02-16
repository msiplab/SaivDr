classdef NsoltAnalysis2dSystem < ...
        saivdr.dictionary.AbstAnalysisSystem %#codegen
    %NSOLTANALYSISSYSTEM Abstract class of NSOLT analysis system
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
    % http://msiplab.eng.niigata-u.ac.jp  
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
        NumberOfLevels = 1
    end
    
    properties (Nontunable, Logical)
        IsCloneLpPuFb = true
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
    end
    
    methods
        
        % Constructor
        function obj = NsoltAnalysis2dSystem(varargin)
            setProperties(obj,nargin,varargin{:});
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
        end
        
    end
    
    methods (Access=protected)
        
        function s = saveObjectImpl(obj)
            % Call the base class method
            s = saveObjectImpl@saivdr.dictionary.AbstAnalysisSystem(obj);
            % Save the child System objects            
            s.LpPuFb2d = matlab.System.saveObject(obj.LpPuFb2d);
            
            % Save the protected & private properties           
            s.nAllCoefs  = obj.nAllCoefs;
            s.nAllChs    = obj.nAllChs;
            s.decimationFactor = obj.decimationFactor;
            s.polyPhaseOrder   = obj.polyPhaseOrder;
            s.allScales  = obj.allScales;
            s.allCoefs   = obj.allCoefs;
            s.fcnAtomExt = obj.fcnAtomExt;            
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            % Load protected and private properties
            obj.nAllCoefs  = s.nAllCoefs;
            obj.nAllChs    = s.nAllChs;
            obj.decimationFactor = s.decimationFactor;
            obj.polyPhaseOrder   = s.polyPhaseOrder;
            obj.allScales   = s.allScales;
            obj.allCoefs    = s.allCoefs;
            obj.fcnAtomExt  = s.fcnAtomExt;            
            
            % Call base class method to load public properties
            loadObjectImpl@saivdr.dictionary.AbstAnalysisSystem(obj,s,wasLocked);
            % Load the child System objects            
            obj.LpPuFb2d = matlab.System.loadObject(s.LpPuFb2d);
        end
        
        function setupImpl(obj, srcImg)
            nLevels = obj.NumberOfLevels;
            dec = obj.decimationFactor;
            nch = [ obj.NumberOfSymmetricChannels ...
                obj.NumberOfAntisymmetricChannels ];
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
            if exist('fcn_NsoltAtomExtender2dCodeGen_mex','file')==3
                obj.fcnAtomExt = @fcn_NsoltAtomExtender2dCodeGen_mex;
            else
                import saivdr.dictionary.nsoltx.mexsrcs.fcn_NsoltAtomExtender2dCodeGen
                obj.fcnAtomExt = @fcn_NsoltAtomExtender2dCodeGen;
            end

        end
        
        function [ coefs, scales ] = stepImpl(obj, srcImg)
            nLevels = obj.NumberOfLevels;
            %if obj.IsDifferentiation
            %else
                pmMtx = step(obj.LpPuFb2d,[],[]);
                pmMtxCoefs = get(pmMtx,'Coefficients');
            %end
            [ coefs, scales ] = analyze_(obj, srcImg, nLevels, pmMtxCoefs);
        end
        
    end
    
    methods (Access = private)
        
        function [ coefs, scales ] = ...
                analyze_(obj, srcImg, nLevels, pmCoefs)
            import saivdr.dictionary.utility.Direction            
            %
            nChs = obj.NumberOfSymmetricChannels ...
                + obj.NumberOfAntisymmetricChannels;
            decY  = obj.decimationFactor(Direction.VERTICAL);
            decX  = obj.decimationFactor(Direction.HORIZONTAL);            
            %
            iSubband = obj.nAllChs;
            eIdx     = obj.nAllCoefs;
            %
            subImg = srcImg;
            for iLevel = 1:nLevels
                height = size(subImg,1);
                width  = size(subImg,2);
                obj.nRows = uint32(height/decY);
                obj.nCols = uint32(width/decX);
                arrayCoefs = subAnalyze_(obj,subImg,pmCoefs);
                for iCh = nChs:-1:2
                    subbandCoefs = arrayCoefs(iCh,:);
                    obj.allScales(iSubband,:) = [ obj.nRows obj.nCols ];
                    sIdx = eIdx - (obj.nRows*obj.nCols) + 1;
                    obj.allCoefs(sIdx:eIdx) = subbandCoefs(:).';
                    iSubband = iSubband-1;
                    eIdx = sIdx - 1;
                end
                subImg = reshape(arrayCoefs(1,:),obj.nRows,obj.nCols);
            end
            obj.allScales(1,:) = [ obj.nRows obj.nCols ];
            obj.allCoefs(1:eIdx) = subImg(:).';
            %
            scales = obj.allScales;
            coefs  = obj.allCoefs;
        end
        
        function arrayCoefs = subAnalyze_(obj,subImg,pmCoefs)
            import saivdr.dictionary.utility.Direction
            %
            nChs = obj.NumberOfSymmetricChannels ...
                + obj.NumberOfAntisymmetricChannels;
            ps = obj.NumberOfSymmetricChannels;
            nRows_ = obj.nRows;
            nCols_ = obj.nCols;            
            decY_  = obj.decimationFactor(Direction.VERTICAL);
            decX_  = obj.decimationFactor(Direction.HORIZONTAL);
            %
            blockSize = [ decY_ decX_ ];
            %
            if isinteger(subImg)
                subImg = im2double(subImg);
            end

            % Prepare array
            arrayCoefs = zeros(nChs,nRows_*nCols_);
            
            % Block DCT
            if decY_ == 1 && decX_ == 1
                coefs = im2col(subImg,blockSize,'distinct');
                arrayCoefs(1,:) = coefs(1,:);
            elseif decY_ == 2 && decX_ == 2
                subImg1 = subImg(1:2:end,1:2:end);
                subImg2 = subImg(2:2:end,1:2:end);
                subImg3 = subImg(1:2:end,2:2:end);
                subImg4 = subImg(2:2:end,2:2:end);
                %
                subImg1 = subImg1(:).';
                subImg2 = subImg2(:).';
                subImg3 = subImg3(:).';
                subImg4 = subImg4(:).';
                %                
                arrayCoefs(1,:) = ...
                    (subImg1+subImg2+subImg3+subImg4)/2;
                arrayCoefs(2,:) = ...
                    (subImg1-subImg2-subImg3+subImg4)/2;
                arrayCoefs(ps+1,:) = ...
                    (subImg1-subImg2+subImg3-subImg4)/2;
                arrayCoefs(ps+2,:) = ...
                    (subImg1+subImg2-subImg3-subImg4)/2;
            else
                mc = ceil(decX_*decY_/2);
                mf = floor(decX_*decY_/2);
                dctCoefs = blockproc(subImg,blockSize,...
                    @obj.dct2_);
                dctCoefs = blockproc(dctCoefs,blockSize,...
                    @obj.permuteDctCoefs_);
                coefs = im2col(dctCoefs,blockSize,'distinct');
                arrayCoefs(1:mc,:) = coefs(1:mc,:);
                arrayCoefs(ps+1:ps+mf,:) = coefs(mc+1:end,:);
            end
            
            % Atom extension
            subScale = [ obj.nRows obj.nCols ];
            %arrayCoefs = obj.atomExtFcn(arrayCoefs,subScale,pmCoefs,...
            %    ord,fpe);
            nch = [ obj.NumberOfSymmetricChannels ...
                obj.NumberOfAntisymmetricChannels ];
            ord = uint32(obj.polyPhaseOrder);
            fpe = strcmp(obj.BoundaryOperation,'Circular');
            arrayCoefs = obj.fcnAtomExt(...
                arrayCoefs, subScale, pmCoefs, nch, ord, fpe);
        end        
        
    end
    
    methods (Access = private, Static = true)
        
        function value = dct2_(x)
            value = dct2(x.data);
        end
        
        function value = permuteDctCoefs_(x)
            coefs = x.data;
            decY_ = x.blockSize(1);
            decX_ = x.blockSize(2);
            cee = coefs(1:2:end,1:2:end);
            coo = coefs(2:2:end,2:2:end);
            coe = coefs(2:2:end,1:2:end);
            ceo = coefs(1:2:end,2:2:end);
            value = [ cee(:) ; coo(:) ; coe(:) ; ceo(:) ];
            value = reshape(value,decY_,decX_);
        end
        
    end    
    
end

