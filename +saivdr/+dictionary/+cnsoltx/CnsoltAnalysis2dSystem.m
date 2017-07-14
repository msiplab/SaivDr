classdef CnsoltAnalysis2dSystem < ...
        saivdr.dictionary.AbstAnalysisSystem %#~codegen
    %NSOLTANALYSISSYSTEM Abstract class of NSOLT analysis system
    %
    % SVN identifier:
    % $Id: NsoltAnalysis2dSystem.m 683 2015-05-29 08:22:13Z sho $
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
        NumberOfChannels     = 4
        NumberOfHalfChannels = 2
    end
    
    properties (Nontunable, Logical)
        IsCloneLpPuFb2d = true
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
    
    properties (Access = private, Logical)
        isMexFcn = false
    end
    
    methods
        
        % Constructor
        function obj = CnsoltAnalysis2dSystem(varargin)
            setProperties(obj,nargin,varargin{:});
            %
            if isempty(obj.LpPuFb2d)
                import saivdr.dictionary.cnsoltx.CnsoltFactory
                obj.LpPuFb2d = CnsoltFactory.createCplxOvsdLpPuFb2dSystem(...
                    'NumberOfChannels', obj.NumberOfChannels,...
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
        end
        
    end
    
    methods (Access=protected)
        
        function s = saveObjectImpl(obj)
            % Call the base class method
            s = saveObjectImpl@saivdr.dictionary.AbstAnalysisSystem(obj);
            % Save the child System objects            
            s.LpPuFb2d = matlab.System.saveObject(obj.LpPuFb2d);
            
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
            obj.LpPuFb2d = matlab.System.loadObject(s.LpPuFb2d);
        end
        
        function setupImpl(obj, srcImg, nLevels)
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
            if exist('fcn_CnsoltAtomExtender2dCodeGen_mex','file')==3
                obj.fcnAtomExt = @fcn_CnsoltAtomExtender2dCodeGen_mex;
            else
                import saivdr.dictionary.cnsoltx.mexsrcs.fcn_CnsoltAtomExtender2dCodeGen
                obj.fcnAtomExt = @fcn_CnsoltAtomExtender2dCodeGen;
            end
            
%             if ~obj.isMexFcn
%                 import saivdr.dictionary.cnsoltx.mexsrcs.fcn_autobuild_catomext2d
%                 [mexFcn, obj.isMexFcn] = ...
%                     fcn_autobuild_catomext2d(nch);
%             end
%             if ~isempty(mexFcn)
%                 obj.fcnAtomExt = @(coefs,scale,pmcoefs,ord,fpe) ...
%                     mexFcn(coefs,scale,pmcoefs,...
%                     nch,ord,fpe);
%             else
%                 import saivdr.dictionary.cnsoltx.mexsrcs.fcn_CnsoltAtomExtender2d
%                 clear fcn_CnsoltAtomExtender2d
%                 obj.fcnAtomExt = @(coefs,scale,pmcoefs,ord,fpe) ...
%                     fcn_CnsoltAtomExtender2d(coefs,scale,pmcoefs,...
%                     nch,ord,fpe);
%             end

        end
        
        function [ coefs, scales ] = stepImpl(obj, srcImg, nLevels)
            %if obj.IsDifferentiation
            %else
                pmMtx = step(obj.LpPuFb2d,[],[]);
                pmMtxCoefs = get(pmMtx,'Coefficients');
                symmetry = get(obj.LpPuFb2d,'Symmetry');
            %end
            [ coefs, scales ] = analyze_(obj, srcImg, nLevels, pmMtxCoefs, symmetry);
        end
        
    end
    
    methods (Access = private)
        
        function [ coefs, scales ] = ...
                analyze_(obj, srcImg, nLevels, pmCoefs, symmetry)
            import saivdr.dictionary.utility.Direction            
            %
            nChs = obj.NumberOfChannels;
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
                arrayCoefs = subAnalyze_(obj,subImg,pmCoefs,symmetry);
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
        
        function arrayCoefs = subAnalyze_(obj,subImg,pmCoefs,symmetry)
            import saivdr.dictionary.utility.Direction
            %
            nChs = obj.NumberOfChannels;
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
            arrayCoefs = complex(zeros(nChs,nRows_*nCols_));
            
            % Block DFT
            if decY_ == 1 && decX_ == 1
                coefs = im2col(subImg,blockSize,'distinct');
                arrayCoefs(1,:) = coefs(1,:);
%             elseif decY_ == 2 && decX_ == 2
%                 subImg1 = subImg(1:2:end,1:2:end);
%                 subImg2 = subImg(2:2:end,1:2:end);
%                 subImg3 = subImg(1:2:end,2:2:end);
%                 subImg4 = subImg(2:2:end,2:2:end);
%                 %
%                 subImg1 = subImg1(:).';
%                 subImg2 = subImg2(:).';
%                 subImg3 = subImg3(:).';
%                 subImg4 = subImg4(:).';
%                 %                
%                 arrayCoefs(1,:) = ...
%                     (subImg1+subImg2+subImg3+subImg4)/2;
%                 arrayCoefs(2,:) = ...
%                     (subImg1-subImg2-subImg3+subImg4)/2;
%                 arrayCoefs(ps+1,:) = ...
%                     (subImg1-subImg2+subImg3-subImg4)/2;
%                 arrayCoefs(ps+2,:) = ...
%                     (subImg1+subImg2-subImg3-subImg4)/2;
            else
                dftCoefs = blockproc(subImg,blockSize,...
                    @(x) saivdr.utility.HermitianSymmetricDFT.hsdft2(x.data));
                coefs = im2col(dftCoefs,blockSize,'distinct');
                arrayCoefs(1:decX_*decY_,:) = coefs;
            end
            
            % Atom extension
            S = diag(exp(1i*symmetry));
            subScale = [ obj.nRows obj.nCols ];
            nch = obj.NumberOfChannels;
            ord   = uint32(obj.polyPhaseOrder);            
            fpe = strcmp(obj.BoundaryOperation,'Circular');
            arrayCoefs = S*obj.fcnAtomExt(arrayCoefs,subScale,pmCoefs,...
                nch,ord,fpe);
        end
        
    end   
    
end
