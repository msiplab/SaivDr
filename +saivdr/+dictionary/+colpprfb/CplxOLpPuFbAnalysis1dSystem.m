classdef CplxOLpPuFbAnalysis1dSystem < ...
        saivdr.dictionary.AbstAnalysisSystem %#~codegen
    %OLPPUFBANALYSIS1DSYSTEM 1-D OLPPUFB analysis system
    %
    % SVN identifier:
    % $Id: CplxOLpPuFbAnalysis1dSystem.m 690 2015-06-09 09:37:49Z sho $
    %
    % Requirements: MATLAB R2013b
    %
    % Copyright (c) 2015, Shogo MURAMATSU
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
        DATA_DIMENSION = 1
    end
    
    properties (Nontunable)
        LpPuFb1d
        BoundaryOperation = 'Termination'
    end

    properties (Nontunable, PositiveInteger)   
        NumberOfChannels = 4
        NumberOfHalfChannels = 2
    end
    
    properties (Nontunable, Logical)
        IsCloneLpPuFb1d = true
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
        nBlks
    end
    
    properties (Access = private, Logical)
        isMexFcn = false
    end
    
    methods
        
        % Constructor
        function obj = CplxOLpPuFbAnalysis1dSystem(varargin)
            setProperties(obj,nargin,varargin{:});
            %
            if isempty(obj.LpPuFb1d)
                import saivdr.dictionary.colpprfb.CplxOLpPrFbFactory
                obj.LpPuFb1d = CplxOLpPrFbFactory.createCplxOvsdLpPuFb1dSystem(...
                    'NumberOfChannels', obj.NumberOfChannels, ...
                    'NumberOfVanishingMoments',1,...
                    'OutputMode','ParameterMatrixSet');
            end
            %
            if obj.IsCloneLpPuFb1d
                obj.LpPuFb1d = clone(obj.LpPuFb1d); 
            end
            %
            if ~strcmp(get(obj.LpPuFb1d,'OutputMode'),'ParameterMatrixSet')
                release(obj.LpPuFb1d);
                set(obj.LpPuFb1d,'OutputMode','ParameterMatrixSet');            
            end
            %
            obj.decimationFactor = get(obj.LpPuFb1d,'DecimationFactor');
            obj.polyPhaseOrder   = get(obj.LpPuFb1d,'PolyPhaseOrder');
            nch = get(obj.LpPuFb1d,'NumberOfChannels');
            obj.NumberOfChannels = nch;
            obj.NumberOfHalfChannels = floor(nch/2);
        end
        
    end
    
    methods (Access=protected)
        
        function s = saveObjectImpl(obj)
            % Call the base class method
            s = saveObjectImpl@saivdr.dictionary.AbstAnalysisSystem(obj);
            % Save the child System objects            
            s.LpPuFb1d = matlab.System.saveObject(obj.LpPuFb1d);
            
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
            obj.LpPuFb1d = matlab.System.loadObject(s.LpPuFb1d);
        end
        
        function setupImpl(obj, srcSeq, nLevels)
            dec = obj.decimationFactor;
            nch = obj.NumberOfChannels;
            %
            nChs  = sum(nch);
            nDec = dec;
            %
            if nDec == 1
                obj.nAllCoefs = numel(srcSeq)*(...
                    (nChs-1)*(nLevels/nDec) + 1/nDec^nLevels);
            else
                obj.nAllCoefs = numel(srcSeq)*(...
                    (nChs-1)*(nDec^nLevels-1)/(nDec^nLevels*(nDec-1))  ...
                    + 1/nDec^nLevels);
            end
            obj.nAllChs = nLevels*(nChs-1)+1;            
            obj.allCoefs  = zeros(1,obj.nAllCoefs);
            obj.allScales = zeros(obj.nAllChs,obj.DATA_DIMENSION);
            
            % Prepare MEX function
            if exist('fcn_CplxOLpPrFbAtomExtender1dCodeGen_mex','file')==3
                obj.fcnAtomExt = @fcn_CplxOLpPrFbAtomExtender1dCodeGen_mex;
            else
                import saivdr.dictionary.colpprfb.mexsrcs.fcn_CplxOLpPrFbAtomExtender1dCodeGen
                obj.fcnAtomExt = @fcn_CplxOLpPrFbAtomExtender1dCodeGen;
            end
%             
%             if obj.NumberOfChannels <= 3
%                 mexFcn = [];
%             elseif ~obj.isMexFcn
%                 import saivdr.dictionary.colpprfb.mexsrcs.fcn_autobuild_catomext1d
%                 [mexFcn, obj.isMexFcn] = ...
%                     fcn_autobuild_catomext1d(nch);
%             end
%             if ~isempty(mexFcn)
%                 obj.fcnAtomExt = @(coefs,scale,pmcoefs,ord,fpe) ...
%                     mexFcn(coefs,scale,pmcoefs,...
%                     nch,ord,fpe);
%             else
%                 import saivdr.dictionary.colpprfb.mexsrcs.fcn_CplxOLpPrFbAtomExtender1d
%                 clear fcn_CplxOLpPrFbAtomExtender1d
%                 obj.fcnAtomExt = @(coefs,scale,pmcoefs,ord,fpe) ...
%                     fcn_CplxOLpPrFbAtomExtender1d(coefs,scale,pmcoefs,...
%                     nch,ord,fpe);
%             end
        end
        
        function [ coefs, scales ] = stepImpl(obj, srcSeq, nLevels)
            %if obj.IsDifferentiation
            %else
                pmMtx = step(obj.LpPuFb1d,[],[]);
                pmMtxCoefs = get(pmMtx,'Coefficients');
            %end
            [ coefs, scales ] = analyze_(obj, srcSeq, nLevels, pmMtxCoefs);
        end
        
    end
    
    methods (Access = private)
        
        function [ coefs, scales ] = ...
                analyze_(obj, srcSeq, nLevels, pmCoefs)
            %
            nChs = obj.NumberOfChannels;
            dec  = obj.decimationFactor;
            %
            iSubband = obj.nAllChs;
            eIdx     = obj.nAllCoefs;
            %
            subSeq = srcSeq;
            for iLevel = 1:nLevels
                nLen = length(subSeq);
                obj.nBlks = uint32(nLen/dec);
                arrayCoefs = subAnalyze_(obj,subSeq,pmCoefs);
                for iCh = nChs:-1:2
                    subbandCoefs = arrayCoefs(iCh,:);
                    obj.allScales(iSubband,:) = obj.nBlks;
                    sIdx = eIdx - (obj.nBlks) + 1;
                    obj.allCoefs(sIdx:eIdx) = subbandCoefs(:).';
                    iSubband = iSubband-1;
                    eIdx = sIdx - 1;
                end
                subSeq = arrayCoefs(1,:);
            end
            obj.allScales(1,:) = obj.nBlks;
            obj.allCoefs(1:eIdx) = subSeq(:).';
            %
            scales = obj.allScales;
            coefs  = obj.allCoefs;
        end
        
        function arrayCoefs = subAnalyze_(obj,subSeq,pmCoefs)
            import saivdr.dictionary.utility.Direction
            %
            nChs = obj.NumberOfChannels;
            nBlks_ = obj.nBlks;
            dec_  = obj.decimationFactor;
            %
            if isinteger(subSeq)
                subSeq = im2double(subSeq);
            end

            % Prepare array
            arrayCoefs = complex(zeros(nChs,nBlks_));
            
            % Block DCT
            if dec_ == 1
                arrayCoefs(1,:) = subSeq(:).';
%             elseif dec_ == 2
%                 subSeq1 = subSeq(1:2:end);
%                 subSeq2 = subSeq(2:2:end);
%                 %
%                 subSeq1 = subSeq1(:).';
%                 subSeq2 = subSeq2(:).';
%                 %
%                 arrayCoefs(1,:)    = (subSeq1+subSeq2)/sqrt(2);
%                 arrayCoefs(ps+1,:) = -(subSeq1-subSeq2)/sqrt(2);
            else
                import saivdr.utility.HermitianSymmetricDFT
                Edft = HermitianSymmetricDFT.hsdftmtx(dec_);
                hsdftCoefs = Edft*reshape(subSeq,[dec_, nBlks_]);
                arrayCoefs(1:dec_,:) = hsdftCoefs;
            end
            
            % Atom extension
            subScale = obj.nBlks;
            nch = obj.NumberOfChannels;
            ord = uint32(obj.polyPhaseOrder);            
            fpe = strcmp(obj.BoundaryOperation,'Circular');
            arrayCoefs = obj.fcnAtomExt(arrayCoefs,subScale,pmCoefs,...
                nch,ord,fpe);
        end
        
    end

end
