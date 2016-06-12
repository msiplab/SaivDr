classdef OLpPuFbSynthesis1dSystem  < ...
        saivdr.dictionary.AbstSynthesisSystem %#~codegen
    %OLPPUFBSYNTHESIS1DSYSTEM Synthesis system of Type-I OLPPURFB
    %
    % SVN identifier:
    % $Id: OLpPuFbSynthesis1dSystem.m 657 2015-03-17 00:45:15Z sho $
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
        NumberOfSymmetricChannels     = 2
        NumberOfAntisymmetricChannels = 2
    end
    
    properties (Nontunable, Logical)
        IsCloneLpPuFb1d = true;
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
        nBlks
    end 
    
    properties (Access = private, Logical)
        isMexFcn = false
    end
    
    methods
        
        % Constractor
        function obj = OLpPuFbSynthesis1dSystem(varargin)
            setProperties(obj,nargin,varargin{:})
            %
            if isempty(obj.LpPuFb1d)
                import saivdr.dictionary.olpprfb.OLpPrFbFactory
                obj.LpPuFb1d = OLpPrFbFactory.createOvsdLpPuFb1dSystem(...
                    'NumberOfChannels', ...
                    [ obj.NumberOfSymmetricChannels ...
                      obj.NumberOfAntisymmetricChannels ], ...
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
            obj.NumberOfSymmetricChannels = ceil(nch/2);
            obj.NumberOfAntisymmetricChannels = floor(nch/2);
            %
            obj.FrameBound = 1;            
        end
        
    end
    
    methods (Access=protected)
        
        function s = saveObjectImpl(obj)
            % Call the base class method
            s = saveObjectImpl@saivdr.dictionary.AbstSynthesisSystem(obj);
            % Save the child System objects            
            s.LpPuFb1d = matlab.System.saveObject(obj.LpPuFb1d);
            
            % Save the protected & private properties
            s.atomCncFcn       = obj.atomCncFcn;            
            s.decimationFactor = obj.decimationFactor;
            s.polyPhaseOrder   = obj.polyPhaseOrder;
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            % Load protected and private properties            
            obj.atomCncFcn       = s.atomCncFcn;
            obj.decimationFactor = s.decimationFactor;
            obj.polyPhaseOrder   = s.polyPhaseOrder;        
            % Call base class method to load public properties            
            loadObjectImpl@saivdr.dictionary.AbstSynthesisSystem(obj,s,wasLocked);
            % Load the child System objects            
            obj.LpPuFb1d = matlab.System.loadObject(s.LpPuFb1d);            
        end
        
        function validatePropertiesImpl(~)
        end
        
        function setupImpl(obj, ~, ~)
            nch = [ obj.NumberOfSymmetricChannels ...
                obj.NumberOfAntisymmetricChannels ];
            
            % Prepare MEX function
            
            %TODO: MEX‰»‚É‘Î‰ž‚µ‚½‚ç‰º‚Ì‚Qs‚ðíœ‚·‚é
            mexFcn = [];
            obj.isMexFcn = 1;
            
            if obj.NumberOfSymmetricChannels == 1 || ...
                    obj.NumberOfAntisymmetricChannels == 1 
                mexFcn = [];
            elseif ~obj.isMexFcn
                import saivdr.dictionary.olpprfb.mexsrcs.fcn_autobuild_atomcnc1d
                [mexFcn, obj.isMexFcn] = ...
                    fcn_autobuild_atomcnc1d(nch);
            end
            
            if ~isempty(mexFcn)
                obj.atomCncFcn = @(coefs,scale,pmcoefs,ord,fpe) ...
                    mexFcn(coefs,scale,pmcoefs,...
                    nch,ord,fpe);
            else
                import saivdr.dictionary.olpprfb.mexsrcs.fcn_OLpPrFbAtomConcatenator1d
                clear fcn_OLpPrFbAtomConcatenator1d
                obj.atomCncFcn = @(coefs,scale,pmcoefs,ord,fpe) ...
                    fcn_OLpPrFbAtomConcatenator1d(coefs,scale,pmcoefs,...
                    nch,ord,fpe);
            end
            
        end
        
        function recSeq = stepImpl(obj, coefs, scales)
            pmMtx = step(obj.LpPuFb1d,[],[]);
            pmMtxCoefs = get(pmMtx,'Coefficients');
            recSeq = synthesize_(obj, coefs, scales, pmMtxCoefs);
        end
        
    end
    
    methods (Access = private)
        
        function recSeq = synthesize_(obj,coefs,scales,pmCoefs)
            import saivdr.dictionary.utility.Direction
            %
            nChs = obj.NumberOfSymmetricChannels ...
                + obj.NumberOfAntisymmetricChannels;
            nLevels = (size(scales,1)-1)/(nChs-1);
            %
            iSubband = 1;
            eIdx = scales(iSubband);
            %
            nLen = scales(1);
            arrayCoefs = zeros(nChs,nLen);
            arrayCoefs(1,:) = coefs(1:eIdx);
            for iLevel = 1:nLevels
                obj.nBlks = uint32(nLen);
                for iCh = 2:nChs
                    iSubband = iSubband + 1;
                    sIdx = eIdx + 1;
                    eIdx = sIdx + prod(scales(iSubband,:))-1;
                    arrayCoefs(iCh,:) = coefs(sIdx:eIdx);
                end
                subSeq = subSynthesize_(obj,arrayCoefs,pmCoefs);
                if iLevel < nLevels
                    nLen = length(subSeq);
                    arrayCoefs = zeros(nChs,nLen);
                    arrayCoefs(1,:) = subSeq(:).';
                end
            end
            recSeq = subSeq;
        end
        
        function subSeq = subSynthesize_(obj,arrayCoefs,pmCoefs)
            import saivdr.dictionary.utility.Direction
            %
            ps = obj.NumberOfSymmetricChannels;
            nBlks_ = obj.nBlks;
            dec_ = obj.decimationFactor;
            %
            if isinteger(arrayCoefs)
                arrayCoefs = double(arrayCoefs);
            end
            
            % Atom concatenation
            subScale  = nBlks_;
            ord = uint32(obj.polyPhaseOrder);            
            fpe = strcmp(obj.BoundaryOperation,'Circular');        
            arrayCoefs = obj.atomCncFcn(arrayCoefs,subScale,pmCoefs,...
                ord,fpe);
            
            % Block IDCT
            if dec_ == 1
                subSeq = arrayCoefs(1,:);
%             elseif dec_ == 2 
%                 subSeq = zeros(1,2*subScale);
%                 subCoef1 =  arrayCoefs(1,:);
%                 subCoef2 = -arrayCoefs(ps+1,:);
%                 %
%                 subSeq(1:2:end) = (subCoef1+subCoef2)/sqrt(2);
%                 subSeq(2:2:end) = (subCoef1-subCoef2)/sqrt(2);
            else
%                 mc = ceil(dec_/2);
%                 mf = floor(dec_/2);
                %hsdftCoefs = complex(zeros(dec_,size(arrayCoefs,2)));
%                 hsdftCoefs(1:mc,:) = arrayCoefs(1:mc,:);
%                 hsdftCoefs(mc+1:end,:) = arrayCoefs(ps+1:ps+mf,:);
                    hsdftCoefs=arrayCoefs(1:dec_,:);
                    Edft = hsdftmtx_(obj,size(hsdftCoefs,1));
                subSeq = reshape(Edft.'*hsdftCoefs, [1 dec_*nBlks_]);
            end
        end
        
        function value = hsdftmtx_(~, nDec) %Hermitian-Symmetric DFT matrix
            w = exp(-2*pi*1i/nDec);
            value = complex(zeros(nDec));
            for u = 0:nDec-1
                for x =0:nDec-1
                    value(u+1,x+1) = w^(u*(x+0.5))/sqrt(nDec);
                end
            end
        end
        
        function value = ihsdft_(obj, X)
            mtx = hsdftmtx_(obj,size(X,1))';
            value = mtx*X;
        end
        
    end
    
end

