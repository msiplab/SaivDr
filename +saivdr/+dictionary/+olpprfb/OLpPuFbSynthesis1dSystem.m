classdef OLpPuFbSynthesis1dSystem  < ...
        saivdr.dictionary.AbstSynthesisSystem %#codegen
    %OLPPUFBSYNTHESIS1DSYSTEM Synthesis system of Type-I OLPPURFB
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2017, Shogo MURAMATSU
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
    
    properties (Access = private)
        decimationFactor
        polyPhaseOrder
    end    

    properties (Access = private)
        fcnAtomCnc
    end
    
    properties (Access = private, PositiveInteger)
        nBlks
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
            s.LpPuFb1d = matlab.System.saveObject(obj.LpPuFb1d);
            
            % Save the protected & private properties           
            s.decimationFactor = obj.decimationFactor;
            s.polyPhaseOrder   = obj.polyPhaseOrder;
            s.fcnAtomCnc       = obj.fcnAtomCnc;  
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            % Load protected and private properties            
            obj.decimationFactor = s.decimationFactor;
            obj.polyPhaseOrder   = s.polyPhaseOrder;        
            obj.fcnAtomCnc       = s.fcnAtomCnc;    
           
            % Call base class method to load public properties            
            loadObjectImpl@saivdr.dictionary.AbstSynthesisSystem(obj,s,wasLocked);
            % Load the child System objects            
            obj.LpPuFb1d = matlab.System.loadObject(s.LpPuFb1d);            
        end
        
        function validatePropertiesImpl(~)
        end
        
        function setupImpl(obj, ~, ~)
            if exist('fcn_OLpPrFbAtomConcatenator1dCodeGen_mex','file')==3
                obj.fcnAtomCnc = @fcn_OLpPrFbAtomConcatenator1dCodeGen_mex;
            else
                import saivdr.dictionary.olpprfb.mexsrcs.fcn_OLpPrFbAtomConcatenator1dCodeGen
                obj.fcnAtomCnc = @fcn_OLpPrFbAtomConcatenator1dCodeGen;
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
            %arrayCoefs = obj.atomCncObj.step(arrayCoefs,subScale,pmCoefs);
            nch = [ obj.NumberOfSymmetricChannels ...
                obj.NumberOfAntisymmetricChannels ];
            ord = uint32(obj.polyPhaseOrder);
            fpe = strcmp(obj.BoundaryOperation,'Circular');
            arrayCoefs = obj.fcnAtomCnc(...
                arrayCoefs, subScale, pmCoefs, nch, ord, fpe);

            % Block IDCT
            if dec_ == 1
                subSeq = arrayCoefs(1,:);
            elseif dec_ == 2 
                subSeq = zeros(1,2*subScale);
                subCoef1 =  arrayCoefs(1,:);
                subCoef2 = -arrayCoefs(ps+1,:);
                %
                subSeq(1:2:end) = (subCoef1+subCoef2)/sqrt(2);
                subSeq(2:2:end) = (subCoef1-subCoef2)/sqrt(2);
            else 
                mc = ceil(dec_/2);
                mf = floor(dec_/2);
                dctCoefs = zeros(dec_,size(arrayCoefs,2));
                dctCoefs(1:2:end,:) =  arrayCoefs(1:mc,:);
                dctCoefs(2:2:end,:) = -arrayCoefs(ps+1:ps+mf,:);
                subSeq = reshape(idct(dctCoefs),[1 dec_*nBlks_]);
            end
        end
        
    end
    
end

