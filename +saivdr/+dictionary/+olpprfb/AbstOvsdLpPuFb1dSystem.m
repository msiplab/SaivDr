classdef AbstOvsdLpPuFb1dSystem < matlab.System %#codegen
    %ABSTOVSDLPPUFB1DSYSTEM Abstract class 1-D OLPPUFB
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

    properties
        DecimationFactor = 4;
        PolyPhaseOrder   = [];
        NumberOfChannels = [];
        OutputMode = 'Coefficients'
    end
    
    properties (Hidden, Transient)
        OutputModeSet = ...
            matlab.system.StringSet({...
            'Coefficients',...
            'AnalysisFilters',...
            'AnalysisFilterAt',...
            'SynthesisFilters',...
            'SynthesisFilterAt',...
            'ParameterMatrixSet',...
            'Char'});            
    end
    
    properties (Hidden)
        Angles = 0;
        Mus    = 1;   
    end

    properties (GetAccess = public, SetAccess = protected)
        ParameterMatrixSet
    end
        
    properties (Access = protected)
        Coefficients
    end

    properties(Access = protected, Logical)
        mexFlag = false
    end
    
    methods (Access = protected, Abstract = true)
        value = getAnalysisFilterBank_(obj)
        updateParameterMatrixSet_(obj)
        updateAngles_(obj)
        updateMus_(obj)
    end
    
    methods 
        function obj = AbstOvsdLpPuFb1dSystem(varargin)
            setProperties(obj,nargin,varargin{:});
        end
        
        function atmshow(obj)
            % Show Atomic Images
            updateParameterMatrixSet_(obj);
            obj.mexFlag = false;            
            H = getAnalysisFilterBank_(obj);
            for ib=1:sum(obj.NumberOfChannels)
                subplot(2,obj.NumberOfChannels(1),ib);
                impz(flipud(H(:,ib)));
            end
        end

    end
    
    methods (Access = protected)
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@matlab.System(obj);
            s.ParameterMatrixSet = matlab.System.saveObject(obj.ParameterMatrixSet);
            s.Coefficients = obj.Coefficients;
            s.mexFlag = obj.mexFlag;
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            import saivdr.dictionary.utility.ParameterMatrixContainer
            obj.mexFlag = s.mexFlag;            
            obj.Coefficients = s.Coefficients;
            loadObjectImpl@matlab.System(obj,s,wasLocked);
            %
            obj.ParameterMatrixSet = matlab.System.loadObject(s.ParameterMatrixSet);
        end        
        
        function validatePropertiesImpl(obj)
            id = 'SaivDr:IllegalPropertyException';
            if obj.DecimationFactor > sum(obj.NumberOfChannels)
                error('%s:\n sum(NumberOfChannels) must be greater than or equalto DecimationFactor.',...
                    id);
            end
        end
        
        function resetImpl(obj)
            obj.mexFlag = false;
        end
        
        function output = stepImpl(obj,varargin)
            
            if ~isempty(varargin{1})
                obj.Angles = varargin{1};
                updateAngles_(obj);
            end
            if ~isempty(varargin{2})
                obj.Mus = varargin{2};
                updateMus_(obj);
            end
            
            updateParameterMatrixSet_(obj);
            updateCoefficients_(obj);
            
            if strcmp(obj.OutputMode,'Coefficients') 
                output = obj.Coefficients;
            elseif strcmp(obj.OutputMode,'ParameterMatrixSet')
                output = clone(obj.ParameterMatrixSet);
            elseif strcmp(obj.OutputMode,'AnalysisFilterAt')     
                idx = varargin{3};
                H = getAnalysisFilterBank_(obj);
                output = H(:,idx);
            elseif strcmp(obj.OutputMode,'AnalysisFilters')     
                output = getAnalysisFilterBank_(obj);
            elseif strcmp(obj.OutputMode,'SynthesisFilterAt')     
                idx = varargin{3};
                H = getAnalysisFilterBank_(obj);
                output = flipud(H(:,idx));
            elseif strcmp(obj.OutputMode,'SynthesisFilters')     
                H = getAnalysisFilterBank_(obj);          
                output = flip(H,1);
            else
                output = [];
            end
        end
        
        function N = getNumInputsImpl(obj)
            if strcmp(obj.OutputMode,'Coefficients') || ...
                    strcmp(obj.OutputMode,'Char') || ...
                    strcmp(obj.OutputMode,'ParameterMatrixSet') || ...
                    strcmp(obj.OutputMode,'AnalysisFilters') || ...
                    strcmp(obj.OutputMode,'SynthesisFilters') 
                N = 2;
            else
                N = 3;
            end
        end
        
        function N = getNumOutputsImpl(~)
            N = 1;
        end        
        
        function updateCoefficients_(obj)
            dec = obj.DecimationFactor;
            ord = obj.PolyPhaseOrder;
            nChs = obj.NumberOfChannels;
            %
            H = getAnalysisFilterBank_(obj);
            nSubbands = sum(nChs);
            coefs = zeros(nSubbands,dec,ord+1);
            for iSubband = 1:nSubbands
                hi = H(:,iSubband);
                for iOrd = 0:ord
                    idx = iOrd*dec;
                    b = hi(idx+1:idx+dec);
                    coefs(iSubband,:,iOrd+1) = b(:).';
                end
            end
            obj.Coefficients = coefs;
        end
        
        function value = getMatrixE0_(obj)
            nCoefs = obj.DecimationFactor;
            nElmBi = nCoefs;
            %dctMtx = dctmtx(nCoefs);
            coefs = zeros(nElmBi);
            iElm = 1; % E0.'= [ Bee Boo Boe Beo ] % Byx
            for iCoef = 1:2:nCoefs % y-e
                dctCoef = zeros(nCoefs,1);
                dctCoef(iCoef) = 1;
                %basisVector = dctMtx.'*dctCoef;
                basisVector = idct(dctCoef);
                coefs(iElm,:) = basisVector(:).';
                iElm = iElm + 1;
            end
            for iCoef = 2:2:nCoefs % y-e
                dctCoef = zeros(nCoefs,1);
                dctCoef(iCoef) = 1;
                %basisVector = dctMtx.'*dctCoef;
                basisVector = idct(dctCoef);
                coefs(iElm,:) = basisVector(:).';
                iElm = iElm + 1;
            end
            value = coefs;
        end
        
        function value = permuteCoefs_(~,arr_,phs_)
            len_ = size(arr_,2)/phs_;
            value = zeros(size(arr_));
            for idx = 0:phs_-1
                value(:,idx*len_+1:(idx+1)*len_) = arr_(:,idx+1:phs_:end);
            end
        end
        
        function value = ipermuteCoefs_(~,arr_,phs_)
            len_ = size(arr_,2)/phs_;
            value = zeros(size(arr_));
            for idx = 0:phs_-1
                value(:,idx+1:phs_:end) = arr_(:,idx*len_+1:(idx+1)*len_);
            end
        end
        
    end
    
end
