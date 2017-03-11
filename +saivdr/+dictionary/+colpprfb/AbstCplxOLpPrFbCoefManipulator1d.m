classdef AbstCplxOLpPrFbCoefManipulator1d < matlab.System %#codegen
    %ABSTOLPPRFBCOEFMANIPULATOR1D 1-D Coefficient Manipulator for OLPPRFB
    %
    % Requirements: MATLAB R2013b
    %
    % Copyright (c) 2015-2016, Shogo MURAMATSU
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
    
    properties (Nontunable, PositiveInteger)
        NumberOfChannels = 4
        NumberOfHalfChannels = 2
    end

    properties (Logical)
        IsPeriodicExt = false
    end

    properties 
        PolyPhaseOrder = 0;
    end

    properties (SetAccess = protected, GetAccess = public, Nontunable)
        OLpPrFbType = 'Type I'
    end
    
    properties (Hidden, Transient)
        OLpPrFbTypeSet = ...
            matlab.system.StringSet({'Type I','Type II'});
    end   
    
    properties (Access = protected)
        paramMtxCoefs
        indexOfParamMtxSzTab
        paramMtxSzTab
        tmpArray
    end
    
    properties (Access = protected, PositiveInteger)
        nBlks
    end
    
    methods
        
        % Constructor
        function obj = AbstCplxOLpPrFbCoefManipulator1d(varargin)
            setProperties(obj,nargin,varargin{:});
            
            obj.NumberOfHalfChannels = floor(obj.NumberOfChannels/2);
            if mod(obj.NumberOfChannels,2) ~= 0
                obj.OLpPrFbType = 'Type II';
            end
        end
        
    end
    
    methods ( Access = protected )
        
        function s = saveObjectImpl(obj)
            % Call the base class method
            s = saveObjectImpl@matlab.System(obj);
            
            % Save the child System objects
            %s.xxx = matlab.System.saveObject(obj.xxx);
            
            % Save the protected & private properties
            s.indexOfParamMtxSzTab = obj.indexOfParamMtxSzTab;
            s.paramMtxSzTab = obj.paramMtxSzTab;
            s.OLpPrFbType = obj.OLpPrFbType;
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            % Load child System objects
            %obj.xxx = matlab.System.loadObject(s.xxx);
            
            % Load protected and private properties
            obj.indexOfParamMtxSzTab = s.indexOfParamMtxSzTab;
            obj.paramMtxSzTab = s.paramMtxSzTab;
            obj.OLpPrFbType = s.OLpPrFbType;
        
            % Call base class method to load public properties
            loadObjectImpl@matlab.System(obj,s,wasLocked);
        end
        
        function validatePropertiesImpl(obj)
            %
            id = 'SaivDr:IllegalPropertyException';
            lenOrd = length(obj.PolyPhaseOrder);
            if lenOrd ~= saivdr.dictionary.colpprfb.AbstCplxOLpPrFbCoefManipulator1d.DATA_DIMENSION
                error('%s:\n lentgh(PolyPhaseOrder) must be %d.',...
                    id, saivdr.dictionary.colpprfb.AbstCplxOLpPrFbCoefManipulator1d.DATA_DIMENSION);
            end

            %TODO: create test cases of followings:
            if obj.NumberOfChannels < 2
                error('%s:\n NumberOfChannels must be more than 2.',id);
            end
            if obj.NumberOfHalfChannels ~= floor(obj.NumberOfChannels/2)
                error('%s;\n NumberOfHalfChannels must be %d.',id,floor(obj.NumberOfChannels/2));
            end
        end
        
        function validateInputsImpl(~, coefs, subScale, ~)
            %
            id = 'SaivDr:IllegalArgumentException';
            if size(coefs,2) ~= prod(subScale)
                error('%s:\n size(coefs,2) should be equal to prod(subScale)',...
                    id);
            end
            %
        end
        
        function setupImpl(obj, ~, subScale, ~)
            obj.nBlks = subScale;
            setupParamMtx_(obj);
        end

        function processTunedPropertiesImpl(obj)
            setupParamMtx_(obj);
        end
        
        function stepImpl(obj,coefs,subScale,pmCoefs)
            %
            obj.paramMtxCoefs = pmCoefs;
            if size(coefs,2) ~= (obj.nBlks)
                obj.tmpArray = zeros(size(coefs)); 
            end
            obj.nBlks = subScale;
            %
        end

        function value = getNumInputsImpl(~)
            value = 3;
        end

        function value = getNumOutputsImpl(~)
            value = 1;
        end
        
        function setupParamMtx_(obj)
            ord = obj.PolyPhaseOrder; 
            pa  = obj.NumberOfHalfChannels;
            ps  = obj.NumberOfChannels - pa;
            %
            obj.paramMtxSzTab = zeros(3*sum(ord)+1, 2);
            obj.paramMtxSzTab(1,:) = [ps+pa, ps+pa];
            for iOrd = 1:sum(ord)/2
                obj.paramMtxSzTab(6*iOrd-4,:) = [ pa pa ];
                obj.paramMtxSzTab(6*iOrd-3,:) = [ pa pa ];
                obj.paramMtxSzTab(6*iOrd-2,:) = [ floor(pa/2) 1 ];
                obj.paramMtxSzTab(6*iOrd-1,:) = [ ps ps ];
                obj.paramMtxSzTab(6*iOrd  ,:) = [ ps ps ];
                obj.paramMtxSzTab(6*iOrd+1,:) = [ floor(pa/2) 1 ];
            end
            %
            nPm = size(obj.paramMtxSzTab,1);
            obj.indexOfParamMtxSzTab = zeros(nPm,3);
            cidx = 1;
            for idx = uint32(1):nPm
                obj.indexOfParamMtxSzTab(idx,:) = ...
                    [ cidx obj.paramMtxSzTab(idx,:)];
                cidx = cidx + prod(obj.paramMtxSzTab(idx,:));
            end            
        end
        
        function value = getParamMtx_(obj,index)
            startIdx  = obj.indexOfParamMtxSzTab(index,1);
            dimension = obj.indexOfParamMtxSzTab(index,2:3);
            nElements = prod(dimension);
            endIdx = startIdx + nElements - 1;
            pmCoefs = obj.paramMtxCoefs(startIdx:endIdx);
            value = complex(zeros(dimension));
            nRows_ = dimension(1);
            for iRow = 1:nRows_
                value(iRow,:) = pmCoefs(iRow:nRows_:end);
            end
        end
        
        % B'*arrayCoefs
        function arrayCoefs = blockButterflyPre_(obj,arrayCoefs,Cs,Ss)
            hLen = obj.NumberOfHalfChannels;
            upper = arrayCoefs(1:hLen,:);
            lower = arrayCoefs(hLen+1:2*hLen,:);
            
            %parfor idx = 1:floor(hLen/2)
            for idx = 1:floor(hLen/2)
                range = 2*idx-1:2*idx;
                C = Cs(:,:,idx);
                S = Ss(:,:,idx);
                arrayCoefs(range,:)      = C' *upper(range,:) + S'*lower(range,:);
                arrayCoefs(range+hLen,:) = C.'*upper(range,:) + S.' *lower(range,:);
            end
            if mod(hLen,2) ~= 0
                arrayCoefs(hLen,:) = upper(hLen,:) - 1i*lower(hLen,:);
                arrayCoefs(end,:)  = upper(hLen,:) + 1i*lower(hLen,:);
            end
        end
        
        % B*arrayCoefs
        function arrayCoefs = blockButterflyPost_(obj,arrayCoefs,Cs,Ss)
            hLen = obj.NumberOfHalfChannels;
            upper = arrayCoefs(1:hLen,:);
            lower = arrayCoefs(hLen+1:2*hLen,:);
            
            %parfor idx = 1:floor(hLen/2)
            for idx = 1:floor(hLen/2)
                range = 2*idx-1:2*idx;
                C = Cs(:,:,idx);
                S = Ss(:,:,idx);
                
                arrayCoefs(range,:)      = C*upper(range,:) + conj(C)*lower(range,:);
                arrayCoefs(range+hLen,:) = S*upper(range,:) + conj(S)*lower(range,:);
            end
            if mod(hLen,2) ~= 0
                arrayCoefs(hLen,:) =    upper(hLen,:) +    lower(hLen,:);
                arrayCoefs(end,:)  = 1i*upper(hLen,:) - 1i*lower(hLen,:);
            end
        end
   
    end
    
end