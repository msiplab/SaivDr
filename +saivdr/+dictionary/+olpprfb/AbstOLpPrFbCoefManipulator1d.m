classdef AbstOLpPrFbCoefManipulator1d < matlab.System %#codegen
    %ABSTOLPPRFBCOEFMANIPULATOR1D 1-D Coefficient Manipulator for OLPPRFB
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
    
    properties (Access = protected, Constant = true)
        DATA_DIMENSION = 1
    end
    
    properties (PositiveInteger)
        NumberOfSymmetricChannels      = 2
        NumberOfAntisymmetricChannels  = 2
    end

    properties (Logical)
        IsPeriodicExt = false
    end

    properties 
        PolyPhaseOrder = 0;
    end

    properties (SetAccess = protected, GetAccess = public)
        OLpPrFbType = 'Type I'
    end
    
    properties (Hidden, Transient)
        OLpPrFbTypeSet = ...
            matlab.system.StringSet({'Type I','Type II'});
    end
    
    properties (SetAccess = protected, GetAccess = public, Logical)
        IsPsGreaterThanPa = true;
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
        function obj = AbstOLpPrFbCoefManipulator1d(varargin)
            setProperties(obj,nargin,varargin{:});
            %
            ps = obj.NumberOfSymmetricChannels;
            pa = obj.NumberOfAntisymmetricChannels;
            %
            if ps > pa
                obj.OLpPrFbType = 'Type II';
                obj.IsPsGreaterThanPa = true;
            elseif ps < pa
                obj.OLpPrFbType = 'Type II';
                obj.IsPsGreaterThanPa = false;
            else
                obj.OLpPrFbType = 'Type I';
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
            if lenOrd ~= saivdr.dictionary.olpprfb.AbstOLpPrFbCoefManipulator1d.DATA_DIMENSION
                error('%s:\n lentgh(PolyPhaseOrder) must be %d.',...
                    id, saivdr.dictionary.olpprfb.AbstOLpPrFbCoefManipulator1d.DATA_DIMENSION);
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
              
        function processTunedPropertiesImpl(obj)
            propChange = ...
                isChangedProperty(obj,'NumberOfSymmetricChannels') ||...
                isChangedProperty(obj,'NumberOfAntisymmetricChannels') ||...
                isChangedProperty(obj,'PolyPhaseOrder');
            if propChange
                ps = obj.NumberOfSymmetricChannels;
                pa = obj.NumberOfAntisymmetricChannels;
                %
                if ps > pa
                    obj.OLpPrFbType = 'Type II';
                    obj.IsPsGreaterThanPa = true;
                elseif ps < pa
                    obj.OLpPrFbType = 'Type II';
                    obj.IsPsGreaterThanPa = false;
                else
                    obj.OLpPrFbType = 'Type I';
                end
            end
            setupParamMtx_(obj);
        end
        
        function setupImpl(obj, ~, subScale, ~)
            obj.nBlks = subScale;
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
            ps  = obj.NumberOfSymmetricChannels;
            pa  = obj.NumberOfAntisymmetricChannels;
            %
            obj.paramMtxSzTab = zeros(sum(ord)+2, 2);
            for iOrd = 0:sum(ord)/2
                obj.paramMtxSzTab(2*iOrd+1,:) = [ ps ps ];
                obj.paramMtxSzTab(2*iOrd+2,:) = [ pa pa ];
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
            value = zeros(dimension);
            nRows_ = dimension(1);
            for iRow = 1:nRows_
                value(iRow,:) = pmCoefs(iRow:nRows_:end);
            end
        end
        
        function arrayCoefs = blockButterflyTypeI_(obj,arrayCoefs)
            hLen = obj.NumberOfSymmetricChannels;
            upper = arrayCoefs(1:hLen,:);
            lower = arrayCoefs(hLen+1:end,:);
            arrayCoefs(1:hLen,:)     = upper + lower;
            arrayCoefs(hLen+1:end,:) = upper - lower;
        end
        
        function arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs)
            chs = [obj.NumberOfSymmetricChannels ...
                obj.NumberOfAntisymmetricChannels ];
            nChMx  = max(chs);
            nChMn  = min(chs);
            upper  = arrayCoefs(1:nChMn,:);
            middle = arrayCoefs(nChMn+1:nChMx,:);
            lower  = arrayCoefs(nChMx+1:end,:);
            arrayCoefs(1:nChMn,:)       = upper + lower;
            arrayCoefs(nChMn+1:nChMx,:) = 1.414213562373095*middle;
            arrayCoefs(nChMx+1:end,:)   = upper - lower;
        end
               
    end
    
end
