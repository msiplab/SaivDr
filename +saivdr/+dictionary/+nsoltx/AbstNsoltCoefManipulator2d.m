classdef AbstNsoltCoefManipulator2d < matlab.System
    %ABSTNSOLTCOEFMANIPULATOR2D 2-D Coefficient Manipulator for NSOLT
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2014-2017, Shogo MURAMATSU
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
    
    properties (PositiveInteger)
        NumberOfSymmetricChannels      = 2
        NumberOfAntisymmetricChannels  = 2
    end
    
    properties (Logical)
        IsPeriodicExt = false
    end
    
    properties
        PolyPhaseOrder = [ 0 0 ]
    end
    
    properties (SetAccess = protected, GetAccess = public)
        NsoltType = 'Type I'
    end
    
    properties (Hidden, Transient)
        NsoltTypeSet = ...
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
        nRows
        nCols
    end
    
    methods
        
        % Constructor
        function obj = AbstNsoltCoefManipulator2d(varargin)
            setProperties(obj,nargin,varargin{:});
            %
            ps = obj.NumberOfSymmetricChannels;
            pa = obj.NumberOfAntisymmetricChannels;
            %
            if ps > pa
                obj.NsoltType = 'Type II';
                obj.IsPsGreaterThanPa = true;
            elseif ps < pa
                obj.NsoltType = 'Type II';
                obj.IsPsGreaterThanPa = false;
            else
                obj.NsoltType = 'Type I';
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
            s.NsoltType = obj.NsoltType;
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            % Load child System objects
            %obj.xxx = matlab.System.loadObject(s.xxx);
            
            % Load protected and private properties
            obj.indexOfParamMtxSzTab = s.indexOfParamMtxSzTab;
            obj.paramMtxSzTab = s.paramMtxSzTab;
            obj.NsoltType = s.NsoltType;
            
            % Call base class method to load public properties
            loadObjectImpl@matlab.System(obj,s,wasLocked);
        end
        
        function validatePropertiesImpl(obj)
            %
            id = 'SaivDr:IllegalPropertyException';
            lenOrd = length(obj.PolyPhaseOrder);
            if lenOrd ~= saivdr.dictionary.nsoltx.AbstNsoltCoefManipulator2d.DATA_DIMENSION
                error('%s:\n lentgh(PolyPhaseOrder) must be %d.',...
                    id, saivdr.dictionary.nsoltx.AbstNsoltCoefManipulator2d.DATA_DIMENSION);
            end
            
        end
        
        function validateInputsImpl(~, coefs, subScale, ~)
            %
            id = 'SaivDr:IllegalArgumentException';
            if size(coefs,2) ~= prod(subScale)
                error('%s:\n size(coefs,2) should be equalt to prod(subScale)',...
                    id);
            end
            %
        end
        
        function setupImpl(obj, ~, subScale, ~)
            obj.nRows = subScale(saivdr.dictionary.utility.Direction.VERTICAL);
            obj.nCols = subScale(saivdr.dictionary.utility.Direction.HORIZONTAL);
            %
            ps = obj.NumberOfSymmetricChannels;
            pa = obj.NumberOfAntisymmetricChannels;
            %
            if ps > pa
                obj.NsoltType = 'Type II';
                obj.IsPsGreaterThanPa = true;
            elseif ps < pa
                obj.NsoltType = 'Type II';
                obj.IsPsGreaterThanPa = false;
            else
                obj.NsoltType = 'Type I';
            end
            %
            setupParamMtx_(obj);
        end
        
        function processTunedPropertiesImpl(obj)
            propChange = ...
                isChangedProperty(obj,'NumberOfSymmetricChannels') ||...
                isChangedProperty(obj,'NumberOfAntisymmetricChannels'); % ||...
                isChangedProperty(obj,'PolyPhaseOrder');
            if propChange
                ps = obj.NumberOfSymmetricChannels;
                pa = obj.NumberOfAntisymmetricChannels;
                %
                if ps > pa
                    obj.NsoltType = 'Type II';
                    obj.IsPsGreaterThanPa = true;
                elseif ps < pa
                    obj.NsoltType = 'Type II';
                    obj.IsPsGreaterThanPa = false;
                else
                    obj.NsoltType = 'Type I';
                end
                setupParamMtx_(obj);
            end
        end
        
        function stepImpl(obj,coefs,subScale,pmCoefs)
            %
            obj.paramMtxCoefs = pmCoefs;
            %
            if size(coefs,2) ~= (obj.nRows*obj.nCols)
                obj.tmpArray = zeros(size(coefs));
            end
            %
            obj.nRows = subScale(saivdr.dictionary.utility.Direction.VERTICAL);
            obj.nCols = subScale(saivdr.dictionary.utility.Direction.HORIZONTAL);
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
            paramMtxSzTab_ = zeros(sum(ord)+2, 2);
            paramMtxSzTab_(1,:) = [ ps ps ];
            paramMtxSzTab_(2,:) = [ pa pa ];
            if strcmp(obj.NsoltType,'Type I')
                for iOrd = 1:sum(ord)
                    paramMtxSzTab_(iOrd+2,:) = [ pa pa ];
                end
            else
                for iOrd = 1:sum(ord)/2
                    paramMtxSzTab_(2*iOrd+1,:)   = [ ps ps ];
                    paramMtxSzTab_(2*iOrd+2,:) = [ pa pa ];
                end
            end
            %
            nRowsPm = size(paramMtxSzTab_,1);
            indexOfParamMtxSzTab_ = zeros(nRowsPm,3);
            cidx = 1;
            for iRow = uint32(1):nRowsPm
                indexOfParamMtxSzTab_(iRow,:) = ...
                    [ cidx paramMtxSzTab_(iRow,:)];
                cidx = cidx + prod(paramMtxSzTab_(iRow,:));
            end
            obj.paramMtxSzTab = paramMtxSzTab_;
            obj.indexOfParamMtxSzTab = indexOfParamMtxSzTab_;
        end
        
        function value = getParamMtx_(obj,index)
            startIdx  = obj.indexOfParamMtxSzTab(index,1);
            dimension = obj.indexOfParamMtxSzTab(index,2:3);
            nElements = prod(dimension);
            endIdx = startIdx + nElements - 1;
            value = reshape(... % <- Coder doesn't support this. (R2014a)
                obj.paramMtxCoefs(startIdx:endIdx),...
                dimension);
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
            nChMx = max(chs);
            nChMn = min(chs);
            upper = arrayCoefs(1:nChMn,:);
            middle = arrayCoefs(nChMn+1:nChMx,:);
            lower = arrayCoefs(nChMx+1:end,:);
            
            arrayCoefs(1:nChMn,:) = upper + lower;
            arrayCoefs(nChMn+1:nChMx,:) = 1.414213562373095*middle;
            arrayCoefs(nChMx+1:end,:) =  upper - lower;
        end
        
        function arrayCoefs = lowerBlockRot_(obj,arrayCoefs,iCol,U)
            hLen = obj.NumberOfSymmetricChannels;
            nRows_ = obj.nRows;
            indexCol = (iCol-1)*nRows_;
            arrayCoefs(hLen+1:end,indexCol+1:indexCol+nRows_) = ...
                U*arrayCoefs(hLen+1:end,indexCol+1:indexCol+nRows_);
        end
        
        function arrayCoefs = upperBlockRot_(obj,arrayCoefs,iCol,W)
            hLen = obj.NumberOfSymmetricChannels;
            nRows_ = obj.nRows;
            indexCol = (iCol-1)*nRows_;
            arrayCoefs(1:hLen,indexCol+1:indexCol+nRows_) = ...
                W*arrayCoefs(1:hLen,indexCol+1:indexCol+nRows_);
        end
        
        function arrayCoefs = permuteCoefs_(obj,arrayCoefs)
            nRows_ = obj.nRows;
            nCols_ = obj.nCols;
            obj.tmpArray = arrayCoefs;
            for idx = 0:nRows_-1
                arrayCoefs(:,idx*nCols_+1:(idx+1)*nCols_) = ...
                    obj.tmpArray(:,idx+1:nRows_:end);
            end
            obj.nRows = nCols_;
            obj.nCols = nRows_;
        end
        
        function arrayCoefs = ipermuteCoefs_(obj,arrayCoefs)
            nRows_ = obj.nRows;
            nCols_ = obj.nCols;
            obj.tmpArray = arrayCoefs;
            for idx = 0:nCols_-1
                arrayCoefs(:,idx+1:nCols_:end) = ...
                    obj.tmpArray(:,idx*nRows_+1:(idx+1)*nRows_);
            end
            obj.nRows = nCols_;
            obj.nCols = nRows_;
        end
        
    end
    
end
