classdef AbstCnsoltCoefManipulator2d < matlab.System 
    %ABSTNSOLTCOEFMANIPULATOR2D 2-D Coefficient Manipulator for NSOLT
    %
    % SVN identifier:
    % $Id: AbstNsoltCoefManipulator2d.m 866 2015-11-24 04:29:42Z sho $
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
    
    properties (PositiveInteger)
        NumberOfChannels      = 4
        NumberOfHalfChannels  = 2
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
        function obj = AbstCnsoltCoefManipulator2d(varargin)
            setProperties(obj,nargin,varargin{:});
            %
            obj.NumberOfHalfChannels = floor(obj.NumberOfChannels/2);
            if mod(obj.NumberOfChannels,2) ~= 0
                obj.NsoltType = 'Type II';
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
            if lenOrd ~= saivdr.dictionary.cnsoltx.AbstCnsoltCoefManipulator2d.DATA_DIMENSION
                error('%s:\n lentgh(PolyPhaseOrder) must be %d.',...
                    id, saivdr.dictionary.cnsoltx.AbstCnsoltCoefManipulator2d.DATA_DIMENSION);
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
            setupParamMtx_(obj);
            %
        end

        function processTunedPropertiesImpl(obj)
            setupParamMtx_(obj);
        end
        
        function stepImpl(obj,coefs,subScale,pmCoefs)
            %
            obj.paramMtxCoefs = pmCoefs;
            %
            if size(coefs,2) ~= (obj.nRows*obj.nCols)
                obj.tmpArray = complex(zeros(size(coefs))); 
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
            pa  = obj.NumberOfHalfChannels;
            ps  = obj.NumberOfChannels - pa;
            %
            paramMtxSzTab_ = zeros(3*sum(ord)+1, 2);
            paramMtxSzTab_(1,:) = [ ps+pa, ps+pa ];
            for iOrd = 1:sum(ord)/2
                paramMtxSzTab_(6*iOrd-4,:) = [ pa pa ];
                paramMtxSzTab_(6*iOrd-3,:) = [ pa pa ];
                paramMtxSzTab_(6*iOrd-2,:) = [ floor(pa/2) 1 ];
                paramMtxSzTab_(6*iOrd-1,:) = [ ps ps ];
                paramMtxSzTab_(6*iOrd  ,:) = [ ps ps ];
                paramMtxSzTab_(6*iOrd+1,:) = [ floor(pa/2) 1 ];
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
        
        function arrayCoefs = lowerBlockRot_(obj,arrayCoefs,iCol,U)
            hLen = obj.NumberOfHalfChannels;
            nRows_ = obj.nRows; 
            indexCol = (iCol-1)*nRows_;
            arrayCoefs(hLen+1:end,indexCol+1:indexCol+nRows_) = ...
                U*arrayCoefs(hLen+1:end,indexCol+1:indexCol+nRows_);
        end
        
        function arrayCoefs = upperBlockRot_(obj,arrayCoefs,iCol,W)
            hLen = obj.NumberOfHalfChannels;
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
