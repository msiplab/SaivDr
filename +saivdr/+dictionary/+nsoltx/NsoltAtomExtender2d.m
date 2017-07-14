classdef NsoltAtomExtender2d <  ...
        saivdr.dictionary.nsoltx.AbstNsoltCoefManipulator2d %#codegen
    %NSOLTATOMEXTENDER2D 2-D Atom Extender for NSOLT
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
    
    methods
        
        % Constructor
        function obj = NsoltAtomExtender2d(varargin)
            obj = obj@saivdr.dictionary.nsoltx.AbstNsoltCoefManipulator2d(varargin{:});
        end
        
    end
    
    methods (Access = protected)
        
        function s = saveObjectImpl(obj)
            % Call the base class method
            s = saveObjectImpl@saivdr.dictionary.nsoltx.AbstNsoltCoefManipulator2d(obj);
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            % Call base class method to load public properties
            loadObjectImpl@saivdr.dictionary.nsoltx.AbstNsoltCoefManipulator2d(obj,s,wasLocked);
        end
        
        function arrayCoefs = stepImpl(obj, arrayCoefs, subScale, pmCoefs)
            stepImpl@saivdr.dictionary.nsoltx.AbstNsoltCoefManipulator2d(obj,arrayCoefs,subScale,pmCoefs);        
            %
            arrayCoefs = initialStep_(obj,arrayCoefs);
            %
            if strcmp(obj.NsoltType,'Type I')
                arrayCoefs = fullAtomExtTypeI_(obj,arrayCoefs);
            else
                arrayCoefs = fullAtomExtTypeII_(obj,arrayCoefs);
            end
            
        end
        
    end
    
    methods ( Access = private )
        
        function arrayCoefs = initialStep_(obj,arrayCoefs)
            
            hLenU = obj.NumberOfSymmetricChannels;            
            %
            if ~isempty(obj.paramMtxCoefs)
                W0 = getParamMtx_(obj,uint32(1));
                U0 = getParamMtx_(obj,uint32(2));
                arrayCoefs(1:hLenU,:) ...
                     = W0*arrayCoefs(1:hLenU,:);
                arrayCoefs(hLenU+1:end,:) = ...
                    U0*arrayCoefs(hLenU+1:end,:);
            end

        end
        
        function arrayCoefs = fullAtomExtTypeI_(obj,arrayCoefs)
            %
            isPeriodicExt = obj.IsPeriodicExt; % BoundaryOperation = 'Circular'
            %
            ordY = obj.PolyPhaseOrder(saivdr.dictionary.utility.Direction.VERTICAL);
            ordX = obj.PolyPhaseOrder(saivdr.dictionary.utility.Direction.HORIZONTAL);
            
            % Width extension
            for iOrd = uint32(1):uint32(ordX/2)
                paramMtx1 = getParamMtx_(obj,2*iOrd+1);
                paramMtx2 = getParamMtx_(obj,2*iOrd+2);
                %
                arrayCoefs = supportExtTypeI_(obj,arrayCoefs,paramMtx1,paramMtx2,isPeriodicExt);
            end
            
            % Height extension
            if ordY > 0
                arrayCoefs = permuteCoefs_(obj,arrayCoefs);
                for iOrd = uint32(1):uint32(ordY/2)
                    paramMtx1 = getParamMtx_(obj,2*iOrd+ordX+1);
                    paramMtx2 = getParamMtx_(obj,2*iOrd+ordX+2);
                    %
                    arrayCoefs = supportExtTypeI_(obj,arrayCoefs,paramMtx1,paramMtx2,isPeriodicExt);
                end
                arrayCoefs = ipermuteCoefs_(obj,arrayCoefs);
            end
        end
        
        function arrayCoefs = fullAtomExtTypeII_(obj,arrayCoefs)
            %
            isPeriodicExt = obj.IsPeriodicExt; % BoundaryOperation = 'Circular'
            isPsGtPa      = obj.IsPsGreaterThanPa;            
            %
            ordY = obj.PolyPhaseOrder(saivdr.dictionary.utility.Direction.VERTICAL);
            ordX = obj.PolyPhaseOrder(saivdr.dictionary.utility.Direction.HORIZONTAL);
            
            % Width extension
            for iOrd = uint32(1):uint32(ordX/2)
                paramMtx1 = getParamMtx_(obj,2*iOrd+1); % W
                paramMtx2 = getParamMtx_(obj,2*iOrd+2); % U
                %
                if isPsGtPa
                    arrayCoefs = supportExtTypeIIPsGtPa_(obj,arrayCoefs,paramMtx1,paramMtx2,isPeriodicExt);
                else
                    arrayCoefs = supportExtTypeIIPsLtPa_(obj,arrayCoefs,paramMtx1,paramMtx2,isPeriodicExt);
                end
            end
            
            % Height extension
            if ordY > 0
                arrayCoefs = permuteCoefs_(obj,arrayCoefs);
                for iOrd = uint32(1):uint32(ordY/2)  % Vertical process
                    paramMtx1 = getParamMtx_(obj,2*iOrd+ordX+1); % W
                    paramMtx2 = getParamMtx_(obj,2*iOrd+ordX+2); % U
                    %
                    if isPsGtPa
                        arrayCoefs = supportExtTypeIIPsGtPa_(obj,arrayCoefs,paramMtx1,paramMtx2,isPeriodicExt);
                    else
                        arrayCoefs = supportExtTypeIIPsLtPa_(obj,arrayCoefs,paramMtx1,paramMtx2,isPeriodicExt);
                    end
                end
                arrayCoefs = ipermuteCoefs_(obj,arrayCoefs);
            end
        end
        
        function arrayCoefs = supportExtTypeI_(obj,arrayCoefs,paramMtx1,paramMtx2,isPeriodicExt)
            hLen = obj.NumberOfSymmetricChannels;
            nCols_ = obj.nCols;
            
            % Phase 1
            Ux1 = paramMtx1;
            I = eye(size(Ux1));
            arrayCoefs = blockButterflyTypeI_(obj,arrayCoefs);
            arrayCoefs = rightShiftLowerCoefs_(obj,arrayCoefs);
            arrayCoefs = blockButterflyTypeI_(obj,arrayCoefs);
            arrayCoefs = arrayCoefs/2.0;
            % Lower channel rotation
            if isPeriodicExt
                arrayCoefs(hLen+1:end,:) = Ux1*arrayCoefs(hLen+1:end,:);
            else
                for iCol = 1:nCols_
                    if iCol == 1 %&& ~isPeriodicExt
                        U = -I;
                    else
                        U = Ux1;
                    end
                    arrayCoefs = lowerBlockRot_(obj,arrayCoefs,iCol,U);
                end
            end
            
            % Phase 2
            Ux2 = paramMtx2;
            arrayCoefs = blockButterflyTypeI_(obj,arrayCoefs);
            arrayCoefs = leftShiftUpperCoefs_(obj,arrayCoefs);
            arrayCoefs = blockButterflyTypeI_(obj,arrayCoefs);
            arrayCoefs = arrayCoefs/2.0;
            % Lower channel rotation
            arrayCoefs(hLen+1:end,:) = Ux2*arrayCoefs(hLen+1:end,:);
        end
        
        function arrayCoefs = supportExtTypeIIPsGtPa_(obj,arrayCoefs,paramMtx1,paramMtx2,isPeriodicExt)
            hLen = obj.NumberOfSymmetricChannels;
            nCols_ = obj.nCols;
            
            % Phase 1
            Ux = paramMtx2;
            I = eye(size(Ux));
            arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs);
            arrayCoefs = rightShiftLowerCoefs_(obj,arrayCoefs);
            arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs);
            arrayCoefs = arrayCoefs/2.0;
            % Lower channel rotation
            if isPeriodicExt
                arrayCoefs(hLen+1:end,:) = Ux*arrayCoefs(hLen+1:end,:);
            else
                for iCol = 1:nCols_
                    if iCol == 1
                        U = -I;
                    else
                        U = Ux;
                    end
                    arrayCoefs = lowerBlockRot_(obj,arrayCoefs,iCol,U);
                end
            end
            
            % Phase 2
            Wx = paramMtx1;
            arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs);
            arrayCoefs = leftShiftUpperCoefs_(obj,arrayCoefs);
            arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs);
            arrayCoefs = arrayCoefs/2.0;
            % Upper channel rotation
            arrayCoefs(1:hLen,:) = Wx*arrayCoefs(1:hLen,:);
        end
        
        function arrayCoefs = supportExtTypeIIPsLtPa_(obj,arrayCoefs,paramMtx1,paramMtx2,isPeriodicExt)
            hLen = obj.NumberOfSymmetricChannels;
            nCols_ = obj.nCols;
            
            % Phase 1
            Wx = paramMtx1;
            I = eye(size(Wx));
            arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs);
            arrayCoefs = rightShiftLowerCoefs_(obj,arrayCoefs);
            arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs);
            arrayCoefs = arrayCoefs/2.0;
            % Upper channel rotation
            if isPeriodicExt
                arrayCoefs(1:hLen,:) = Wx*arrayCoefs(1:hLen,:);
            else
                for iCol = 1:nCols_
                    if iCol == 1
                        W = -I;
                    else
                        W = Wx;
                    end
                    arrayCoefs = upperBlockRot_(obj,arrayCoefs,iCol,W);
                end
            end
            
            % Phase 2
            Ux = paramMtx2;
            arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs);
            arrayCoefs = leftShiftUpperCoefs_(obj,arrayCoefs);
            arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs);
            arrayCoefs = arrayCoefs/2.0;
            % Lower channel rotation
            arrayCoefs(hLen+1:end,:) = Ux*arrayCoefs(hLen+1:end,:);
        end
        
        function arrayCoefs = rightShiftLowerCoefs_(obj,arrayCoefs)
            hLenMn = max([ obj.NumberOfSymmetricChannels
                obj.NumberOfAntisymmetricChannels]);
            nRows_ = obj.nRows;
            %
            lowerCoefsPre = arrayCoefs(hLenMn+1:end,end-nRows_+1:end);
            arrayCoefs(hLenMn+1:end,nRows_+1:end) = ...
                arrayCoefs(hLenMn+1:end,1:end-nRows_);
            arrayCoefs(hLenMn+1:end,1:nRows_) = ...
                lowerCoefsPre;
        end
        
        function arrayCoefs = leftShiftUpperCoefs_(obj,arrayCoefs)
            hLenMx = min([ obj.NumberOfSymmetricChannels
                obj.NumberOfAntisymmetricChannels]);
            nRows_ = obj.nRows;
            %
            upperCoefsPost = arrayCoefs(1:hLenMx,1:nRows_);
            arrayCoefs(1:hLenMx,1:end-nRows_) = ...
                arrayCoefs(1:hLenMx,nRows_+1:end);
            arrayCoefs(1:hLenMx,end-nRows_+1:end) = ...
                upperCoefsPost;            
        end
    end
    
end
