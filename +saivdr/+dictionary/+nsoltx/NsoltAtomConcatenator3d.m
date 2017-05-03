classdef NsoltAtomConcatenator3d < ...
        saivdr.dictionary.nsoltx.AbstNsoltCoefManipulator3d %#codegen
    %NSOLTSYNTHESIZER3D 3-D NSOLT Synthesizer
    %
    % SVN identifier:
    % $Id: NsoltAtomConcatenator3d.m 683 2015-05-29 08:22:13Z sho $
    %
    % Requirements: MATLAB R2015b
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
    % http://msiplab.eng.niigata-u.ac.jp/
    %
    
    methods
        
        % Constructor
        function obj = NsoltAtomConcatenator3d(varargin)
            obj = obj@saivdr.dictionary.nsoltx.AbstNsoltCoefManipulator3d(varargin{:});
        end
        
    end
    
    methods (Access = protected)
        
        function s = saveObjectImpl(obj)
            % Call the base class method
            s = saveObjectImpl@saivdr.dictionary.nsoltx.AbstNsoltCoefManipulator3d(obj);
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            % Call base class method to load public properties
            loadObjectImpl@saivdr.dictionary.nsoltx.AbstNsoltCoefManipulator3d(obj,s,wasLocked);
        end

        function arrayCoefs = stepImpl(obj,arrayCoefs,subScale,pmCoefs)
            stepImpl@saivdr.dictionary.nsoltx.AbstNsoltCoefManipulator3d(obj,arrayCoefs,subScale,pmCoefs);
            %
            if strcmp(obj.NsoltType,'Type I')
                arrayCoefs = fullAtomCncTypeI_(obj,arrayCoefs);
            else
                arrayCoefs = fullAtomCncTypeII_(obj,arrayCoefs);
            end
            %
            arrayCoefs = finalStep_(obj,arrayCoefs);
        end
        
    end
    
    methods ( Access = private )
        
        function arrayCoefs = finalStep_(obj,arrayCoefs)

            hLenU = obj.NumberOfSymmetricChannels;
            %
            if ~isempty(obj.paramMtxCoefs)
                W0 = getParamMtx_(obj,uint32(1)).';
                U0 = getParamMtx_(obj,uint32(2)).';
                arrayCoefs(1:hLenU,:)     = W0*arrayCoefs(1:hLenU,:);
                arrayCoefs(hLenU+1:end,:) = U0*arrayCoefs(hLenU+1:end,:);
            end
            
        end
        
        function arrayCoefs = fullAtomCncTypeI_(obj,arrayCoefs)
            %
            isPeriodicExt = obj.IsPeriodicExt; % BoundaryOperation = 'Circular'
            %
            ordY = obj.PolyPhaseOrder(saivdr.dictionary.utility.Direction.VERTICAL);
            ordX = obj.PolyPhaseOrder(saivdr.dictionary.utility.Direction.HORIZONTAL);
            ordZ = obj.PolyPhaseOrder(saivdr.dictionary.utility.Direction.DEPTH);
            numOfPMtx = size(obj.paramMtxSzTab,1);

            % Vertical concatenation
            arrayCoefs = ipermuteCoefs_(obj,arrayCoefs); % Y X Z -> X Z Y
            for iOrd = uint32(1):uint32(ordY/2)
                paramMtx1 = getParamMtx_(obj,numOfPMtx-2*iOrd+2); % U2 
                paramMtx2 = getParamMtx_(obj,numOfPMtx-2*iOrd+1); % U1
                %
                arrayCoefs = atomCncTypeI_(obj,arrayCoefs,paramMtx1,paramMtx2,isPeriodicExt);
            end
            
            % Horizontal concatenation
            arrayCoefs = ipermuteCoefs_(obj,arrayCoefs); % X Z Y -> Z Y X 
            for iOrd = uint32(1):uint32(ordX/2);
                paramMtx1 = getParamMtx_(obj,numOfPMtx-2*iOrd-ordY+2);
                paramMtx2 = getParamMtx_(obj,numOfPMtx-2*iOrd-ordY+1);
                %
                arrayCoefs = atomCncTypeI_(obj,arrayCoefs,paramMtx1,paramMtx2,isPeriodicExt);
            end

            % Depth concatenation
            arrayCoefs = ipermuteCoefs_(obj,arrayCoefs); % Z Y X -> Y X Z
            for iOrd = uint32(1):uint32(ordZ/2)
                paramMtx1 = getParamMtx_(obj,numOfPMtx-2*iOrd-ordY-ordX+2);
                paramMtx2 = getParamMtx_(obj,numOfPMtx-2*iOrd-ordY-ordX+1);
                %
                arrayCoefs = atomCncTypeI_(obj,arrayCoefs,paramMtx1,paramMtx2,isPeriodicExt);
            end

        end
        
        function arrayCoefs = fullAtomCncTypeII_(obj,arrayCoefs)
            %
            isPeriodicExt = obj.IsPeriodicExt; % BoundaryOperation = 'Circular';
            isPsGtPa      = obj.IsPsGreaterThanPa;            
            %
            ordY = obj.PolyPhaseOrder(saivdr.dictionary.utility.Direction.VERTICAL);
            ordX = obj.PolyPhaseOrder(saivdr.dictionary.utility.Direction.HORIZONTAL);
            ordZ = obj.PolyPhaseOrder(saivdr.dictionary.utility.Direction.DEPTH);
            numOfPMtx = size(obj.paramMtxSzTab,1);
            
            % Vertical concatenation
            arrayCoefs = ipermuteCoefs_(obj,arrayCoefs); % Y X Z -> Z Y X
            for iOrd = uint32(1):uint32(ordY/2);
                paramMtx1 = getParamMtx_(obj,numOfPMtx-2*iOrd+1); % W
                paramMtx2 = getParamMtx_(obj,numOfPMtx-2*iOrd+2); % U
                %
                if isPsGtPa
                    arrayCoefs = atomCncTypeIIPsGtPa_(obj,arrayCoefs,paramMtx1,paramMtx2,isPeriodicExt);
                else
                    arrayCoefs = atomCncTypeIIPsLtPa_(obj,arrayCoefs,paramMtx1,paramMtx2,isPeriodicExt);
                end
            end
            
            % Horizontal concatenation
            arrayCoefs = ipermuteCoefs_(obj,arrayCoefs); % Z Y X -> X Z Y
            for iOrd = uint32(1):uint32(ordX/2)
                paramMtx1 = getParamMtx_(obj,numOfPMtx-2*iOrd-ordY+1); % W
                paramMtx2 = getParamMtx_(obj,numOfPMtx-2*iOrd-ordY+2); % U
                %
                if isPsGtPa
                    arrayCoefs = atomCncTypeIIPsGtPa_(obj,arrayCoefs,paramMtx1,paramMtx2,isPeriodicExt);
                else
                    arrayCoefs = atomCncTypeIIPsLtPa_(obj,arrayCoefs,paramMtx1,paramMtx2,isPeriodicExt);
                end
            end
            
            % Depth concatenation
            arrayCoefs = ipermuteCoefs_(obj,arrayCoefs); % X Z Y -> Y X Z
            for iOrd = uint32(1):uint32(ordZ/2)
                paramMtx1 = getParamMtx_(obj,numOfPMtx-2*iOrd-ordY-ordX+1); % W
                paramMtx2 = getParamMtx_(obj,numOfPMtx-2*iOrd-ordY-ordX+2); % U
                %
                if isPsGtPa
                    arrayCoefs = atomCncTypeIIPsGtPa_(obj,arrayCoefs,paramMtx1,paramMtx2,isPeriodicExt);
                else
                    arrayCoefs = atomCncTypeIIPsLtPa_(obj,arrayCoefs,paramMtx1,paramMtx2,isPeriodicExt);
                end
            end
            
        end
        
        function arrayCoefs = atomCncTypeI_(obj,arrayCoefs,paramMtx1,paramMtx2,isPeriodicExt)
            hLen = obj.NumberOfSymmetricChannels;
            nLays_ = obj.nLays;

            % Phase 1
            Ux2 = paramMtx1.';
            % Lower channel rotation
            arrayCoefs(hLen+1:end,:) = Ux2*arrayCoefs(hLen+1:end,:);           
            arrayCoefs = blockButterflyTypeI_(obj,arrayCoefs);
            arrayCoefs = rightShiftUpperCoefs_(obj,arrayCoefs);
            arrayCoefs = blockButterflyTypeI_(obj,arrayCoefs);
            arrayCoefs = arrayCoefs/2.0;

            %arrayCoefs
            % Phase 2
            Ux1 = paramMtx2.';
            I = eye(size(Ux1));
            % Lower channel rotation
            if isPeriodicExt
                arrayCoefs(hLen+1:end,:) = Ux1*arrayCoefs(hLen+1:end,:);
            else
                for iLay = 1:nLays_
                    if iLay == 1
                        U = -I;
                    else
                        U = Ux1;
                    end
                    arrayCoefs = lowerBlockRot_(obj,arrayCoefs,iLay,U);
                end
            end
            arrayCoefs = blockButterflyTypeI_(obj,arrayCoefs);
            arrayCoefs = leftShiftLowerCoefs_(obj,arrayCoefs);
            arrayCoefs = blockButterflyTypeI_(obj,arrayCoefs);
            arrayCoefs = arrayCoefs/2.0;
        end
        
        function arrayCoefs = atomCncTypeIIPsGtPa_(obj,arrayCoefs,paramMtx1,paramMtx2,isPeriodicExt)
            hLen = obj.NumberOfSymmetricChannels;
            nLays_ = obj.nLays;
            
            % Phase 1
            Wx = paramMtx1.';
            % Upper channel rotation
            arrayCoefs(1:hLen,:) = Wx*arrayCoefs(1:hLen,:);
            arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs);
            arrayCoefs = rightShiftUpperCoefs_(obj,arrayCoefs);
            arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs);
            arrayCoefs = arrayCoefs/2.0;
            
            % Phase 2
            Ux = paramMtx2.';
            I = eye(size(Ux));
            % Lower channel rotation
            if isPeriodicExt
                arrayCoefs(hLen+1:end,:) = Ux*arrayCoefs(hLen+1:end,:);
            else
                for iLay = 1:nLays_
                    if iLay == 1
                        U = -I;
                    else
                        U = Ux;
                    end
                    arrayCoefs = lowerBlockRot_(obj,arrayCoefs,iLay,U);
                end
            end
            arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs);
            arrayCoefs = leftShiftLowerCoefs_(obj,arrayCoefs);
            arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs);
            arrayCoefs = arrayCoefs/2.0;
        end
        
        function arrayCoefs = atomCncTypeIIPsLtPa_(obj,arrayCoefs,paramMtx1,paramMtx2,isPeriodicExt)
            hLen = obj.NumberOfSymmetricChannels;
            nLays_ = obj.nLays;
            
            % Phase 1
            Ux = paramMtx2.';
            % Lower channel rotation
            arrayCoefs(hLen+1:end,:) = Ux*arrayCoefs(hLen+1:end,:);
            arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs);
            arrayCoefs = rightShiftUpperCoefs_(obj,arrayCoefs);
            arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs);
            arrayCoefs = arrayCoefs/2.0;
            
            % Phase 2
            Wx = paramMtx1.';
            I = eye(size(Wx));
            % Upper channel rotation
            if isPeriodicExt
                arrayCoefs(1:hLen,:) = Wx*arrayCoefs(1:hLen,:);
            else
                for iLay = 1:nLays_
                    if iLay == 1
                        W = -I;
                    else
                        W = Wx;
                    end
                    arrayCoefs = upperBlockRot_(obj,arrayCoefs,iLay,W);
                end
            end
            arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs);
            arrayCoefs = leftShiftLowerCoefs_(obj,arrayCoefs);
            arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs);
            arrayCoefs = arrayCoefs/2.0;
        end        
        
        function arrayCoefs = leftShiftLowerCoefs_(obj,arrayCoefs)
            hLenMn = min([ obj.NumberOfSymmetricChannels
                obj.NumberOfAntisymmetricChannels]);
            nRowsxnCols_ = obj.nRows*obj.nCols;
            %
            lowerCoefsPost = arrayCoefs(hLenMn+1:end,1:nRowsxnCols_);
            arrayCoefs(hLenMn+1:end,1:end-nRowsxnCols_) = ...
                arrayCoefs(hLenMn+1:end,nRowsxnCols_+1:end);
            arrayCoefs(hLenMn+1:end,end-nRowsxnCols_+1:end) = ...
                lowerCoefsPost;
        end
        
        function arrayCoefs = rightShiftUpperCoefs_(obj,arrayCoefs)
             hLenMx = max([ obj.NumberOfSymmetricChannels
                 obj.NumberOfAntisymmetricChannels]);
             nRowsxnCols_ = obj.nRows*obj.nCols;
             %
             upperCoefsPre = arrayCoefs(1:hLenMx,end-nRowsxnCols_+1:end);
             arrayCoefs(1:hLenMx,nRowsxnCols_+1:end) = ...
                 arrayCoefs(1:hLenMx,1:end-nRowsxnCols_);
             arrayCoefs(1:hLenMx,1:nRowsxnCols_) = ...
                 upperCoefsPre;
        end
    end
    
end
