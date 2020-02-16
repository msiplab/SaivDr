classdef NsoltAtomConcatenator2d < ...
        saivdr.dictionary.nsoltx.AbstNsoltCoefManipulator2d %#codegen
    %NSOLTSYNTHESIZER2D 2-D NSOLT Synthesizer
    %
    % SVN identifier:
    % $Id: NsoltAtomConcatenator2d.m 683 2015-05-29 08:22:13Z sho $
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
        function obj = NsoltAtomConcatenator2d(varargin)
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

        function arrayCoefs = stepImpl(obj,arrayCoefs,subScale,pmCoefs)
            stepImpl@saivdr.dictionary.nsoltx.AbstNsoltCoefManipulator2d(obj,arrayCoefs,subScale,pmCoefs);
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
                arrayCoefs(1:hLenU,:)  = ...
                    W0*arrayCoefs(1:hLenU,:);
                arrayCoefs(hLenU+1:end,:) = ...
                    U0*arrayCoefs(hLenU+1:end,:);
            end
            
        end
        
        function arrayCoefs = fullAtomCncTypeI_(obj,arrayCoefs)
            %
            isPeriodicExt = obj.IsPeriodicExt; % BoundaryOperation = 'Circular'
            %
            ordY = obj.PolyPhaseOrder(saivdr.dictionary.utility.Direction.VERTICAL);
            ordX = obj.PolyPhaseOrder(saivdr.dictionary.utility.Direction.HORIZONTAL);
            numOfPMtx = size(obj.paramMtxSzTab,1);
            %
            hOrdY = uint32(ordY/2);
            if hOrdY > 0
                arrayCoefs = permuteCoefs_(obj,arrayCoefs);
                for iOrd = uint32(1):hOrdY  % Vertical process
                    paramMtx1 = getParamMtx_(obj,numOfPMtx-2*iOrd+2);
                    paramMtx2 = getParamMtx_(obj,numOfPMtx-2*iOrd+1);
                    %
                    arrayCoefs = atomCncTypeI_(obj,arrayCoefs,paramMtx1,paramMtx2,isPeriodicExt);
                end
                arrayCoefs = ipermuteCoefs_(obj,arrayCoefs);
            end
            %
            hOrdX = uint32(ordX/2);
            for iOrd = uint32(1):hOrdX % Horizontal process
                paramMtx1 = getParamMtx_(obj,numOfPMtx-2*(hOrdY+iOrd)+2);
                paramMtx2 = getParamMtx_(obj,numOfPMtx-2*(hOrdY+iOrd)+1);
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
            numOfPMtx = size(obj.paramMtxSzTab,1);
            %
            hOrdY = uint32(ordY/2);
            if hOrdY > 0
                arrayCoefs = permuteCoefs_(obj,arrayCoefs);
                for iOrd = uint32(1):hOrdY % Vertical process
                    paramMtx1 = getParamMtx_(obj,numOfPMtx-2*iOrd+1); % W
                    paramMtx2 = getParamMtx_(obj,numOfPMtx-2*iOrd+2); % U
                    %
                    if isPsGtPa
                        arrayCoefs = atomCncTypeIIPsGtPa_(obj,arrayCoefs,paramMtx1,paramMtx2,isPeriodicExt);
                    else
                        arrayCoefs = atomCncTypeIIPsLtPa_(obj,arrayCoefs,paramMtx1,paramMtx2,isPeriodicExt);                        
                    end
                end
                arrayCoefs = ipermuteCoefs_(obj,arrayCoefs);
            end
            %
            hOrdX = uint32(ordX/2);
            for iOrd = uint32(1):hOrdX % Horizontal process
                paramMtx1 = getParamMtx_(obj,numOfPMtx-2*(hOrdY+iOrd)+1); % W
                paramMtx2 = getParamMtx_(obj,numOfPMtx-2*(hOrdY+iOrd)+2); % U
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
            
            % Phase 1
            Ux2 = paramMtx1.';
            % Lower channel rotation
            arrayCoefs(hLen+1:end,:) = Ux2*arrayCoefs(hLen+1:end,:);
            arrayCoefs = blockButterflyTypeI_(obj,arrayCoefs);
            arrayCoefs = rightShiftUpperCoefs_(obj,arrayCoefs);
            arrayCoefs = blockButterflyTypeI_(obj,arrayCoefs);
            arrayCoefs = arrayCoefs/2.0;
            
            % Phase 2
            Ux1 = paramMtx2.';
            I = eye(size(Ux1));
            % Lower channel rotation
            if isPeriodicExt
                arrayCoefs(hLen+1:end,:) = Ux1*arrayCoefs(hLen+1:end,:);
            else
                for iCol = 1:obj.nCols
                    if iCol == 1 
                        U = -I;
                    else
                        U = Ux1;
                    end
                    arrayCoefs = lowerBlockRot_(obj,arrayCoefs,iCol,U);
                end
            end
            arrayCoefs = blockButterflyTypeI_(obj,arrayCoefs);
            arrayCoefs = leftShiftLowerCoefs_(obj,arrayCoefs);
            arrayCoefs = blockButterflyTypeI_(obj,arrayCoefs);
            arrayCoefs = arrayCoefs/2.0;
        end
        
        function arrayCoefs = atomCncTypeIIPsGtPa_(obj,arrayCoefs,paramMtx1,paramMtx2,isPeriodicExt)
            hLen = obj.NumberOfSymmetricChannels;
            
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
                for iCol = 1:obj.nCols
                    if iCol == 1
                        U = -I;
                    else
                        U = Ux;
                    end
                    arrayCoefs = lowerBlockRot_(obj,arrayCoefs,iCol,U);
                end
            end
            arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs);
            arrayCoefs = leftShiftLowerCoefs_(obj,arrayCoefs);
            arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs);
            arrayCoefs = arrayCoefs/2.0;
        end
        
      function arrayCoefs = atomCncTypeIIPsLtPa_(obj,arrayCoefs,paramMtx1,paramMtx2,isPeriodicExt)
            hLen = obj.NumberOfSymmetricChannels;
            
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
                for iCol = 1:obj.nCols
                    if iCol == 1
                        W = -I;
                    else
                        W = Wx;
                    end
                    arrayCoefs = upperBlockRot_(obj,arrayCoefs,iCol,W);
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
            nRows_ = obj.nRows;
            %
            lowerCoefsPost = arrayCoefs(hLenMn+1:end,1:nRows_);
            arrayCoefs(hLenMn+1:end,1:end-nRows_) = ...
                arrayCoefs(hLenMn+1:end,nRows_+1:end);
            arrayCoefs(hLenMn+1:end,end-nRows_+1:end) = ...
                lowerCoefsPost;            
        end
        
        function arrayCoefs = rightShiftUpperCoefs_(obj,arrayCoefs)
            hLenMx = max([ obj.NumberOfSymmetricChannels
                obj.NumberOfAntisymmetricChannels]);
            nRows_ = obj.nRows;
            %
            upperCoefsPre = arrayCoefs(1:hLenMx,end-nRows_+1:end);
            arrayCoefs(1:hLenMx,nRows_+1:end) = ...
                arrayCoefs(1:hLenMx,1:end-nRows_);
            arrayCoefs(1:hLenMx,1:nRows_) = ...
                upperCoefsPre;
        end
    end
    
end
