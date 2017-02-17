classdef CnsoltAtomExtender3d <  ...
        saivdr.dictionary.cnsoltx.AbstCnsoltCoefManipulator3d %#codegen
    %NSOLTANALYZER3D 3-D NSOLT Atom Extender
    %
    % SVN identifier:
    % $Id: NsoltAtomExtender3d.m 683 2015-05-29 08:22:13Z sho $
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
    
    methods
        
        % Constructor
        function obj = CnsoltAtomExtender3d(varargin)
            obj = obj@saivdr.dictionary.cnsoltx.AbstCnsoltCoefManipulator3d(varargin{:});
        end
        
    end
    
    methods (Access = protected)
        
        function s = saveObjectImpl(obj)
            % Call the base class method
            s = saveObjectImpl@saivdr.dictionary.cnsoltx.AbstCnsoltCoefManipulator3d(obj);
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            % Call base class method to load public properties
            loadObjectImpl@saivdr.dictionary.cnsoltx.AbstCnsoltCoefManipulator3d(obj,s,wasLocked);
        end
        
        function arrayCoefs = stepImpl(obj, arrayCoefs, subScale, pmCoefs)
            stepImpl@saivdr.dictionary.cnsoltx.AbstCnsoltCoefManipulator3d(obj,arrayCoefs,subScale,pmCoefs);
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
            %
            if ~isempty(obj.paramMtxCoefs)
                V0 = getParamMtx_(obj,uint32(1));
                arrayCoefs = V0(1:obj.NumberOfChannels,:)*arrayCoefs;
            end

        end
        
        function arrayCoefs = fullAtomExtTypeI_(obj,arrayCoefs)
            %
            %isPeriodicExt = obj.IsPeriodicExt; % BoundaryOperation = 'Circular'
            isPeriodicExt = true;
            %
            ordY = obj.PolyPhaseOrder(saivdr.dictionary.utility.Direction.VERTICAL);
            ordX = obj.PolyPhaseOrder(saivdr.dictionary.utility.Direction.HORIZONTAL);
            ordZ = obj.PolyPhaseOrder(saivdr.dictionary.utility.Direction.DEPTH);
            
            % Depth extension
            for iOrd = uint32(1):uint32(ordZ/2)
                paramMtxW1 = getParamMtx_(obj,6*iOrd-4);
                paramMtxU1 = getParamMtx_(obj,6*iOrd-3);
                paramAngB1 = getParamMtx_(obj,6*iOrd-2);
                paramMtxW2 = getParamMtx_(obj,6*iOrd-1);
                paramMtxU2 = getParamMtx_(obj,6*iOrd  );
                paramAngB2 = getParamMtx_(obj,6*iOrd+1);
                %
                arrayCoefs = supportExtTypeI_(obj,arrayCoefs,...
                    paramMtxW1,paramMtxU1,paramAngB1,...
                    paramMtxW2,paramMtxU2,paramAngB2,...
                    isPeriodicExt);
            end
            
            % Width extension
            arrayCoefs = permuteCoefs_(obj,arrayCoefs); % Y X Z -> Z Y X
            for iOrd = uint32(1):uint32(ordX/2)
                paramMtxW1 = getParamMtx_(obj,6*iOrd+3*ordZ-4);
                paramMtxU1 = getParamMtx_(obj,6*iOrd+3*ordZ-3);
                paramAngB1 = getParamMtx_(obj,6*iOrd+3*ordZ-2);
                paramMtxW2 = getParamMtx_(obj,6*iOrd+3*ordZ-1);
                paramMtxU2 = getParamMtx_(obj,6*iOrd+3*ordZ  );
                paramAngB2 = getParamMtx_(obj,6*iOrd+3*ordZ+1);
                %
                arrayCoefs = supportExtTypeI_(obj,arrayCoefs,...
                    paramMtxW1,paramMtxU1,paramAngB1,...
                    paramMtxW2,paramMtxU2,paramAngB2,...
                    isPeriodicExt);
            end
            
            % Height extension
            arrayCoefs = permuteCoefs_(obj,arrayCoefs); % Z Y X -> X Z Y
            for iOrd = uint32(1):uint32(ordY/2)
                paramMtxW1 = getParamMtx_(obj,6*iOrd+3*(ordZ+ordX)-4);
                paramMtxU1 = getParamMtx_(obj,6*iOrd+3*(ordZ+ordX)-3);
                paramAngB1 = getParamMtx_(obj,6*iOrd+3*(ordZ+ordX)-2);
                paramMtxW2 = getParamMtx_(obj,6*iOrd+3*(ordZ+ordX)-1);
                paramMtxU2 = getParamMtx_(obj,6*iOrd+3*(ordZ+ordX)  );
                paramAngB2 = getParamMtx_(obj,6*iOrd+3*(ordZ+ordX)+1);
                %
                arrayCoefs = supportExtTypeI_(obj,arrayCoefs,...
                    paramMtxW1,paramMtxU1,paramAngB1,...
                    paramMtxW2,paramMtxU2,paramAngB2,...
                    isPeriodicExt);
            end
            
            % To original coordinate
            arrayCoefs = permuteCoefs_(obj,arrayCoefs); % X Z Y -> Y X Z
        end
        
        function arrayCoefs = fullAtomExtTypeII_(obj,arrayCoefs)
            %
            %isPeriodicExt = obj.IsPeriodicExt; % BoundaryOperation = 'Circular'
            isPeriodicExt = true;
            %
            ordY = obj.PolyPhaseOrder(saivdr.dictionary.utility.Direction.VERTICAL);
            ordX = obj.PolyPhaseOrder(saivdr.dictionary.utility.Direction.HORIZONTAL);
            ordZ = obj.PolyPhaseOrder(saivdr.dictionary.utility.Direction.DEPTH);
            
            % Depth extension
            for iOrd = uint32(1):uint32(ordZ/2) 
                paramMtxW1 = getParamMtx_(obj,6*iOrd-4);
                paramMtxU1 = getParamMtx_(obj,6*iOrd-3);
                paramAngB1 = getParamMtx_(obj,6*iOrd-2);
                paramMtxW2 = getParamMtx_(obj,6*iOrd-1);
                paramMtxU2 = getParamMtx_(obj,6*iOrd  );
                paramAngB2 = getParamMtx_(obj,6*iOrd+1);
                %
                arrayCoefs = supportExtTypeII_(obj,arrayCoefs,...
                    paramMtxW1,paramMtxU1,paramAngB1,...
                    paramMtxW2,paramMtxU2,paramAngB2,...
                    isPeriodicExt);
            end
            
            % Width extension
            arrayCoefs = permuteCoefs_(obj,arrayCoefs); % Y X Z -> Z Y X
            for iOrd = uint32(1):uint32(ordX/2)
                paramMtxW1 = getParamMtx_(obj,6*iOrd+3*ordZ-4);
                paramMtxU1 = getParamMtx_(obj,6*iOrd+3*ordZ-3);
                paramAngB1 = getParamMtx_(obj,6*iOrd+3*ordZ-2);
                paramMtxW2 = getParamMtx_(obj,6*iOrd+3*ordZ-1);
                paramMtxU2 = getParamMtx_(obj,6*iOrd+3*ordZ  );
                paramAngB2 = getParamMtx_(obj,6*iOrd+3*ordZ+1);
                %
                arrayCoefs = supportExtTypeII_(obj,arrayCoefs,...
                    paramMtxW1,paramMtxU1,paramAngB1,...
                    paramMtxW2,paramMtxU2,paramAngB2,...
                    isPeriodicExt);
            end
            
            % Height extension
            arrayCoefs = permuteCoefs_(obj,arrayCoefs); % Z Y X -> X Z Y
            for iOrd = uint32(1):uint32(ordY/2)
                paramMtxW1 = getParamMtx_(obj,6*iOrd+3*(ordZ+ordX)-4);
                paramMtxU1 = getParamMtx_(obj,6*iOrd+3*(ordZ+ordX)-3);
                paramAngB1 = getParamMtx_(obj,6*iOrd+3*(ordZ+ordX)-2);
                paramMtxW2 = getParamMtx_(obj,6*iOrd+3*(ordZ+ordX)-1);
                paramMtxU2 = getParamMtx_(obj,6*iOrd+3*(ordZ+ordX)  );
                paramAngB2 = getParamMtx_(obj,6*iOrd+3*(ordZ+ordX)+1);
                %
                arrayCoefs = supportExtTypeII_(obj,arrayCoefs,...
                    paramMtxW1,paramMtxU1,paramAngB1,...
                    paramMtxW2,paramMtxU2,paramAngB2,...
                    isPeriodicExt);
            end
            
            % To original coordinate
            arrayCoefs = permuteCoefs_(obj,arrayCoefs); % X Z Y -> Y X Z
        end
        
        function arrayCoefs = supportExtTypeI_(obj,arrayCoefs,paramMtx1,paramMtx2,paramMtx3,paramMtx4,paramMtx5,paramMtx6,isPeriodicExt)
            hLen = obj.NumberOfHalfChannels;
            nLays_ = obj.nLays;
            
            % Phase 1
            Wx1 = paramMtx1;
            Ux1 = paramMtx2;
            B1 = saivdr.dictionary.cnsoltx.mexsrcs.AbstCplxBuildingBlock.butterflyMtx(hLen,paramMtx3);
            I = eye(size(Ux1));
%             arrayCoefs = blockButterflyTypeI_(obj,arrayCoefs);
            arrayCoefs = B1'*arrayCoefs;
            arrayCoefs = rightShiftLowerCoefs_(obj,arrayCoefs);
            arrayCoefs = B1*arrayCoefs;
%             arrayCoefs = blockButterflyTypeI_(obj,arrayCoefs);
%             arrayCoefs = arrayCoefs/2.0;
            % Lower channel rotation
            if isPeriodicExt
                arrayCoefs(1:hLen,:) = Wx1*arrayCoefs(1:hLen,:);
                arrayCoefs(hLen+1:end,:) = Ux1*arrayCoefs(hLen+1:end,:);
            else
                arrayCoefs(1:hLen,:) = Wx1*arrayCoefs(1:hLen,:);
                for iLay = 1:nLays_
                    if iLay == 1
                        U = -I;
                    else
                        U = Ux1;
                    end
                    arrayCoefs = lowerBlockRot_(obj,arrayCoefs,iLay,U);
                end
            end
            
            % Phase 2
            Wx2 = paramMtx4;
            Ux2 = paramMtx5;
            B2 = saivdr.dictionary.cnsoltx.mexsrcs.AbstCplxBuildingBlock.butterflyMtx(hLen,paramMtx6);
%             arrayCoefs = blockButterflyTypeI_(obj,arrayCoefs);
            arrayCoefs = B2'*arrayCoefs;
            arrayCoefs = leftShiftUpperCoefs_(obj,arrayCoefs);
            arrayCoefs = B2*arrayCoefs;
%             arrayCoefs = blockButterflyTypeI_(obj,arrayCoefs);
%             arrayCoefs = arrayCoefs/2.0;
            % Lower channel rotation
            arrayCoefs(hLen+1:end,:) = Ux2*arrayCoefs(hLen+1:end,:);
            arrayCoefs(1:hLen,:) = Wx2*arrayCoefs(1:hLen,:);
        end
        
        function arrayCoefs = supportExtTypeII_(obj,arrayCoefs,paramMtx1,paramMtx2,paramMtx3,paramMtx4,paramMtx5,paramMtx6,isPeriodicExt)
            hLen = obj.NumberOfHalfChannels;
            nLays_ = obj.nLays;        
            
            % Phase 1
            Wx = paramMtx1;
            Ux = paramMtx2;
            B = saivdr.dictionary.cnsoltx.mexsrcs.AbstCplxBuildingBlock.butterflyMtx(hLen,paramMtx3);
            I = eye(size(Ux));
%             arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs);
            arrayCoefs(1:end-1,:) = B'*arrayCoefs(1:end-1,:);
            arrayCoefs(1:end-1,:) = rightShiftLowerCoefs_(obj,arrayCoefs(1:end-1,:));
            arrayCoefs(1:end-1,:) = B*arrayCoefs(1:end-1,:);
%             arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs);
%             arrayCoefs = arrayCoefs/2.0;
            % Lower channel rotation
            if isPeriodicExt
                arrayCoefs(1:hLen,:) = Wx*arrayCoefs(1:hLen,:);
                arrayCoefs(hLen+1:end-1,:) = Ux*arrayCoefs(hLen+1:end-1,:);
            else
                arrayCoefs(1:hLen,:) = Wx*arrayCoefs(1:hLen,:);
                for iLay = 1:nLays_
                    if iLay == 1
                        U = -I;
                    else
                        U = Ux;
                    end
                    arrayCoefs = lowerBlockRot_(obj,arrayCoefs(1:end-1,:),iLay,U);
                end
            end
            
            % Phase 2
            Wx = paramMtx4;
            Ux = paramMtx5;
            B = saivdr.dictionary.cnsoltx.mexsrcs.AbstCplxBuildingBlock.butterflyMtx(hLen,paramMtx6);
%             arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs);
            arrayCoefs(1:end-1,:) = B'*arrayCoefs(1:end-1,:);
            arrayCoefs(1:end-1,:) = leftShiftUpperCoefs_(obj,arrayCoefs(1:end-1,:));
            arrayCoefs(1:end-1,:) = B*arrayCoefs(1:end-1,:);
%             arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs);
%             arrayCoefs = arrayCoefs/2.0;
            % Upper channel rotation
            arrayCoefs(hLen+1:end,:) = Ux*arrayCoefs(hLen+1:end,:);
            arrayCoefs(1:hLen+1,:) = Wx*arrayCoefs(1:hLen+1,:);
        end     
        
        function arrayCoefs = rightShiftLowerCoefs_(obj,arrayCoefs)
            hLenMn = obj.NumberOfHalfChannels;
            nRowsxnCols_ = obj.nRows*obj.nCols;
            %
            lowerCoefsPre = arrayCoefs(hLenMn+1:end,end-nRowsxnCols_+1:end);
            arrayCoefs(hLenMn+1:end,nRowsxnCols_+1:end) = ...
                arrayCoefs(hLenMn+1:end,1:end-nRowsxnCols_);
            arrayCoefs(hLenMn+1:end,1:nRowsxnCols_) = ...
                lowerCoefsPre;
        end

        function arrayCoefs = leftShiftUpperCoefs_(obj,arrayCoefs)
            hLenMn = obj.NumberOfHalfChannels;
            nRowsxnCols_ = obj.nRows*obj.nCols;
            %
            upperCoefsPost = arrayCoefs(1:hLenMn,1:nRowsxnCols_);
            arrayCoefs(1:hLenMn,1:end-nRowsxnCols_) = ...
                arrayCoefs(1:hLenMn,nRowsxnCols_+1:end);
            arrayCoefs(1:hLenMn,end-nRowsxnCols_+1:end) = ...
                upperCoefsPost;    
        end
    end
    
end
