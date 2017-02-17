classdef CnsoltAtomConcatenator2d < ...
        saivdr.dictionary.cnsoltx.AbstCnsoltCoefManipulator2d %#codegen
    %NSOLTSYNTHESIZER2D 2-D NSOLT Synthesizer
    %
    % SVN identifier:
    % $Id: NsoltAtomConcatenator2d.m 683 2015-05-29 08:22:13Z sho $
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
        function obj = CnsoltAtomConcatenator2d(varargin)
            obj = obj@saivdr.dictionary.cnsoltx.AbstCnsoltCoefManipulator2d(varargin{:});
        end
        
    end
    
    methods (Access = protected)
        
        function s = saveObjectImpl(obj)
            % Call the base class method
            s = saveObjectImpl@saivdr.dictionary.cnsoltx.AbstCnsoltCoefManipulator2d(obj);
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            % Call base class method to load public properties
            loadObjectImpl@saivdr.dictionary.cnsoltx.AbstCnsoltCoefManipulator2d(obj,s,wasLocked);
        end

        function arrayCoefs = stepImpl(obj,arrayCoefs,subScale,pmCoefs)
            stepImpl@saivdr.dictionary.cnsoltx.AbstCnsoltCoefManipulator2d(obj,arrayCoefs,subScale,pmCoefs);
            
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
            %
            if ~isempty(obj.paramMtxCoefs)
                V0 = getParamMtx_(obj,uint32(1)).';
                arrayCoefs = V0(1:obj.NumberOfChannels,:)*arrayCoefs;
            end
            
        end
        
        function arrayCoefs = fullAtomCncTypeI_(obj,arrayCoefs)
            %
            isPeriodicExt = obj.IsPeriodicExt; % BoundaryOperation = 'Circular'
            %TODO:
            isPeriodicExt = true;
            %
            ordY = obj.PolyPhaseOrder(saivdr.dictionary.utility.Direction.VERTICAL);
            ordX = obj.PolyPhaseOrder(saivdr.dictionary.utility.Direction.HORIZONTAL);
            numOfPMtx = size(obj.paramMtxSzTab,1);
            %
            hOrdY = uint32(ordY/2);
            if hOrdY > 0
                arrayCoefs = permuteCoefs_(obj,arrayCoefs);
                for iOrd = uint32(1):hOrdY  % Vertical process
                    paramMtx1 = getParamMtx_(obj,numOfPMtx-6*iOrd+4); % Wy2
                    paramMtx2 = getParamMtx_(obj,numOfPMtx-6*iOrd+5); % Uy2
                    paramMtx3 = getParamMtx_(obj,numOfPMtx-6*iOrd+6); % angBy2
                    paramMtx4 = getParamMtx_(obj,numOfPMtx-6*iOrd+1); % Wy1
                    paramMtx5 = getParamMtx_(obj,numOfPMtx-6*iOrd+2); % Uy1
                    paramMtx6 = getParamMtx_(obj,numOfPMtx-6*iOrd+3); % angBy1
                    %
                    arrayCoefs = atomCncTypeI_(obj,arrayCoefs,...
                        paramMtx1,paramMtx2,paramMtx3,...
                        paramMtx4,paramMtx5,paramMtx6,...
                        isPeriodicExt);
                end
                arrayCoefs = ipermuteCoefs_(obj,arrayCoefs);
            end
            %
            hOrdX = uint32(ordX/2);
            for iOrd = uint32(1):hOrdX % Horizontal process
                paramMtx1 = getParamMtx_(obj,numOfPMtx-6*(hOrdY+iOrd)+4); % Wx2
                paramMtx2 = getParamMtx_(obj,numOfPMtx-6*(hOrdY+iOrd)+5); % Ux2
                paramMtx3 = getParamMtx_(obj,numOfPMtx-6*(hOrdY+iOrd)+6); % angBx2
                paramMtx4 = getParamMtx_(obj,numOfPMtx-6*(hOrdY+iOrd)+1); % Wx1
                paramMtx5 = getParamMtx_(obj,numOfPMtx-6*(hOrdY+iOrd)+2); % Ux1
                paramMtx6 = getParamMtx_(obj,numOfPMtx-6*(hOrdY+iOrd)+3); % angBx1
                %
                arrayCoefs = atomCncTypeI_(obj,arrayCoefs,...
                    paramMtx1,paramMtx2,paramMtx3,...
                    paramMtx4,paramMtx5,paramMtx6,...
                    isPeriodicExt);
            end
            
        end
        
        function arrayCoefs = fullAtomCncTypeII_(obj,arrayCoefs)
            %
            isPeriodicExt = obj.IsPeriodicExt; % BoundaryOperation = 'Circular';
            isPeriodicExt = true;
            %
            ordY = obj.PolyPhaseOrder(saivdr.dictionary.utility.Direction.VERTICAL);
            ordX = obj.PolyPhaseOrder(saivdr.dictionary.utility.Direction.HORIZONTAL);
            numOfPMtx = size(obj.paramMtxSzTab,1);
            %
            hOrdY = uint32(ordY/2);
            if hOrdY > 0
                arrayCoefs = permuteCoefs_(obj,arrayCoefs);
                for iOrd = uint32(1):hOrdY % Vertical process
                    paramMtx1 = getParamMtx_(obj,numOfPMtx-6*iOrd+4); % Wy2
                    paramMtx2 = getParamMtx_(obj,numOfPMtx-6*iOrd+5); % Uy2
                    paramMtx3 = getParamMtx_(obj,numOfPMtx-6*iOrd+6); % angBy2
                    paramMtx4 = getParamMtx_(obj,numOfPMtx-6*iOrd+1); % Wy1
                    paramMtx5 = getParamMtx_(obj,numOfPMtx-6*iOrd+2); % Uy1
                    paramMtx6 = getParamMtx_(obj,numOfPMtx-6*iOrd+3); % angBy1
                    %
                    arrayCoefs = atomCncTypeII_(obj,arrayCoefs,...
                        paramMtx1,paramMtx2,paramMtx3,...
                        paramMtx4,paramMtx5,paramMtx6,...
                        isPeriodicExt);
                end
                arrayCoefs = ipermuteCoefs_(obj,arrayCoefs);
            end
            %
            hOrdX = uint32(ordX/2);
            for iOrd = uint32(1):hOrdX % Horizontal process
                paramMtx1 = getParamMtx_(obj,numOfPMtx-6*(hOrdY+iOrd)+4); % Wx2
                paramMtx2 = getParamMtx_(obj,numOfPMtx-6*(hOrdY+iOrd)+5); % Ux2
                paramMtx3 = getParamMtx_(obj,numOfPMtx-6*(hOrdY+iOrd)+6); % angBx2
                paramMtx4 = getParamMtx_(obj,numOfPMtx-6*(hOrdY+iOrd)+1); % Wx1
                paramMtx5 = getParamMtx_(obj,numOfPMtx-6*(hOrdY+iOrd)+2); % Ux1
                paramMtx6 = getParamMtx_(obj,numOfPMtx-6*(hOrdY+iOrd)+3); % angBx1
                %
                arrayCoefs = atomCncTypeII_(obj,arrayCoefs,...
                    paramMtx1,paramMtx2,paramMtx3,...
                    paramMtx4,paramMtx5,paramMtx6,...
                    isPeriodicExt);
            end
            
        end
       
        function arrayCoefs = atomCncTypeI_(obj,arrayCoefs,paramMtx1,paramMtx2,paramMtx3,paramMtx4,paramMtx5,paramMtx6,isPeriodicExt)
            %import saivdr.dictionary.cnsoltx.mexsrcs.AbstCplxBuildingBlock
            hLen = obj.NumberOfHalfChannels;
            
            % Phase 1
            Wx2 = paramMtx1.';
            Ux2 = paramMtx2.';
            Bx2 = saivdr.dictionary.cnsoltx.mexsrcs.AbstCplxBuildingBlock.butterflyMtx(hLen,paramMtx3);
            % Lower channel rotation
            arrayCoefs(1:hLen,:) = Wx2*arrayCoefs(1:hLen,:);
            arrayCoefs(hLen+1:end,:) = Ux2*arrayCoefs(hLen+1:end,:);
%             arrayCoefs = blockButterflyTypeI_(obj,arrayCoefs);
            arrayCoefs = Bx2'*arrayCoefs;
            arrayCoefs = rightShiftUpperCoefs_(obj,arrayCoefs);
            arrayCoefs = Bx2*arrayCoefs;
%             arrayCoefs = blockButterflyTypeI_(obj,arrayCoefs);
%             arrayCoefs = arrayCoefs/2.0;
            
            % Phase 2
            Wx1 = paramMtx4.';
            Ux1 = paramMtx5.';
            Bx1 = saivdr.dictionary.cnsoltx.mexsrcs.AbstCplxBuildingBlock.butterflyMtx(hLen,paramMtx6);
            I = eye(size(Ux1));
            % Lower channel rotation
            if isPeriodicExt
                arrayCoefs(1:hLen,:) = Wx1*arrayCoefs(1:hLen,:);
                arrayCoefs(hLen+1:end,:) = Ux1*arrayCoefs(hLen+1:end,:);
            else
                arrayCoefs(1:hLen,:) = Wx1*arrayCoefs(1:hLen,:);
                for iCol = 1:obj.nCols
                    if iCol == 1 
                        U = -I;
                    else
                        U = Ux1;
                    end
                    arrayCoefs = lowerBlockRot_(obj,arrayCoefs,iCol,U);
                end
            end
%             arrayCoefs = blockButterflyTypeI_(obj,arrayCoefs);
            arrayCoefs = Bx1'*arrayCoefs;
            arrayCoefs = leftShiftLowerCoefs_(obj,arrayCoefs);
            arrayCoefs = Bx1*arrayCoefs;
%             arrayCoefs = blockButterflyTypeI_(obj,arrayCoefs);
%             arrayCoefs = arrayCoefs/2.0;
        end
        
        function arrayCoefs = atomCncTypeII_(obj,arrayCoefs,paramMtx1,paramMtx2,paramMtx3,paramMtx4,paramMtx5,paramMtx6,isPeriodicExt)
            %import saivdr.dictionary.cnsoltx.mexsrcs.AbstCplxBuildingBlock
            hLen = obj.NumberOfHalfChannels;
            
            % Phase 1
            Wx2 = paramMtx1.';
            Ux2 = paramMtx2.';
            B2 = saivdr.dictionary.cnsoltx.mexsrcs.AbstCplxBuildingBlock.butterflyMtx(hLen,paramMtx3);
            % Upper channel rotation
            arrayCoefs(1:hLen+1,:) = Wx2*arrayCoefs(1:hLen+1,:);
            arrayCoefs(hLen+1:end,:) = Ux2*arrayCoefs(hLen+1:end,:);
%             arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs);
            arrayCoefs(1:end-1,:) = B2'*arrayCoefs(1:end-1,:);
            arrayCoefs(1:end-1,:) = rightShiftUpperCoefs_(obj,arrayCoefs(1:end-1,:));
            arrayCoefs(1:end-1,:) = B2*arrayCoefs(1:end-1,:);
%             arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs);
%             arrayCoefs = arrayCoefs/2.0;
            
            % Phase 2
            Wx1 = paramMtx4.';
            Ux1 = paramMtx5.';
            B1 = saivdr.dictionary.cnsoltx.mexsrcs.AbstCplxBuildingBlock.butterflyMtx(hLen,paramMtx6);
            I = eye(size(Ux1));
            % Lower channel rotation
            if isPeriodicExt
                arrayCoefs(1:hLen,:) = Wx1*arrayCoefs(1:hLen,:);
                arrayCoefs(hLen+1:end-1,:) = Ux1*arrayCoefs(hLen+1:end-1,:);
            else
                arrayCoefs(1:hLen,:) = Wx1*arrayCoefs(1:hLen,:);
                for iCol = 1:obj.nCols
                    if iCol == 1
                        U = -I;
                    else
                        U = Ux1;
                    end
                    arrayCoefs(1:end-1,:) = lowerBlockRot_(obj,arrayCoefs(1:end-1,:),iCol,U);
                end
            end
%             arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs);
            arrayCoefs(1:end-1,:) = B1'*arrayCoefs(1:end-1,:);
            arrayCoefs(1:end-1,:) = leftShiftLowerCoefs_(obj,arrayCoefs(1:end-1,:));
            arrayCoefs(1:end-1,:) = B1*arrayCoefs(1:end-1,:);
%             arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs);
%             arrayCoefs = arrayCoefs/2.0;
        end      
        
        function arrayCoefs = leftShiftLowerCoefs_(obj,arrayCoefs)
            hLenMn = obj.NumberOfHalfChannels;
            nRows_ = obj.nRows;
            %
            lowerCoefsPost = arrayCoefs(hLenMn+1:end,1:nRows_);
            arrayCoefs(hLenMn+1:end,1:end-nRows_) = ...
                arrayCoefs(hLenMn+1:end,nRows_+1:end);
            arrayCoefs(hLenMn+1:end,end-nRows_+1:end) = ...
                lowerCoefsPost;            
        end
        
        function arrayCoefs = rightShiftUpperCoefs_(obj,arrayCoefs)
            hLenMn = obj.NumberOfHalfChannels;
            nRows_ = obj.nRows;
            %
            upperCoefsPre = arrayCoefs(1:hLenMn,end-nRows_+1:end);
            arrayCoefs(1:hLenMn,nRows_+1:end) = ...
                arrayCoefs(1:hLenMn,1:end-nRows_);
            arrayCoefs(1:hLenMn,1:nRows_) = ...
                upperCoefsPre;
        end
        
    end
    
end
