classdef CnsoltAtomExtender2d <  ...
        saivdr.dictionary.cnsoltx.AbstCnsoltCoefManipulator2d %#codegen
    %NSOLTATOMEXTENDER2D 2-D Atom Extender for NSOLT
    %
    % SVN identifier:
    % $Id: NsoltAtomExtender2d.m 683 2015-05-29 08:22:13Z sho $
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
        function obj = CnsoltAtomExtender2d(varargin)
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
        
        function arrayCoefs = stepImpl(obj, arrayCoefs, subScale, pmCoefs)
            stepImpl@saivdr.dictionary.cnsoltx.AbstCnsoltCoefManipulator2d(obj,arrayCoefs,subScale,pmCoefs);        
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
            
            if ~isempty(obj.paramMtxCoefs)
                V0 = getParamMtx_(obj,uint32(1));
                arrayCoefs = V0*arrayCoefs;
            end

        end
        
        function arrayCoefs = fullAtomExtTypeI_(obj,arrayCoefs)
            %
            %isPeriodicExt = obj.IsPeriodicExt; % BoundaryOperation = 'Circular'
            isPeriodicExt = true;
            %
            ordY = obj.PolyPhaseOrder(saivdr.dictionary.utility.Direction.VERTICAL);
            ordX = obj.PolyPhaseOrder(saivdr.dictionary.utility.Direction.HORIZONTAL);
            
            % Width extension
            for iOrd = uint32(1):uint32(ordX/2)
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
            
            % Height extension
            if ordY > 0
                arrayCoefs = permuteCoefs_(obj,arrayCoefs);
                for iOrd = uint32(1):uint32(ordY/2)
                    paramMtxW1 = getParamMtx_(obj,6*iOrd+3*ordX-4);
                    paramMtxU1 = getParamMtx_(obj,6*iOrd+3*ordX-3);
                    paramAngB1 = getParamMtx_(obj,6*iOrd+3*ordX-2);
                    paramMtxW2 = getParamMtx_(obj,6*iOrd+3*ordX-1);
                    paramMtxU2 = getParamMtx_(obj,6*iOrd+3*ordX  );
                    paramAngB2 = getParamMtx_(obj,6*iOrd+3*ordX+1);
                    %
                    arrayCoefs = supportExtTypeI_(obj,arrayCoefs,...
                        paramMtxW1,paramMtxU1,paramAngB1,...
                        paramMtxW2,paramMtxU2,paramAngB2,...
                        isPeriodicExt);
                end
                arrayCoefs = ipermuteCoefs_(obj,arrayCoefs);
            end
        end
        
        function arrayCoefs = fullAtomExtTypeII_(obj,arrayCoefs)
            %
            %isPeriodicExt = obj.IsPeriodicExt; % BoundaryOperation = 'Circular'
            isPeriodicExt = true;
            %
            ordY = obj.PolyPhaseOrder(saivdr.dictionary.utility.Direction.VERTICAL);
            ordX = obj.PolyPhaseOrder(saivdr.dictionary.utility.Direction.HORIZONTAL);
            
            % Width extension
            for iOrd = uint32(1):uint32(ordX/2)
                paramMtx1 = getParamMtx_(obj,6*iOrd-4); % WE
                paramMtx2 = getParamMtx_(obj,6*iOrd-3); % UE
                paramMtx3 = getParamMtx_(obj,6*iOrd-2); % angB1
                paramMtx4 = getParamMtx_(obj,6*iOrd-1); % WO
                paramMtx5 = getParamMtx_(obj,6*iOrd  ); % UO
                paramMtx6 = getParamMtx_(obj,6*iOrd+1); % angB2
                
                arrayCoefs = supportExtTypeII_(obj,arrayCoefs,...
                    paramMtx1,paramMtx2,paramMtx3,...
                    paramMtx4,paramMtx5,paramMtx6,...
                    isPeriodicExt);
            end
            
            % Height extension
            if ordY > 0
                arrayCoefs = permuteCoefs_(obj,arrayCoefs);
                for iOrd = uint32(1):uint32(ordY/2)  % Vertical process
                paramMtx1 = getParamMtx_(obj,6*iOrd+3*ordX-4); % WE
                paramMtx2 = getParamMtx_(obj,6*iOrd+3*ordX-3); % UE
                paramMtx3 = getParamMtx_(obj,6*iOrd+3*ordX-2); % angB1
                paramMtx4 = getParamMtx_(obj,6*iOrd+3*ordX-1); % WO
                paramMtx5 = getParamMtx_(obj,6*iOrd+3*ordX  ); % UO
                paramMtx6 = getParamMtx_(obj,6*iOrd+3*ordX+1); % angB2
                
                arrayCoefs = supportExtTypeII_(obj,arrayCoefs,...
                    paramMtx1,paramMtx2,paramMtx3,...
                    paramMtx4,paramMtx5,paramMtx6,...
                    isPeriodicExt);
                end
                arrayCoefs = ipermuteCoefs_(obj,arrayCoefs);
            end
        end
        
        function arrayCoefs = supportExtTypeI_(obj,arrayCoefs,paramMtx1,paramMtx2,paramMtx3,paramMtx4,paramMtx5,paramMtx6,isPeriodicExt)
            hLen = obj.NumberOfAntisymmetricChannels;
            nCols_ = obj.nCols;
            
            % Phase 1
            Wx1 = paramMtx1;
            Ux1 = paramMtx2;
            B1 = butterflyMtx_(obj,paramMtx3);
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
                for iCol = 1:nCols_
                    if iCol == nCols_ %&& ~isPeriodicExt
                        U = -I;
                    else
                        U = Ux1;
                    end
                    arrayCoefs = lowerBlockRot_(obj,arrayCoefs,iCol,U);
                end
                %arrayCoefs(hLen+1:end,:) = Ux1*arrayCoefs(hLen+1:end,:);
            end
            
            % Phase 2
            Wx2 = paramMtx4;
            Ux2 = paramMtx5;
            B2 = butterflyMtx_(obj,paramMtx6);
            
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
            hLen = obj.NumberOfAntisymmetricChannels;
            nCols_ = obj.nCols;
            
            % Phase 1
            Wx = paramMtx1;
            Ux = paramMtx2;
            B = butterflyMtx_(obj,paramMtx3);
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
                for iCol = 1:nCols_
                    if iCol == nCols_
                        U = -I;
                    else
                        U = Ux;
                    end
                    arrayCoefs(1:end-1,:) = lowerBlockRot_(obj,arrayCoefs(1:end-1,:),iCol,U);
                end
            end
            
            % Phase 2
            Wx = paramMtx4;
            Ux = paramMtx5;
            B = butterflyMtx_(obj,paramMtx6);
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
            hLenMn = min([ obj.NumberOfSymmetricChannels
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
            hLenMn = min([ obj.NumberOfSymmetricChannels
                obj.NumberOfAntisymmetricChannels]);
            nRows_ = obj.nRows;
            %
            upperCoefsPost = arrayCoefs(1:hLenMn,1:nRows_);
            arrayCoefs(1:hLenMn,1:end-nRows_) = ...
                arrayCoefs(1:hLenMn,nRows_+1:end);
            arrayCoefs(1:hLenMn,end-nRows_+1:end) = ...
                upperCoefsPost;
        end
        
        function hB = butterflyMtx_(obj, angles)%TODO: ????????????AbstBuildingBlock.m?????????????????????????????????????D
            hchs = obj.NumberOfAntisymmetricChannels;
            
            hC = complex(eye(hchs));
            hS = complex(eye(hchs));
            for p = 1:floor(hchs/2)
                tp = angles(p)/2;
                
                hC(2*p-1:2*p, 2*p-1:2*p) = [ -1i*cos(tp), -1i*sin(tp);
                    cos(tp) , -sin(tp)]; %c^
                hS(2*p-1:2*p, 2*p-1:2*p) = [ -1i*sin(tp), -1i*cos(tp);
                    sin(tp) , -cos(tp)]; %s^
            end
            
            hB = [hC, conj(hC); 1i*hS, -1i*conj(hS)]/sqrt(2);
        end
    end
    
end
