classdef NsoltAtomExtender2d <  ...
        saivdr.dictionary.nsoltx.AbstNsoltCoefManipulator2d %#codegen
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
            
            %hLenU = obj.NumberOfSymmetricChannels;            
            %
            if ~isempty(obj.paramMtxCoefs)
%                 W0 = getParamMtx_(obj,uint32(1));
%                 U0 = getParamMtx_(obj,uint32(2));
%                 arrayCoefs(1:hLenU,:) ...
%                      = W0*arrayCoefs(1:hLenU,:);
%                 arrayCoefs(hLenU+1:end,:) = ...
%                     U0*arrayCoefs(hLenU+1:end,:);
                V0 = getParamMtx_(obj,uint32(1));
                arrayCoefs = V0*arrayCoefs;
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
                paramMtxW1 = getParamMtx_(obj,6*iOrd-4);
                paramMtxU1 = getParamMtx_(obj,6*iOrd-3);
                paramAngB1 = getParamMtx_(obj,6*iOrd-2);
                paramMtxW2 = getParamMtx_(obj,6*iOrd-1);
                paramMtxU2 = getParamMtx_(obj,6*iOrd+0);
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
                    paramMtxU2 = getParamMtx_(obj,6*iOrd+3*ordX+0);
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
            isPeriodicExt = obj.IsPeriodicExt; % BoundaryOperation = 'Circular'
            isPsGtPa      = obj.IsPsGreaterThanPa;            
            %
            ordY = obj.PolyPhaseOrder(saivdr.dictionary.utility.Direction.VERTICAL);
            ordX = obj.PolyPhaseOrder(saivdr.dictionary.utility.Direction.HORIZONTAL);
            
            % Width extension
            for iOrd = uint32(1):uint32(ordX/2)
                paramMtx1 = getParamMtx_(obj,6*iOrd-4); % WE
                paramMtx2 = getParamMtx_(obj,6*iOrd-3); % UE
                paramMtx3 = getParamMtx_(obj,6*iOrd-2); % angB1
                paramMtx4 = getParamMtx_(obj,6*iOrd-1); % WO
                paramMtx5 = getParamMtx_(obj,6*iOrd+0); % UO
                paramMtx6 = getParamMtx_(obj,6*iOrd+1); % angB2
                %
%                 if isPsGtPa
                    arrayCoefs = supportExtTypeIIPsGtPa_(obj,arrayCoefs,...
                        paramMtx1,paramMtx2,paramMtx3,...
                        paramMtx4,paramMtx5,paramMtx6,...
                        isPeriodicExt);
%                 else
%                     arrayCoefs = supportExtTypeIIPsLtPa_(obj,arrayCoefs,paramMtx1,paramMtx2,isPeriodicExt);
%                 end
            end
            
            % Height extension
            if ordY > 0
                arrayCoefs = permuteCoefs_(obj,arrayCoefs);
                for iOrd = uint32(1):uint32(ordY/2)  % Vertical process
                paramMtx1 = getParamMtx_(obj,6*iOrd+3*ordX-4); % WE
                paramMtx2 = getParamMtx_(obj,6*iOrd+3*ordX-3); % UE
                paramMtx3 = getParamMtx_(obj,6*iOrd+3*ordX-2); % angB1
                paramMtx4 = getParamMtx_(obj,6*iOrd+3*ordX-1); % WO
                paramMtx5 = getParamMtx_(obj,6*iOrd+3*ordX+0); % UO
                paramMtx6 = getParamMtx_(obj,6*iOrd+3*ordX+1); % angB2
                    %
%                     if isPsGtPa
                        arrayCoefs = supportExtTypeIIPsGtPa_(obj,arrayCoefs,...
                            paramMtx1,paramMtx2,paramMtx3,...
                            paramMtx4,paramMtx5,paramMtx6,...
                            isPeriodicExt);
%                     else
%                         arrayCoefs = supportExtTypeIIPsLtPa_(obj,arrayCoefs,paramMtx1,paramMtx2,isPeriodicExt);
%                     end
                end
                arrayCoefs = ipermuteCoefs_(obj,arrayCoefs);
            end
        end
        
        function arrayCoefs = supportExtTypeI_(obj,arrayCoefs,paramMtx1,paramMtx2,paramMtx3,paramMtx4,paramMtx5,paramMtx6,isPeriodicExt)
            hLen = obj.NumberOfSymmetricChannels;
            nCols_ = obj.nCols;
            
            % Phase 1
            Wx1 = paramMtx1;
            Ux1 = paramMtx2;
            B1 = butterflyMtx_(obj,paramMtx3);
            cB1 = conj(B1);
            I = eye(size(Ux1));
%             arrayCoefs = blockButterflyTypeI_(obj,arrayCoefs);
            arrayCoefs = cB1'*arrayCoefs;
            arrayCoefs = leftShiftLowerCoefs_(obj,arrayCoefs);
            arrayCoefs = cB1*arrayCoefs;
%             arrayCoefs = blockButterflyTypeI_(obj,arrayCoefs);
%             arrayCoefs = arrayCoefs/2.0;
            % Lower channel rotation
            if isPeriodicExt
                arrayCoefs(1:hLen,:) = Wx1*arrayCoefs(1:hLen,:);
                arrayCoefs(hLen+1:end,:) = Ux1*arrayCoefs(hLen+1:end,:);
            else
                arrayCoefs(1:hLen,:) = Wx1*arrayCoefs(1:hLen,:);
                for iCol = 1:nCols_
                    if iCol == 1 %&& ~isPeriodicExt
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
            cB2 = conj(B2);
            
%             arrayCoefs = blockButterflyTypeI_(obj,arrayCoefs);
            arrayCoefs = cB2'*arrayCoefs;
            arrayCoefs = rightShiftUpperCoefs_(obj,arrayCoefs);
            arrayCoefs = cB2*arrayCoefs;
%             arrayCoefs = blockButterflyTypeI_(obj,arrayCoefs);
%             arrayCoefs = arrayCoefs/2.0;
            % Lower channel rotation
            arrayCoefs(hLen+1:end,:) = Ux2*arrayCoefs(hLen+1:end,:);
            arrayCoefs(1:hLen,:) = Wx2*arrayCoefs(1:hLen,:);
        end
        
        function arrayCoefs = supportExtTypeIIPsGtPa_(obj,arrayCoefs,paramMtx1,paramMtx2,paramMtx3,paramMtx4,paramMtx5,paramMtx6,isPeriodicExt)
            hLen = obj.NumberOfAntisymmetricChannels;
            nCols_ = obj.nCols;
            
            % Phase 1
            Wx = paramMtx1;
            Ux = paramMtx2;
            B = butterflyMtx_(obj,paramMtx3);
            cB = conj(B);
            I = eye(size(Ux));
%             arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs);
            arrayCoefs(1:end-1,:) = cB'*arrayCoefs(1:end-1,:);
            arrayCoefs(1:end-1,:) = leftShiftLowerCoefs_(obj,arrayCoefs(1:end-1,:));
            arrayCoefs(1:end-1,:) = cB*arrayCoefs(1:end-1,:);
%             arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs);
%             arrayCoefs = arrayCoefs/2.0;
            % Lower channel rotation
            if isPeriodicExt
                arrayCoefs(1:hLen,:) = Wx*arrayCoefs(1:hLen,:);
                arrayCoefs(hLen+1:end-1,:) = Ux*arrayCoefs(hLen+1:end-1,:);
            else
                arrayCoefs(1:hLen,:) = Wx*arrayCoefs(1:hLen,:);
                for iCol = 1:nCols_
                    if iCol == 1
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
            cB = conj(B);
%             arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs);
            arrayCoefs(1:end-1,:) = cB'*arrayCoefs(1:end-1,:);
            arrayCoefs(1:end-1,:) = rightShiftUpperCoefs_(obj,arrayCoefs(1:end-1,:));
            arrayCoefs(1:end-1,:) = cB*arrayCoefs(1:end-1,:);
%             arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs);
%             arrayCoefs = arrayCoefs/2.0;
            % Upper channel rotation
            arrayCoefs(hLen+1:end,:) = Ux*arrayCoefs(hLen+1:end,:);
            arrayCoefs(1:hLen+1,:) = Wx*arrayCoefs(1:hLen+1,:);
        end
        
%         function arrayCoefs = supportExtTypeIIPsLtPa_(obj,arrayCoefs,paramMtx1,paramMtx2,isPeriodicExt)
%             hLen = obj.NumberOfSymmetricChannels;
%             nCols_ = obj.nCols;
%             
%             % Phase 1
%             Wx = paramMtx1;
%             I = eye(size(Wx));
%             arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs);
%             arrayCoefs = rightShiftLowerCoefs_(obj,arrayCoefs);
%             arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs);
%             arrayCoefs = arrayCoefs/2.0;
%             % Upper channel rotation
%             if isPeriodicExt
%                 arrayCoefs(1:hLen,:) = Wx*arrayCoefs(1:hLen,:);
%             else
%                 for iCol = 1:nCols_
%                     if iCol == 1
%                         W = -I;
%                     else
%                         W = Wx;
%                     end
%                     arrayCoefs = upperBlockRot_(obj,arrayCoefs,iCol,W);
%                 end
%             end
%             
%             % Phase 2
%             Ux = paramMtx2;
%             arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs);
%             arrayCoefs = leftShiftUpperCoefs_(obj,arrayCoefs);
%             arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs);
%             arrayCoefs = arrayCoefs/2.0;
%             % Lower channel rotation
%             arrayCoefs(hLen+1:end,:) = Ux*arrayCoefs(hLen+1:end,:);
%         end
        
%         function arrayCoefs = leftShiftLowerCoefs_(obj,arrayCoefs)
%             hLenMn = min([ obj.NumberOfSymmetricChannels
%                 obj.NumberOfAntisymmetricChannels]);
%             nRows_ = obj.nRows;
%             %
%             lowerCoefsPre = arrayCoefs(hLenMn+1:end,end-nRows_+1:end);
%             arrayCoefs(hLenMn+1:end,nRows_+1:end) = ...
%                 arrayCoefs(hLenMn+1:end,1:end-nRows_);
%             arrayCoefs(hLenMn+1:end,1:nRows_) = ...
%                 lowerCoefsPre;
%         end
%         
%         function arrayCoefs = leftShiftUpperCoefs_(obj,arrayCoefs)
%             hLenMx = min([ obj.NumberOfSymmetricChannels
%                 obj.NumberOfAntisymmetricChannels]);
%             nRows_ = obj.nRows;
%             %
%             upperCoefsPost = arrayCoefs(1:hLenMx,1:nRows_);
%             arrayCoefs(1:hLenMx,1:end-nRows_) = ...
%                 arrayCoefs(1:hLenMx,nRows_+1:end);
%             arrayCoefs(1:hLenMx,end-nRows_+1:end) = ...
%                 upperCoefsPost;            
%         end       
        function arrayCoefs = leftShiftLowerCoefs_(obj,arrayCoefs)
            hLenMn = min([ obj.NumberOfSymmetricChannels
                obj.NumberOfAntisymmetricChannels]);
            %TODO: nRows_の値の設定方法を確認する
            nRows_ = obj.nRows;
            %
            lowerCoefsPost = arrayCoefs(hLenMn+1:end,1:nRows_);
            arrayCoefs(hLenMn+1:end,1:end-nRows_) = ...
                arrayCoefs(hLenMn+1:end,nRows_+1:end);
            arrayCoefs(hLenMn+1:end,end-nRows_+1:end) = ...
                lowerCoefsPost;            
        end
        
        function arrayCoefs = rightShiftUpperCoefs_(obj,arrayCoefs)
            hLenMn = min([ obj.NumberOfSymmetricChannels
                obj.NumberOfAntisymmetricChannels]);
            nRows_ = obj.nRows;
            %
            upperCoefsPre = arrayCoefs(1:hLenMn,end-nRows_+1:end);
            arrayCoefs(1:hLenMn,nRows_+1:end) = ...
                arrayCoefs(1:hLenMn,1:end-nRows_);
            arrayCoefs(1:hLenMn,1:nRows_) = ...
                upperCoefsPre;
        end
        
        function hB = butterflyMtx_(obj, angles)%TODO: 同一の関数がAbstBuildingBlock.mで実装されているので一箇所にまとめる．
            hchs = obj.NumberOfAntisymmetricChannels;
            
            hC = complex(eye(hchs));
            hS = complex(eye(hchs));
            for p = 1:floor(hchs/2)
                tp = angles(p);
                
                hC(2*p-1:2*p, 2*p-1:2*p) = [ -1i*cos(tp), -1i*sin(tp);
                    cos(tp) , -sin(tp)]; %c^
                hS(2*p-1:2*p, 2*p-1:2*p) = [ -1i*sin(tp), -1i*cos(tp);
                    sin(tp) , -cos(tp)]; %s^
            end
            
            hB = [hC, conj(hC); 1i*hS, -1i*conj(hS)]/sqrt(2);
        end
    end
    
end
