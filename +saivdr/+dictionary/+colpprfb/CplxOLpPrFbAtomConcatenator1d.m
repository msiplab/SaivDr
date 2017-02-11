classdef CplxOLpPrFbAtomConcatenator1d < ...
        saivdr.dictionary.colpprfb.AbstCplxOLpPrFbCoefManipulator1d %#codegen
    %OLPPRFBATOMCONCATENATOR1D Atom concatenator for 1-D OLPPRFB 
    %
    % SVN identifier:
    % 
    % $Id: OLpPrFbAtomConcatenator1d.m 657 2015-03-17 00:45:15Z sho $
    %
    % Requirements: MATLAB R2013b
    %
    % Copyright (c) 2015, Shogo MURAMATSU
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
        function obj = CplxOLpPrFbAtomConcatenator1d(varargin)
            obj = obj@saivdr.dictionary.colpprfb.AbstCplxOLpPrFbCoefManipulator1d(varargin{:});
        end
        
    end
    
    methods (Access = protected)
        
        function s = saveObjectImpl(obj)
            % Call the base class method
            s = saveObjectImpl@saivdr.dictionary.colpprfb.AbstCplxOLpPrFbCoefManipulator1d(obj);
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            % Call base class method to load public properties
            loadObjectImpl@saivdr.dictionary.colpprfb.AbstCplxOLpPrFbCoefManipulator1d(obj,s,wasLocked);
        end

        function arrayCoefs = stepImpl(obj,arrayCoefs,subScale,pmCoefs)
            stepImpl@saivdr.dictionary.colpprfb.AbstCplxOLpPrFbCoefManipulator1d(obj,arrayCoefs,subScale,pmCoefs);
            %
            if strcmp(obj.OLpPrFbType,'Type I')
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
                arrayCoefs = V0*arrayCoefs;
            end
            
        end
        
        function arrayCoefs = fullAtomCncTypeI_(obj,arrayCoefs)
            %
            isPeriodicExt = obj.IsPeriodicExt; % BoundaryOperation = 'Circular'
            %
            ord = obj.PolyPhaseOrder;
            numOfPMtx = size(obj.paramMtxSzTab,1);
            %
            hOrd = uint32(ord/2);
            if hOrd > 0
                for iOrd = uint32(1):hOrd  
                    paramMtx1 = getParamMtx_(obj,numOfPMtx-6*iOrd+4); % W2
                    paramMtx2 = getParamMtx_(obj,numOfPMtx-6*iOrd+5); % U2
                    paramMtx3 = getParamMtx_(obj,numOfPMtx-6*iOrd+6); % angB2
                    paramMtx4 = getParamMtx_(obj,numOfPMtx-6*iOrd+1); % W1
                    paramMtx5 = getParamMtx_(obj,numOfPMtx-6*iOrd+2); % U1
                    paramMtx6 = getParamMtx_(obj,numOfPMtx-6*iOrd+3); % angB1
                    %
                    arrayCoefs = atomCncTypeI_(obj,arrayCoefs,paramMtx1,paramMtx2,paramMtx3,paramMtx4,paramMtx5,paramMtx6,isPeriodicExt);
                end
            end
            
        end
        
        function arrayCoefs = fullAtomCncTypeII_(obj,arrayCoefs)
            %
            isPeriodicExt = obj.IsPeriodicExt; % BoundaryOperation = 'Circular';
            %
            ord = obj.PolyPhaseOrder;
            numOfPMtx = size(obj.paramMtxSzTab,1);
            %
            hOrd = uint32(ord/2);
            if hOrd > 0
                for iOrd = uint32(1):hOrd 
                    paramMtx1 = getParamMtx_(obj,numOfPMtx-6*iOrd+4); % W2
                    paramMtx2 = getParamMtx_(obj,numOfPMtx-6*iOrd+5); % U2
                    paramMtx3 = getParamMtx_(obj,numOfPMtx-6*iOrd+6); % angB2
                    paramMtx4 = getParamMtx_(obj,numOfPMtx-6*iOrd+1); % W1
                    paramMtx5 = getParamMtx_(obj,numOfPMtx-6*iOrd+2); % U1
                    paramMtx6 = getParamMtx_(obj,numOfPMtx-6*iOrd+3); % angB1
                    %
                    arrayCoefs = atomCncTypeII_(...
                        obj,arrayCoefs,paramMtx1,paramMtx2,paramMtx3,...
                        paramMtx4,paramMtx5,paramMtx6,isPeriodicExt);
                end
            end
            
        end
       
        function arrayCoefs = atomCncTypeI_(obj,arrayCoefs,paramMtx1,paramMtx2,paramMtx3,paramMtx4,paramMtx5,paramMtx6,isPeriodicExt)
            hLen = obj.NumberOfSymmetricChannels;
            
            % Phase 1
            Wx2 = paramMtx1.';
            Ux2 = paramMtx2.';
            Bx2 = butterflyMtx_(obj,paramMtx3);
            
            % Lower channel rotation
            arrayCoefs(1:hLen,:) = Wx2*arrayCoefs(1:hLen,:);
            arrayCoefs(hLen+1:end,:) = Ux2*arrayCoefs(hLen+1:end,:);
            %arrayCoefs = blockButterflyTypeI_(obj,arrayCoefs,[]);
            arrayCoefs = Bx2'*arrayCoefs;
            arrayCoefs = rightShiftUpperCoefs_(obj,arrayCoefs);
            arrayCoefs = Bx2*arrayCoefs;
            %arrayCoefs = blockButterflyTypeI_(obj,arrayCoefs,[]);
            %arrayCoefs = arrayCoefs/2.0;
            
            % Phase 2
            Wx1 = paramMtx4.';
            Ux1 = paramMtx5.';
            B1 = butterflyMtx_(obj, paramMtx6);
            % Lower channel rotation
            if isPeriodicExt
                 arrayCoefs(1:hLen,:) = Wx1*arrayCoefs(1:hLen,:);
                 arrayCoefs(hLen+1:end,:) = Ux1*arrayCoefs(hLen+1:end,:);
            else
                % TODO:é¸ä˙ägí£ÇÃíËã`
                arrayCoefs(1:hLen,:) = Wx1*arrayCoefs(1:hLen,:);
                arrayCoefs(hLen+1:end,1) = Ux1*arrayCoefs(hLen+1:end,1);
                arrayCoefs(hLen+1:end,2:end) = Ux1*arrayCoefs(hLen+1:end,2:end);
            end
            %arrayCoefs = blockButterflyTypeI_(obj,arrayCoefs,[]);
            arrayCoefs = B1'*arrayCoefs;
            arrayCoefs = leftShiftLowerCoefs_(obj,arrayCoefs);
            arrayCoefs = B1*arrayCoefs;
            %arrayCoefs = blockButterflyTypeI_(obj,arrayCoefs,[]);
            %arrayCoefs = arrayCoefs/2.0;
        end
        
        function arrayCoefs = atomCncTypeII_(obj,arrayCoefs,paramMtx1,paramMtx2,paramMtx3,paramMtx4,paramMtx5,paramMtx6,isPeriodicExt)
            hLen = obj.NumberOfSymmetricChannels;
            
            % Phase 1
            Wx2 = paramMtx1.';
            Ux2 = paramMtx2.';
            B2 = butterflyMtx_(obj,paramMtx3);
            % Upper channel rotation
            arrayCoefs(1:hLen,:) = Wx2*arrayCoefs(1:hLen,:);
            arrayCoefs(hLen:end,:) = Ux2*arrayCoefs(hLen:end,:);
            
            %arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs,[]);
            arrayCoefs(1:end-1,:) = B2'*arrayCoefs(1:end-1,:);
            arrayCoefs(1:end-1,:) = rightShiftUpperCoefs_(obj,arrayCoefs(1:end-1,:));
            arrayCoefs(1:end-1,:) = B2*arrayCoefs(1:end-1,:);
%             arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs,[]);
%             arrayCoefs = arrayCoefs/2.0;
            
            % Phase 2
            Wx1 = paramMtx4.';
            Ux1 = paramMtx5.';
            B1 = butterflyMtx_(obj,paramMtx6);
            % Lower channel rotation
            if isPeriodicExt % TODO:é¸ä˙ägí£ÇÃíËã`
                arrayCoefs(1:hLen-1,:) = Wx1*arrayCoefs(1:hLen-1,:);
                arrayCoefs(hLen:end-1,:) = Ux1*arrayCoefs(hLen:end-1,:);
            else
                arrayCoefs(1:hLen-1,:) = Wx1*arrayCoefs(1:hLen-1,:);
                
                arrayCoefs(hLen:end-1,1) = Ux1*arrayCoefs(hLen:end-1,1);
                arrayCoefs(hLen:end-1,2:end) = Ux1*arrayCoefs(hLen:end-1,2:end);
            end
            %arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs,[]);
            arrayCoefs(1:end-1,:) = B1'*arrayCoefs(1:end-1,:);
            arrayCoefs(1:end-1,:) = leftShiftLowerCoefs_(obj,arrayCoefs(1:end-1,:));
            arrayCoefs(1:end-1,:) = B1*arrayCoefs(1:end-1,:);
            %arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs,[]);
            %arrayCoefs = arrayCoefs/2.0;
        end      
        
        function arrayCoefs = leftShiftLowerCoefs_(obj,arrayCoefs)
            hLenMn = min([ obj.NumberOfSymmetricChannels
                obj.NumberOfAntisymmetricChannels]);
            %
            lowerCoefsPre = arrayCoefs(hLenMn+1:end,1);
            arrayCoefs(hLenMn+1:end,1:end-1) = arrayCoefs(hLenMn+1:end,2:end);
            arrayCoefs(hLenMn+1:end,end) = lowerCoefsPre;            
        end
        
        function arrayCoefs = rightShiftUpperCoefs_(obj,arrayCoefs)
            hLenMn = min([ obj.NumberOfSymmetricChannels
                obj.NumberOfAntisymmetricChannels]);
            %
            upperCoefsPost = arrayCoefs(1:hLenMn,end);
            arrayCoefs(1:hLenMn,2:end) = arrayCoefs(1:hLenMn,1:end-1);
            arrayCoefs(1:hLenMn,1) = upperCoefsPost;
        end
        
        function hB = butterflyMtx_(obj, angles)%TODO: ìØàÍÇÃä÷êîÇ™AbstBuildingBlock.mÇ≈é¿ëïÇ≥ÇÍÇƒÇ¢ÇÈÇÃÇ≈àÍâ”èäÇ…Ç‹Ç∆ÇﬂÇÈÅD
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