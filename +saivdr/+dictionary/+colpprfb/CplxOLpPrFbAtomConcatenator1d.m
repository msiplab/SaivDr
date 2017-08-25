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
                arrayCoefs = V0(1:obj.NumberOfChannels,:)*arrayCoefs;
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
            hLen = obj.NumberOfHalfChannels;
            bmg = saivdr.dictionary.utility.ButterflyMatrixGenerationSystem('NumberOfSubmatrices',floor(obj.NumberOfHalfChannels/2));
            
            % Phase 1
            Wx2 = paramMtx1.';
            Ux2 = paramMtx2.';
            [ Cs, Ss ] = step(bmg,paramMtx3);
            
            % Lower channel rotation
            arrayCoefs(1:hLen,:) = Wx2*arrayCoefs(1:hLen,:);
            arrayCoefs(hLen+1:end,:) = Ux2*arrayCoefs(hLen+1:end,:);
            arrayCoefs = blockButterflyPre_(obj,arrayCoefs,Cs,Ss);
            arrayCoefs = rightShiftUpperCoefs_(obj,arrayCoefs);
            arrayCoefs = blockButterflyPost_(obj,arrayCoefs,Cs,Ss);
            arrayCoefs = arrayCoefs/2.0;
            
            % Phase 2
            Wx1 = paramMtx4.';
            Ux1 = paramMtx5.';
            [ Cs, Ss ] = step(bmg,paramMtx6);
            
            % Lower channel rotation
            if isPeriodicExt
                 arrayCoefs(1:hLen,:) = Wx1*arrayCoefs(1:hLen,:);
                 arrayCoefs(hLen+1:end,:) = Ux1*arrayCoefs(hLen+1:end,:);
            else
                % TODO:�����g���̒�`
                arrayCoefs(1:hLen,:) = Wx1*arrayCoefs(1:hLen,:);
                arrayCoefs(hLen+1:end,1) = Ux1*arrayCoefs(hLen+1:end,1);
                arrayCoefs(hLen+1:end,2:end) = Ux1*arrayCoefs(hLen+1:end,2:end);
            end
            arrayCoefs = blockButterflyPre_(obj,arrayCoefs,Cs,Ss);
            arrayCoefs = leftShiftLowerCoefs_(obj,arrayCoefs);
            arrayCoefs = blockButterflyPost_(obj,arrayCoefs,Cs,Ss);
            arrayCoefs = arrayCoefs/2.0;
        end
        
        function arrayCoefs = atomCncTypeII_(obj,arrayCoefs,paramMtx1,paramMtx2,paramMtx3,paramMtx4,paramMtx5,paramMtx6,isPeriodicExt)
            hLen = obj.NumberOfHalfChannels;
            bmg = saivdr.dictionary.utility.ButterflyMatrixGenerationSystem('NumberOfSubmatrices',floor(obj.NumberOfHalfChannels/2));
            
            % Phase 1
            Wx2 = paramMtx1.';
            Ux2 = paramMtx2.';
            [ Cs, Ss ] = step(bmg,paramMtx3);
            % Upper channel rotation
            arrayCoefs(1:hLen+1,:) = Wx2*arrayCoefs(1:hLen+1,:);
            arrayCoefs(hLen+1:end,:) = Ux2*arrayCoefs(hLen+1:end,:);
            
            arrayCoefs(1:end-1,:) = blockButterflyPre_(obj,arrayCoefs(1:end-1,:),Cs,Ss);
            arrayCoefs(1:end-1,:) = rightShiftUpperCoefs_(obj,arrayCoefs(1:end-1,:));
            arrayCoefs(1:end-1,:) = blockButterflyPost_(obj,arrayCoefs(1:end-1,:),Cs,Ss);
            arrayCoefs(1:end-1,:) = arrayCoefs(1:end-1,:)/2.0;
            
            % Phase 2
            Wx1 = paramMtx4.';
            Ux1 = paramMtx5.';
            [ Cs, Ss ] = step(bmg,paramMtx6);
            % Lower channel rotation
            if isPeriodicExt % TODO:�����g���̒�`
                arrayCoefs(1:hLen,:) = Wx1*arrayCoefs(1:hLen,:);
                arrayCoefs(hLen+1:end-1,:) = Ux1*arrayCoefs(hLen+1:end-1,:);
            else
                arrayCoefs(1:hLen,:) = Wx1*arrayCoefs(1:hLen,:);
                
                arrayCoefs(hLen+1:end-1,1) = Ux1*arrayCoefs(hLen+1:end-1,1);
                arrayCoefs(hLen+1:end-1,2:end) = Ux1*arrayCoefs(hLen+1:end-1,2:end);
            end
            arrayCoefs(1:end-1,:) = blockButterflyPre_(obj,arrayCoefs(1:end-1,:),Cs,Ss);
            arrayCoefs(1:end-1,:) = leftShiftLowerCoefs_(obj,arrayCoefs(1:end-1,:));
            arrayCoefs(1:end-1,:) = blockButterflyPost_(obj,arrayCoefs(1:end-1,:),Cs,Ss);
            arrayCoefs(1:end-1,:) = arrayCoefs(1:end-1,:)/2.0;
        end      
        
        function arrayCoefs = leftShiftLowerCoefs_(obj,arrayCoefs)
            hLen = obj.NumberOfHalfChannels;
            %
            lowerCoefsPre = arrayCoefs(hLen+1:end,1);
            arrayCoefs(hLen+1:end,1:end-1) = arrayCoefs(hLen+1:end,2:end);
            arrayCoefs(hLen+1:end,end) = lowerCoefsPre;            
        end
        
        function arrayCoefs = rightShiftUpperCoefs_(obj,arrayCoefs)
            hLen = obj.NumberOfHalfChannels;
            %
            upperCoefsPost = arrayCoefs(1:hLen,end);
            arrayCoefs(1:hLen,2:end) = arrayCoefs(1:hLen,1:end-1);
            arrayCoefs(1:hLen,1) = upperCoefsPost;
        end
    end
    
end