classdef OLpPrFbAtomConcatenator1d < ...
        saivdr.dictionary.olpprfb.AbstOLpPrFbCoefManipulator1d %#codegen
    %OLPPRFBATOMCONCATENATOR1D Atom concatenator for 1-D OLPPRFB 
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2015-2017, Shogo MURAMATSU
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
        function obj = OLpPrFbAtomConcatenator1d(varargin)
            obj = obj@saivdr.dictionary.olpprfb.AbstOLpPrFbCoefManipulator1d(varargin{:});
        end
        
    end
    
    methods (Access = protected)
        
        function s = saveObjectImpl(obj)
            % Call the base class method
            s = saveObjectImpl@saivdr.dictionary.olpprfb.AbstOLpPrFbCoefManipulator1d(obj);
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            % Call base class method to load public properties
            loadObjectImpl@saivdr.dictionary.olpprfb.AbstOLpPrFbCoefManipulator1d(obj,s,wasLocked);
        end
        
        function arrayCoefs = stepImpl(obj,arrayCoefs,subScale,pmCoefs)
            stepImpl@saivdr.dictionary.olpprfb.AbstOLpPrFbCoefManipulator1d(obj,arrayCoefs,subScale,pmCoefs);
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
            ord = obj.PolyPhaseOrder;
            numOfPMtx = size(obj.paramMtxSzTab,1);
            %
            hOrd = uint32(ord/2);
            if hOrd > 0
                for iOrd = uint32(1):hOrd  
                    paramMtx1 = getParamMtx_(obj,numOfPMtx-2*iOrd+2);
                    paramMtx2 = getParamMtx_(obj,numOfPMtx-2*iOrd+1);
                    %
                    arrayCoefs = atomCncTypeI_(obj,arrayCoefs,paramMtx1,paramMtx2,isPeriodicExt);
                end
            end
            
        end
        
        function arrayCoefs = fullAtomCncTypeII_(obj,arrayCoefs)
            %
            isPeriodicExt = obj.IsPeriodicExt; % BoundaryOperation = 'Circular';
            isPsGtPa      = obj.IsPsGreaterThanPa;
            %
            ord = obj.PolyPhaseOrder;
            numOfPMtx = size(obj.paramMtxSzTab,1);
            %
            hOrd = uint32(ord/2);
            if hOrd > 0
                for iOrd = uint32(1):hOrd 
                    paramMtx1 = getParamMtx_(obj,numOfPMtx-2*iOrd+1); % W
                    paramMtx2 = getParamMtx_(obj,numOfPMtx-2*iOrd+2); % U
                    %
                    if isPsGtPa
                        arrayCoefs = atomCncTypeIIPsGtPa_(obj,arrayCoefs,paramMtx1,paramMtx2,isPeriodicExt);
                    else
                        arrayCoefs = atomCncTypeIIPsLtPa_(obj,arrayCoefs,paramMtx1,paramMtx2,isPeriodicExt);                        
                    end
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
            % Lower channel rotation
            if isPeriodicExt
                arrayCoefs(hLen+1:end,:) = Ux1*arrayCoefs(hLen+1:end,:);
            else
                arrayCoefs(hLen+1:end,1) = -arrayCoefs(hLen+1:end,1);                
                arrayCoefs(hLen+1:end,2:end) = Ux1*arrayCoefs(hLen+1:end,2:end);                
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
            % Lower channel rotation
            if isPeriodicExt
                 arrayCoefs(hLen+1:end,:) = Ux*arrayCoefs(hLen+1:end,:);
            else
                arrayCoefs(hLen+1:end,1) = -arrayCoefs(hLen+1:end,1);
                arrayCoefs(hLen+1:end,2:end) = Ux*arrayCoefs(hLen+1:end,2:end);
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
            % Upper channel rotation
            if isPeriodicExt
                 arrayCoefs(1:hLen,:) = Wx*arrayCoefs(1:hLen,:);
            else
                 arrayCoefs(1:hLen,1) = -arrayCoefs(1:hLen,1);
                 arrayCoefs(1:hLen,2:end) = Wx*arrayCoefs(1:hLen,2:end);
            end
            arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs);
            arrayCoefs = leftShiftLowerCoefs_(obj,arrayCoefs);
            arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs);
            arrayCoefs = arrayCoefs/2.0;
        end        
        
        function arrayCoefs = leftShiftLowerCoefs_(obj,arrayCoefs)
            hLenMn = min([ obj.NumberOfSymmetricChannels
                obj.NumberOfAntisymmetricChannels]);
            %
            lowerCoefsPost = arrayCoefs(hLenMn+1:end,1);
            arrayCoefs(hLenMn+1:end,1:end-1) = arrayCoefs(hLenMn+1:end,2:end);
            arrayCoefs(hLenMn+1:end,end) = lowerCoefsPost;            
        end
        
        function arrayCoefs = rightShiftUpperCoefs_(obj,arrayCoefs)
            hLenMx = max([ obj.NumberOfSymmetricChannels
                obj.NumberOfAntisymmetricChannels]);
            %
            upperCoefsPre = arrayCoefs(1:hLenMx,end);
            arrayCoefs(1:hLenMx,2:end) = arrayCoefs(1:hLenMx,1:end-1);
            arrayCoefs(1:hLenMx,1) = upperCoefsPre;
        end
    end
    
end
