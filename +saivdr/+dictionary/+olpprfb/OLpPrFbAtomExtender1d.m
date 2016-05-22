classdef OLpPrFbAtomExtender1d <  ...
        saivdr.dictionary.olpprfb.AbstOLpPrFbCoefManipulator1d %#codegen
    %OLPPRFBATOMEXTENDER1D 1-D Atom Extender for OLPPRFB
    %
    % SVN identifier:
    % $Id: OLpPrFbAtomExtender1d.m 657 2015-03-17 00:45:15Z sho $
    %
    % Requirements: MATLAB R2013b
    %
    % Copyright (c) 2014, Shogo MURAMATSU
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
        function obj = OLpPrFbAtomExtender1d(varargin)
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
        
        function arrayCoefs = stepImpl(obj, arrayCoefs, subScale, pmCoefs)
            stepImpl@saivdr.dictionary.olpprfb.AbstOLpPrFbCoefManipulator1d(obj,arrayCoefs,subScale,pmCoefs);        
            %
            arrayCoefs = initialStep_(obj,arrayCoefs);
            %
            if strcmp(obj.OLpPrFbType,'Type I')
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
                %V0 = getParamMtx_(obj,uint32(1));
                arrayCoefs(1:hLenU,:)     = W0*arrayCoefs(1:hLenU,:);
                arrayCoefs(hLenU+1:end,:) = U0*arrayCoefs(hLenU+1:end,:);
                %arrayCoefs = V0*arrayCoefs;
            end

        end
        
        function arrayCoefs = fullAtomExtTypeI_(obj,arrayCoefs)
            %
            isPeriodicExt = obj.IsPeriodicExt; % BoundaryOperation = 'Circular'
            %
            ord = obj.PolyPhaseOrder;
    
            % Order extension
            hOrd = uint32(ord/2);
            if hOrd > 0
                for iOrd = uint32(1):hOrd
                    paramMtxW1 = getParamMtx_(obj,4*iOrd-1);
                    paramMtxU1 = getParamMtx_(obj,4*iOrd+0);
                    paramMtxW2 = getParamMtx_(obj,4*iOrd+1);
                    paramMtxU2 = getParamMtx_(obj,4*iOrd+2);
                    %
                    arrayCoefs = supportExtTypeI_(obj,arrayCoefs,...
                        paramMtxW1,paramMtxU1,paramMtxW2,paramMtxU2,...
                        [],isPeriodicExt);
                end
            end
            
        end
        
        function arrayCoefs = fullAtomExtTypeII_(obj,arrayCoefs)
            %
            isPeriodicExt = obj.IsPeriodicExt; % BoundaryOperation = 'Circular'
            isPsGtPa      = obj.IsPsGreaterThanPa;            
            %
            ord = obj.PolyPhaseOrder;
            
            % Order extension
            for iOrd = uint32(1):uint32(ord/2)
                paramMtx1 = getParamMtx_(obj,4*iOrd-1); % hW
                paramMtx2 = getParamMtx_(obj,4*iOrd+0); % hU
                paramMtx3 = getParamMtx_(obj,4*iOrd+1); % W
                paramMtx4 = getParamMtx_(obj,4*iOrd+2); % U
                %
                %if isPsGtPa
                    arrayCoefs = supportExtTypeIIPsGtPa_(obj,arrayCoefs,...
                        paramMtx1,paramMtx2,paramMtx3,paramMtx4,...
                        [],isPeriodicExt);
                %else
                %    arrayCoefs = supportExtTypeIIPsLtPa_(obj,arrayCoefs,paramMtx1,paramMtx2,[],isPeriodicExt);
                %end
            end

        end
        
        function arrayCoefs = supportExtTypeI_(obj,arrayCoefs,paramMtxW1,paramMtxU1,paramMtxW2,paramMtxU2,~,isPeriodicExt)
            hLen = obj.NumberOfSymmetricChannels;

            % Phase 1
            Wx1 = paramMtxW1;
            Ux1 = paramMtxU1;
            arrayCoefs = blockButterflyTypeI_(obj,arrayCoefs,[]);
            arrayCoefs = rightShiftLowerCoefs_(obj,arrayCoefs);
            arrayCoefs = blockButterflyTypeI_(obj,arrayCoefs,[]);
            arrayCoefs = arrayCoefs/2.0;
            % Lower channel rotation
            if isPeriodicExt
                arrayCoefs(1:hLen,:) = Wx1*arrayCoefs(1:hLen,:);
                arrayCoefs(hLen+1:end,:) = Ux1*arrayCoefs(hLen+1:end,:);
            else %TODO:ë√ìñê´ïsñæÅDóvèCê≥
                arrayCoefs(1:hLen,1) = -arrayCoefs(1:hLen,1);
                arrayCoefs(1:hLen,2:end) = Wx1*arrayCoefs(1:hLen,2:end);
                
                arrayCoefs(hLen+1:end,1) = -arrayCoefs(hLen+1:end,1);
                arrayCoefs(hLen+1:end,2:end) = Ux1*arrayCoefs(hLen+1:end,2:end);
            end
            
            % Phase 2
            Wx2 = paramMtxW2;
            Ux2 = paramMtxU2;
            arrayCoefs = blockButterflyTypeI_(obj,arrayCoefs,[]);
            arrayCoefs = leftShiftUpperCoefs_(obj,arrayCoefs);
            arrayCoefs = blockButterflyTypeI_(obj,arrayCoefs,[]);
            arrayCoefs = arrayCoefs/2.0;
            % Lower channel rotation
            arrayCoefs(1:hLen,:) = Wx2*arrayCoefs(1:hLen,:);
            arrayCoefs(hLen+1:end,:) = Ux2*arrayCoefs(hLen+1:end,:);
        end
        
        function arrayCoefs = supportExtTypeIIPsGtPa_(obj,arrayCoefs,paramMtx1,paramMtx2,paramMtx3,paramMtx4,~,isPeriodicExt)
            hLen = obj.NumberOfSymmetricChannels;
            
            % Phase 1
            Wx = paramMtx3;
            Ux = paramMtx4;
            arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs,[]);
            arrayCoefs = rightShiftLowerCoefs_(obj,arrayCoefs);
            arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs,[]);
            arrayCoefs = arrayCoefs/2.0;
            % Lower channel rotation
            if isPeriodicExt
                arrayCoefs(1:hLen,:) = Wx*arrayCoefs(1:hLen,end,:);
                arrayCoefs(hLen+1:end,:) = Ux*arrayCoefs(hLen+1:end,:);
            else
                arrayCoefs(1:hLen,1) = -arrayCoefs(1:hLen,1);
                arrayCoefs(1:hLen,2:end) = Wx*arrayCoefs(1:hLen,2:end);
                
                arrayCoefs(hLen+1:end,1) = -arrayCoefs(hLen+1:end,1);
                arrayCoefs(hLen+1:end,2:end) = Ux*arrayCoefs(hLen+1:end,2:end);
            end
            
            % Phase 2
            Wx = paramMtx1;
            Ux = paramMtx2;
            arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs,[]);
            arrayCoefs = leftShiftUpperCoefs_(obj,arrayCoefs);
            arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs,[]);
            arrayCoefs = arrayCoefs/2.0;
            % Upper channel rotation
            arrayCoefs(1:hLen,:) = Wx*arrayCoefs(1:hLen,:);
            arrayCoefs(hLen+1:end,:) = Ux*arrayCoefs(hLen+1:end,:);
        end
        
%         function arrayCoefs = supportExtTypeIIPsLtPa_(obj,arrayCoefs,paramMtx1,paramMtx2,~,isPeriodicExt)
%             hLen = obj.NumberOfSymmetricChannels;
%             
%             % Phase 1
%             Wx = paramMtx1;
%             arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs,[]);
%             arrayCoefs = rightShiftLowerCoefs_(obj,arrayCoefs);
%             arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs,[]);
%             arrayCoefs = arrayCoefs/2.0;
%             % Upper channel rotation
%             if isPeriodicExt
%                 arrayCoefs(1:hLen,:) = Wx*arrayCoefs(1:hLen,:);
%             else
%                 arrayCoefs(1:hLen,1) = -arrayCoefs(1:hLen,1);
%                 arrayCoefs(1:hLen,2:end) = Wx*arrayCoefs(1:hLen,2:end);                
%             end
%             
%             % Phase 2
%             Ux = paramMtx2;
%             arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs,[]);
%             arrayCoefs = leftShiftUpperCoefs_(obj,arrayCoefs);
%             arrayCoefs = blockButterflyTypeII_(obj,arrayCoefs,[]);
%             arrayCoefs = arrayCoefs/2.0;
%             % Lower channel rotation
%             arrayCoefs(hLen+1:end,:) = Ux*arrayCoefs(hLen+1:end,:);
%         end
        
        function arrayCoefs = rightShiftLowerCoefs_(obj,arrayCoefs)
            hLenMn = max([ obj.NumberOfSymmetricChannels
                obj.NumberOfAntisymmetricChannels]);
            %
            lowerCoefsPre = arrayCoefs(hLenMn+1:end,end);
            arrayCoefs(hLenMn+1:end,2:end) = arrayCoefs(hLenMn+1:end,1:end-1);
            arrayCoefs(hLenMn+1:end,1) = lowerCoefsPre;
        end
        
        function arrayCoefs = leftShiftUpperCoefs_(obj,arrayCoefs)
            hLenMx = min([ obj.NumberOfSymmetricChannels
                obj.NumberOfAntisymmetricChannels]);
            %
            upperCoefsPost = arrayCoefs(1:hLenMx,1);
            arrayCoefs(1:hLenMx,1:end-1) = arrayCoefs(1:hLenMx,2:end);
            arrayCoefs(1:hLenMx,end) = upperCoefsPost;            
        end
    end
    
end
