classdef TypeIAnalysisSystem < saivdr.dictionary.nsolt.AbstNsoltAnalysisSystem %#~codegen
    %TYPEIANALYSISSYSTEM Analysis system of Type-I NSOLT
    %
    % SVN identifier:
    % $Id: TypeIAnalysisSystem.m 683 2015-05-29 08:22:13Z sho $
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

    properties (Access = private)
        supportExtensionHorizontalFcn
        supportExtensionVerticalFcn
        supportExtensionHorizontalObj
        supportExtensionVerticalObj        
    end
    
    methods
        
        % Constructor
        function obj = TypeIAnalysisSystem(varargin)
            obj = ...
                obj@saivdr.dictionary.nsolt.AbstNsoltAnalysisSystem(varargin{:});
        end
        
    end

    methods (Access=protected)
        
        function ps = getDefaultNumberOfSymmetricChannels(~)
            ps = 2;
        end
        
        function pa = getDefaultNumberOfAntisymmetricChannels(~)
            pa = 2;
        end
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@saivdr.dictionary.nsolt.AbstNsoltAnalysisSystem(obj);
            s.supportExtensionHorizontalFcn = obj.supportExtensionHorizontalFcn;
            s.supportExtensionVerticalFcn   = obj.supportExtensionVerticalFcn;
            s.supportExtensionHorizontalObj = obj.supportExtensionHorizontalObj;
            s.supportExtensionVerticalObj   = obj.supportExtensionVerticalObj;            
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            obj.supportExtensionHorizontalFcn = s.supportExtensionHorizontalFcn;
            obj.supportExtensionVerticalFcn   = s.supportExtensionVerticalFcn;
            obj.supportExtensionHorizontalObj = s.supportExtensionHorizontalObj;
            obj.supportExtensionVerticalObj   = s.supportExtensionVerticalObj;            
            loadObjectImpl@saivdr.dictionary.nsolt.AbstNsoltAnalysisSystem(obj,s,wasLocked);
        end
        
        function validatePropertiesImpl(obj)
            validatePropertiesImpl@saivdr.dictionary.nsolt.AbstNsoltAnalysisSystem(obj);
            if obj.NumberOfSymmetricChannels ~= ...
                    obj.NumberOfAntisymmetricChannels
                error('ps and pa must be the same as each other.')
            end
        end
        
        function setupImpl(obj,coefs,scales)
            setupImpl@saivdr.dictionary.nsolt.AbstNsoltAnalysisSystem(obj,coefs,scales);
            import saivdr.dictionary.nsolt.mexsrcs.fcn_autobuild            
            [obj.supportExtensionHorizontalFcn, hflag] = fcn_autobuild(...
                'fcn_SupportExtensionHorizontalTypeI',...
                obj.NumberOfSymmetricChannels,...
                obj.NumberOfAntisymmetricChannels);            
            if ~hflag
                obj.supportExtensionHorizontalObj ...
                    = saivdr.dictionary.nsolt.mexsrcs.SupportExtensionHorizontalTypeI();
                obj.supportExtensionHorizontalFcn = ...
                    @(arrayCoefs,nRows,nCols,paramMtx1,paramMtx2,isPeriodicExt) ...                
                    step(obj.supportExtensionHorizontalObj,...
                    arrayCoefs,nRows,nCols,paramMtx1,paramMtx2,isPeriodicExt);                
            end
            [obj.supportExtensionVerticalFcn,   vflag] = fcn_autobuild(...
                'fcn_SupportExtensionVerticalTypeI',...
                obj.NumberOfSymmetricChannels,...
                obj.NumberOfAntisymmetricChannels);
            if ~vflag
                obj.supportExtensionVerticalObj ...
                    = saivdr.dictionary.nsolt.mexsrcs.SupportExtensionVerticalTypeI();
                obj.supportExtensionVerticalFcn = ...
                    @(arrayCoefs,nRows,nCols,paramMtx1,paramMtx2,isPeriodicExt) ...
                    step(obj.supportExtensionVerticalObj, ...
                    arrayCoefs,nRows,nCols,paramMtx1,paramMtx2,isPeriodicExt);
            end
        end
        
        function analyze_(obj,srcImg)
            import saivdr.dictionary.utility.Direction
            import saivdr.dictionary.nsolt.TypeIAnalysisSystem
            isPeriodicExt = strcmp(obj.BoundaryOperation,'Circular');
            nChs_ = [obj.NumberOfSymmetricChannels obj.NumberOfAntisymmetricChannels];
            hLen_ = obj.NumberOfSymmetricChannels;
            %
            height = size(srcImg,1);
            width = size(srcImg,2);
            obj.nRows = int32(height/double(obj.decY));
            obj.nCols = int32(width/double(obj.decX));
            blockSize = zeros(1,2);
            blockSize(Direction.VERTICAL) = obj.decY;
            blockSize(Direction.HORIZONTAL) = obj.decX;
            
            if isinteger(srcImg)
                srcImg = im2double(srcImg);
            end
            
            % Block DCT
            if obj.decY == 1 && obj.decX == 1
                dctCoefs = srcImg;
            elseif obj.decY == 2 && obj.decX == 2
                dctCoefs = zeros(size(srcImg));
                subImg1 = srcImg(1:2:end,1:2:end);
                subImg2 = srcImg(2:2:end,1:2:end);
                subImg3 = srcImg(1:2:end,2:2:end);
                subImg4 = srcImg(2:2:end,2:2:end);
                dctCoefs(1:2:end,1:2:end) = ...
                    (subImg1+subImg2+subImg3+subImg4)/2;
                dctCoefs(2:2:end,1:2:end)  = ...
                    (subImg1-subImg2-subImg3+subImg4)/2;
                dctCoefs(1:2:end,2:2:end)  = ...
                    (subImg1-subImg2+subImg3-subImg4)/2;
                dctCoefs(2:2:end,2:2:end)  = ...
                    (subImg1+subImg2-subImg3-subImg4)/2;
            else
                dctCoefs = blockproc(srcImg,blockSize,...
                    @(x) TypeIAnalysisSystem.dct2_(x));
                dctCoefs = blockproc(dctCoefs,blockSize,...
                    @(x) TypeIAnalysisSystem.permuteDctCoefs_(x));
            end
            mc = ceil(obj.decX*obj.decY/2);
            mf = floor(obj.decX*obj.decY/2);
            coefs = im2col(dctCoefs,blockSize,'distinct');
            obj.arrayCoefs = zeros(sum(nChs_),size(coefs,2));
            obj.arrayCoefs(1:mc,:) = coefs(1:mc,:);
            obj.arrayCoefs(hLen_(1)+1:hLen_+mf,:) = coefs(mc+1:end,:);
            
            if ~isempty(obj.paramMtx)
                W0 = step(obj.paramMtx,[],uint32(1));
                U0 = step(obj.paramMtx,[],uint32(2));
                upperData = W0*obj.arrayCoefs(1:hLen_(1),:);
                lowerData = U0*obj.arrayCoefs(hLen_(1)+1:end,:);
                obj.arrayCoefs = [ upperData ; lowerData ];
            end
            
            % TODO: Insert codes for odd olyphase order
            
            % Support extension
            for iOrd = uint32(1):obj.ordX/2  % Horizontal process
                paramMtx1 = step(obj.paramMtx,[],2*iOrd+1);
                paramMtx2 = step(obj.paramMtx,[],2*iOrd+2);
                %
                obj.arrayCoefs = obj.supportExtensionHorizontalFcn(...
                    obj.arrayCoefs,obj.nRows,obj.nCols,paramMtx1,paramMtx2,isPeriodicExt);
            end
            
            for iOrd = uint32(1):obj.ordY/2  % Vertical process
                paramMtx1 = ...
                    step(obj.paramMtx,[],2*iOrd+obj.ordX+1);
                paramMtx2 = ...
                    step(obj.paramMtx,[],2*iOrd+obj.ordX+2);
                obj.arrayCoefs = obj.supportExtensionVerticalFcn(...
                    obj.arrayCoefs,obj.nRows,obj.nCols,paramMtx1,paramMtx2,isPeriodicExt);
            end
        end
    end

end
