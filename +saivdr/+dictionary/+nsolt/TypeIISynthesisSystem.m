classdef TypeIISynthesisSystem < saivdr.dictionary.nsolt.AbstNsoltSynthesisSystem %#~codegen
    %TYPEIISYNTHESISSYSTEM Synthesis system of Type-II NSOLT
    %
    % SVN identifier:
    % $Id: TypeIISynthesisSystem.m 683 2015-05-29 08:22:13Z sho $
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
        atomConcatenationHorizontalFcn
        atomConcatenationVerticalFcn
        atomConcatenationHorizontalObj
        atomConcatenationVerticalObj
    end
    
    methods
        
        % Constractor
        function obj = TypeIISynthesisSystem(varargin)
            obj = ...
                obj@saivdr.dictionary.nsolt.AbstNsoltSynthesisSystem(varargin{:});
        end
        
    end
    
    methods (Access=protected)
        
        function ps = getDefaultNumberOfSymmetricChannels(~)
            ps = 3;
        end
        
        function pa = getDefaultNumberOfAntisymmetricChannels(~)
            pa = 2;
        end

        function s = saveObjectImpl(obj)
            s = saveObjectImpl@saivdr.dictionary.nsolt.AbstNsoltSynthesisSystem(obj);
            s.atomConcatenationHorizontalFcn = obj.atomConcatenationHorizontalFcn;
            s.atomConcatenationVerticalFcn   = obj.atomConcatenationVerticalFcn;
            s.atomConcatenationHorizontalObj = obj.atomConcatenationHorizontalObj;
            s.atomConcatenationVerticalObj   = obj.atomConcatenationVerticalObj;            
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            obj.atomConcatenationHorizontalFcn = s.atomConcatenationHorizontalFcn;
            obj.atomConcatenationVerticalFcn   = s.atomConcatenationVerticalFcn;
            obj.atomConcatenationHorizontalObj = s.atomConcatenationHorizontalObj;
            obj.atomConcatenationVerticalObj   = s.atomConcatenationVerticalObj;            
            loadObjectImpl@saivdr.dictionary.nsolt.AbstNsoltSynthesisSystem(obj,s,wasLocked);
        end            
        
        function validatePropertiesImpl(obj)
            validatePropertiesImpl@saivdr.dictionary.nsolt.AbstNsoltSynthesisSystem(obj);
            if obj.NumberOfSymmetricChannels == ...
                    obj.NumberOfAntisymmetricChannels
                error('ps and pa must be the same as each other.')
            end
        end
        
        function setupImpl(obj,coefs,scales)
            setupImpl@saivdr.dictionary.nsolt.AbstNsoltSynthesisSystem(obj,coefs,scales);
            import saivdr.dictionary.nsolt.mexsrcs.fcn_autobuild
            [obj.atomConcatenationHorizontalFcn, hflag] = fcn_autobuild(...
                'fcn_AtomConcatenationHorizontalTypeII',...
                obj.NumberOfSymmetricChannels,...
                obj.NumberOfAntisymmetricChannels);
            if ~hflag
                obj.atomConcatenationHorizontalObj ...
                    = saivdr.dictionary.nsolt.mexsrcs.AtomConcatenationHorizontalTypeII();
                obj.atomConcatenationHorizontalFcn = ...
                    @(arrayCoefs,nRows,nCols,paramMtx1,paramMtx2,isPeriodicExt) ...                
                    step(obj.atomConcatenationHorizontalObj,...
                    arrayCoefs,nRows,nCols,paramMtx1,paramMtx2,isPeriodicExt);                
            end            
            [obj.atomConcatenationVerticalFcn, vflag] = fcn_autobuild(...
                'fcn_AtomConcatenationVerticalTypeII',...
                obj.NumberOfSymmetricChannels,...
                obj.NumberOfAntisymmetricChannels);
            if ~vflag
                obj.atomConcatenationVerticalObj ...
                    = saivdr.dictionary.nsolt.mexsrcs.AtomConcatenationVerticalTypeII();
                obj.atomConcatenationVerticalFcn = ...
                    @(arrayCoefs,nRows,nCols,paramMtx1,paramMtx2,isPeriodicExt) ...                
                    step(obj.atomConcatenationVerticalObj,...
                    arrayCoefs,nRows,nCols,paramMtx1,paramMtx2,isPeriodicExt);                
            end                        
        end
        
        function obj = synthesize_(obj,subCoefs)
            import saivdr.dictionary.utility.Direction
            import saivdr.dictionary.nsolt.TypeIISynthesisSystem
            isPeriodicExt_ = strcmp(obj.BoundaryOperation,'Circular');
            nChs_ = [obj.NumberOfSymmetricChannels obj.NumberOfAntisymmetricChannels];
            hLen_ = [obj.NumberOfSymmetricChannels obj.NumberOfAntisymmetricChannels];
            %            
            blockSize = zeros(1,2);
            blockSize(Direction.VERTICAL) = obj.decY;
            blockSize(Direction.HORIZONTAL) = obj.decX;
            if iscell(subCoefs)     % When input cell matrix
                height = size(subCoefs{1},1)*obj.decY;
                width  = size(subCoefs{1},2)*obj.decX;
                subCoefs = TypeIISynthesisSystem.getMatrix_(subCoefs,sum(nChs_));
            else
               height = size(subCoefs,1)*double(obj.decY)/sum(nChs_);
               width  = size(subCoefs,2)*double(obj.decX);                
            end
            if isinteger(obj.arrayCoefs)
                subCoefs = double(subCoefs);
            end
            
            obj.nRows = int32(height/double(obj.decY));
            obj.nCols = int32(width/double(obj.decX));
            obj.arrayCoefs = im2col(subCoefs,[sum(nChs_) 1],'distinct');
            
            % Linear combination of atoms
            numOfPMtx = double(get(obj.paramMtx,'NumberOfParameterMatrices'));
            import saivdr.dictionary.nsolt.mexsrcs.fcn_autobuild
            hOrdY = uint32(obj.ordY/2);
            for iOrd = uint32(1):hOrdY % Vertical process
                paramMtx1 = step(obj.paramMtx,[],numOfPMtx-2*iOrd+1); % W
                paramMtx2 = step(obj.paramMtx,[],numOfPMtx-2*iOrd+2); % U
                obj.arrayCoefs = obj.atomConcatenationVerticalFcn(...
                    obj.arrayCoefs,obj.nRows,obj.nCols,paramMtx1,paramMtx2,isPeriodicExt_);
            end
            hOrdX = uint32(obj.ordX/2);
            for iOrd = uint32(1):hOrdX % Horizontal process
                paramMtx1 = ...
                    step(obj.paramMtx,[],uint32(numOfPMtx-2*(hOrdY+iOrd)+1)); % W
                paramMtx2 = ...
                    step(obj.paramMtx,[],uint32(numOfPMtx-2*(hOrdY+iOrd)+2)); % U
                obj.arrayCoefs = obj.atomConcatenationHorizontalFcn(...
                    obj.arrayCoefs,obj.nRows,obj.nCols,paramMtx1,paramMtx2,isPeriodicExt_);
            end
            mc = ceil(double(obj.decX*obj.decY)/2);
            mf = floor(double(obj.decX*obj.decY)/2);
            if numOfPMtx > 1
                W0 = step(obj.paramMtx,[],uint32(1)).';
                U0 = step(obj.paramMtx,[],uint32(2)).';
                upperData = W0*obj.arrayCoefs(1:hLen_(1),:);
                lowerData = U0*obj.arrayCoefs(hLen_(1)+1:end,:);
            else
                upperData = obj.arrayCoefs(1:hLen_(1),:);
                lowerData = obj.arrayCoefs(hLen_(1)+1:end,:);
            end
            obj.arrayCoefs = [ upperData(1:mc,:) ; lowerData(1:mf,:) ];
            
            obj.arrayCoefs = col2im(obj.arrayCoefs,...
                [obj.decY obj.decX],[height width],'distinct');
            if obj.decY == 2 && obj.decX == 2
                subCoef1 = obj.arrayCoefs(1:2:end,1:2:end);
                subCoef2 = obj.arrayCoefs(2:2:end,1:2:end);
                subCoef3 = obj.arrayCoefs(1:2:end,2:2:end);
                subCoef4 = obj.arrayCoefs(2:2:end,2:2:end);
                obj.arrayCoefs(1:2:end,1:2:end) = ...
                    (subCoef1+subCoef2+subCoef3+subCoef4)/2;
                obj.arrayCoefs(2:2:end,1:2:end)  = ...
                    (subCoef1-subCoef2-subCoef3+subCoef4)/2;
                obj.arrayCoefs(1:2:end,2:2:end)  = ...
                    (subCoef1-subCoef2+subCoef3-subCoef4)/2;
                obj.arrayCoefs(2:2:end,2:2:end)  = ...
                    (subCoef1+subCoef2-subCoef3-subCoef4)/2;
            elseif ~(obj.decY == 1 && obj.decX == 1)
                obj.arrayCoefs = blockproc(obj.arrayCoefs,blockSize,...
                    @(x) TypeIISynthesisSystem.permuteIdctCoefs_(x));
                obj.arrayCoefs = blockproc(obj.arrayCoefs,blockSize,...
                    @(x) TypeIISynthesisSystem.idct2_(x));
            end
        end
        
    end
   
    methods (Access = private, Static = true )
        
        function value = permuteIdctCoefs_(x)
            coefs = x.data;
            decY_ = x.blockSize(1);
            decX_ = x.blockSize(2);
            nQDecsee = ceil(decY_/2)*ceil(decX_/2);
            nQDecsoo = floor(decY_/2)*floor(decX_/2);
            nQDecsoe = floor(decY_/2)*ceil(decX_/2);
            cee = coefs(         1:  nQDecsee);
            coo = coefs(nQDecsee+1:nQDecsee+nQDecsoo);
            coe = coefs(nQDecsee+nQDecsoo+1:nQDecsee+nQDecsoo+nQDecsoe);
            ceo = coefs(nQDecsee+nQDecsoo+nQDecsoe+1:end);
            value = zeros(decY_,decX_);
            value(1:2:decY_,1:2:decX_) = reshape(cee,ceil(decY_/2),ceil(decX_/2));
            value(2:2:decY_,2:2:decX_) = reshape(coo,floor(decY_/2),floor(decX_/2));
            value(2:2:decY_,1:2:decX_) = reshape(coe,floor(decY_/2),ceil(decX_/2));
            value(1:2:decY_,2:2:decX_) = reshape(ceo,ceil(decY_/2),floor(decX_/2));
        end

    end
    
end

