classdef AbstOvsdLpPuFb2dTypeIISystem < ...
        saivdr.dictionary.nsolt.AbstOvsdLpPuFb2dSystem %#codegen
    %ABSTOVSDLPPUFB2DTYPEIISYSTEM Abstract class 2-D Type-II OLPPUFB
    %
    % SVN identifier:
    % $Id: AbstOvsdLpPuFb2dTypeIISystem.m 683 2015-05-29 08:22:13Z sho $
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
    
    properties (Access = protected)
        matrixE0
    end

    properties (Access = protected,PositiveInteger)
        nStages
    end

    methods (Access = protected, Static = true, Abstract = true)
        value = getDefaultPolyPhaseOrder_()
    end        
    
    methods
        function obj = AbstOvsdLpPuFb2dTypeIISystem(varargin)
            obj = obj@saivdr.dictionary.nsolt.AbstOvsdLpPuFb2dSystem(...
                varargin{:});
            updateProperties_(obj);
            updateAngles_(obj);
            updateMus_(obj);
        end
    end
    
    methods (Access = protected)
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@saivdr.dictionary.nsolt.AbstOvsdLpPuFb2dSystem(obj);
            s.nStages = obj.nStages;
            s.matrixE0 = obj.matrixE0;
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            obj.nStages = s.nStages;
            obj.matrixE0 = s.matrixE0;
            loadObjectImpl@saivdr.dictionary.nsolt.AbstOvsdLpPuFb2dSystem(obj,s,wasLocked);
        end
        
        %function processTunedPropertiesImpl(obj)
        %end
        
        function updateProperties_(obj)
            import saivdr.dictionary.nsolt.ChannelGroup
            import saivdr.dictionary.utility.Direction
            import saivdr.dictionary.utility.ParameterMatrixSet
            
            % Check DecimationFactor
            if length(obj.DecimationFactor) > 2
                error('Dimension of DecimationFactor must be less than or equal to two.');
            end
            nHalfDecs = prod(obj.DecimationFactor)/2;
            
            % Check PolyPhaseOrder
            if isempty(obj.PolyPhaseOrder)
                obj.PolyPhaseOrder = obj.getDefaultPolyPhaseOrder_();
            end            
            if length(obj.PolyPhaseOrder) > 2
                error('Dimension of PolyPhaseOrder must be less than or equal to two.');
            end
            ordX = obj.PolyPhaseOrder(Direction.HORIZONTAL);
            ordY = obj.PolyPhaseOrder(Direction.VERTICAL);
            if mod(ordX,2)~=0 || mod(ordY,2)~=0
                error('Polyphase orders must be even.');
            end
            obj.nStages = uint32(1+double(ordX)/2+double(ordY)/2);
            obj.matrixE0 = getMatrixE0_(obj);
            
            % Check NumberOfChannels
            if length(obj.NumberOfChannels) > 2
                error('Dimension of NumberOfChannels must be less than or equal to two.');
            end
            if isempty(obj.NumberOfChannels)
                obj.NumberOfChannels = [ floor(nHalfDecs+1) floor(nHalfDecs) ];
            elseif length(obj.NumberOfChannels) == 1
                if mod(obj.NumberOfChannels,2) == 0
                    id = 'SaivDr:IllegalArgumentException';
                    msg = '#Channels must be odd.';
                    me = MException(id, msg);
                    throw(me);
                else
                    obj.NumberOfChannels = ...
                        [ ceil(double(obj.NumberOfChannels)/2) ...
                        floor(double(obj.NumberOfChannels)/2) ];
                end
            elseif obj.NumberOfChannels(Direction.VERTICAL) <= ...
                    obj.NumberOfChannels(Direction.HORIZONTAL)
                id = 'SaivDr:IllegalArgumentException';
                msg = 'ps must be greater than pa.';
                me = MException(id, msg);
                throw(me);
            end
            
            % Prepare ParameterMatrixSet
            paramMtxSizeTab = repmat(...
                [ obj.NumberOfChannels(ChannelGroup.UPPER) ;
                obj.NumberOfChannels(ChannelGroup.LOWER) ],...
                obj.nStages,2);
            obj.ParameterMatrixSet = ParameterMatrixSet(...
                'MatrixSizeTable',paramMtxSizeTab);
        end
        
        function updateAngles_(obj)
            import saivdr.dictionary.nsolt.ChannelGroup
            nAngsPerStg = zeros(2,1);
            %
            nAngsPerStg(ChannelGroup.UPPER) = ...
                obj.NumberOfChannels(ChannelGroup.UPPER) ...
                *double(obj.NumberOfChannels(ChannelGroup.UPPER)-1)/2;
            nAngsPerStg(ChannelGroup.LOWER) = ...
                obj.NumberOfChannels(ChannelGroup.LOWER) ...
                *double(obj.NumberOfChannels(ChannelGroup.LOWER)-1)/2;
            sizeOfAngles = [sum(nAngsPerStg) obj.nStages];
            %
            
            if isscalar(obj.Angles) && obj.Angles == 0
                obj.Angles = zeros(sizeOfAngles);
            end
            if size(obj.Angles,1) ~= sizeOfAngles(1) || ...
                    size(obj.Angles,2) ~= sizeOfAngles(2)
                id = 'SaivDr:IllegalArgumentException';
                msg = sprintf(...
                    'Size of angles must be [ %d %d ]',...
                    sizeOfAngles(1), sizeOfAngles(2));
                me = MException(id, msg);
                throw(me);
            end
        end
        
        function updateMus_(obj)
            import saivdr.dictionary.nsolt.ChannelGroup
            %
            sizeOfMus = [ sum(obj.NumberOfChannels) obj.nStages ];
            %
            nChL = obj.NumberOfChannels(ChannelGroup.LOWER);
            nChU = obj.NumberOfChannels(ChannelGroup.UPPER);
            if isscalar(obj.Mus) && obj.Mus == 1
                obj.Mus = [ 
                     ones(nChU, obj.nStages);
                    -ones(nChL, obj.nStages) ];
                obj.Mus(:,1) = ones(size(obj.Mus,1),1);
            end
            if size(obj.Mus,1) ~= sizeOfMus(1) || ...
                    size(obj.Mus,2) ~= sizeOfMus(2)
                id = 'SaivDr:IllegalArgumentException';
                msg = sprintf(...
                    'Size of mus must be [ %d %d ]',...
                    sizeOfMus(1), sizeOfMus(2));
                me = MException(id, msg);
                throw(me);
            end
        end
        
        function value = getAnalysisFilterBank_(obj)
            import saivdr.dictionary.utility.Direction
            import saivdr.dictionary.nsolt.ChannelGroup
            import saivdr.dictionary.nsolt.AbstOvsdLpPuFb2dTypeIISystem
            %
            nChs = obj.NumberOfChannels;
            dec  = obj.DecimationFactor;
            decX = dec(Direction.HORIZONTAL);
            decY = dec(Direction.VERTICAL);
            nHalfDecs = prod(dec)/2;
            ordX = obj.PolyPhaseOrder(Direction.HORIZONTAL);
            ordY = obj.PolyPhaseOrder(Direction.VERTICAL);
            %
            cI = getMatrixIc_(obj);
            fI = getMatrixIf_(obj);
            E0 = obj.matrixE0;
            %
            cM_2 = ceil(nHalfDecs);
            W = step(obj.ParameterMatrixSet,[],uint32(1))*[ eye(cM_2) ;
                zeros(nChs(ChannelGroup.UPPER)-cM_2,cM_2)];
            fM_2 = floor(nHalfDecs);
            U = step(obj.ParameterMatrixSet,[],uint32(2))*[ eye(fM_2);
                zeros(nChs(ChannelGroup.LOWER)-fM_2,fM_2) ];
            R = blkdiag(W,U);
            E = R*E0;
            iParamMtx = uint32(3);
            
            % Horizontal extention
            for iOrdX = 1:uint32(double(ordX)/2)
                W = step(obj.ParameterMatrixSet,[],iParamMtx);
                U = step(obj.ParameterMatrixSet,[],iParamMtx+1);
                R = blkdiag(cI,U);
                E = R*AbstOvsdLpPuFb2dTypeIISystem.processQxO_(E,dec,nChs);
                R = blkdiag(W,fI);
                E = R*AbstOvsdLpPuFb2dTypeIISystem.processQxE_(E,dec,nChs);
                iParamMtx = iParamMtx+2;
            end
            
            % Vertical extention
            ord = zeros(2,1);
            ord(Direction.HORIZONTAL)  = ordX;
            for iOrdY = 1:uint32(double(ordY)/2)
                ord(Direction.VERTICAL)  = 2*iOrdY-1;
                W = step(obj.ParameterMatrixSet,[],iParamMtx);
                U = step(obj.ParameterMatrixSet,[],iParamMtx+1);
                R = blkdiag(cI,U);
                E = R*AbstOvsdLpPuFb2dTypeIISystem.processQyO_(E,dec,nChs,ord);
                ord(Direction.VERTICAL)  = 2*iOrdY;
                R = blkdiag(W,fI);
                E = R*AbstOvsdLpPuFb2dTypeIISystem.processQyE_(E,dec,nChs,ord);
                iParamMtx = iParamMtx+2;
            end
            nSubbands = size(E,1);
            lenY = decY*(ordY+1);
            lenX = decX*(ordX+1);
            value = zeros(lenY,lenX,nSubbands);
            for iSubband = 1:nSubbands
                value(:,:,iSubband) = reshape(E(iSubband,:),lenY,lenX);
            end
        end
        
        function value = getMatrixIc_(obj)
            import saivdr.dictionary.nsolt.ChannelGroup
            value = eye(obj.NumberOfChannels(ChannelGroup.UPPER));
        end
        
        function value = getMatrixIf_(obj)
            import saivdr.dictionary.nsolt.ChannelGroup
            value = eye(obj.NumberOfChannels(ChannelGroup.LOWER));
        end
        %{
        function value = getDelayChain_(obj)
            import saivdr.dictionary.nsolt.PolyPhaseMatrix2d
            idx = 1;
            delay = zeros(obj.decX*obj.decY,1,obj.decY,obj.decX);
            for iBndX = 1:obj.decX
                for iBndY = 1:obj.decY
                    delay(idx,:,iBndY,iBndX) = 1;
                    idx = idx + 1;
                end
            end
            value = PolyPhaseMatrix2d(delay);
        end
        %}
    end
    
    
    methods (Access = protected, Static = true)
        
        function value = processQxO_(x,dec,ch)
            import saivdr.dictionary.nsolt.AbstOvsdLpPuFb2dTypeIISystem
            nChMx = max(ch);
            nChMn = min(ch);
            value = AbstOvsdLpPuFb2dTypeIISystem.butterfly_(x,nChMx,nChMn);
            % Block delay lower Coefs. in Horizontal direction
            Zu = zeros(nChMn,prod(dec));
            Zl = zeros(nChMx,prod(dec));
            value = [
                value(1:nChMn,:) Zu;
                Zl value(nChMn+1:end,:) ];
            value = AbstOvsdLpPuFb2dTypeIISystem.butterfly_(value,nChMx,nChMn)/2.0;
        end
        
        function value = processQxE_(x,dec,ch)
            import saivdr.dictionary.nsolt.AbstOvsdLpPuFb2dTypeIISystem
            nChMx = max(ch);
            nChMn = min(ch);
            value = AbstOvsdLpPuFb2dTypeIISystem.butterfly_(x,nChMx,nChMn);
            % Block delay lower Coefs. in Horizontal direction
            Zu = zeros(nChMx,prod(dec));
            Zl = zeros(nChMn,prod(dec));
            value = [
                value(1:nChMx,:) Zu;
                Zl value(nChMx+1:end,:) ];
            value = AbstOvsdLpPuFb2dTypeIISystem.butterfly_(value,nChMx,nChMn)/2.0;
        end
        
        function value = processQyO_(x,dec,ch,ord)
            import saivdr.dictionary.utility.Direction
            import saivdr.dictionary.nsolt.AbstOvsdLpPuFb2dTypeIISystem
            decY_ = dec(Direction.VERTICAL);
            decX_ = dec(Direction.HORIZONTAL);
            ordY_ = ord(Direction.VERTICAL);
            ordX_ = ord(Direction.HORIZONTAL);
            nChMx = max(ch);
            nChMn = min(ch);
            value = AbstOvsdLpPuFb2dTypeIISystem.butterfly_(x,nChMx,nChMn);
            % Block delay lower Coefs. in Horizontal direction
            nTapsYp = decY_*ordY_;
            nTapsY = decY_*(ordY_+1);
            nTapsX = decX_*(ordX_+1);
            nTaps = nTapsY*nTapsX;
            upper = zeros(nChMn,nTaps);
            lower = zeros(nChMx,nTaps);
            for idx = 0:decX_*(ordX_+1)-1
                range0 = idx*nTapsYp+1:(idx+1)*nTapsYp;
                range1 = idx*nTapsY+1:(idx+1)*nTapsY-decY_;
                range2 = idx*nTapsY+decY_+1:(idx+1)*nTapsY;
                upper(:,range1) = value(1:nChMn,    range0);
                lower(:,range2) = value(nChMn+1:end,range0);
            end
            value = [ upper ; lower ];
            value = AbstOvsdLpPuFb2dTypeIISystem.butterfly_(value,nChMx,nChMn)/2.0;
        end
        
        function value = processQyE_(x,dec,ch,ord)
            import saivdr.dictionary.nsolt.AbstOvsdLpPuFb2dTypeIISystem
            import saivdr.dictionary.utility.Direction
            decY_ = dec(Direction.VERTICAL);
            decX_ = dec(Direction.HORIZONTAL);
            ordY_ = ord(Direction.VERTICAL);
            ordX_ = ord(Direction.HORIZONTAL);
            nChMx = max(ch);
            nChMn = min(ch);
            value = AbstOvsdLpPuFb2dTypeIISystem.butterfly_(x,nChMx,nChMn);
            % Block delay lower Coefs. in Horizontal direction
            nTapsYp = decY_*ordY_;
            nTapsY = decY_*(ordY_+1);
            nTapsX = decX_*(ordX_+1);
            nTaps = nTapsY*nTapsX;
            upper = zeros(nChMx,nTaps);
            lower = zeros(nChMn,nTaps);
            for idx = 0:decX_*(ordX_+1)-1
                range0 = idx*nTapsYp+1:(idx+1)*nTapsYp;
                range1 = idx*nTapsY+1:(idx+1)*nTapsY-decY_;
                range2 = idx*nTapsY+decY_+1:(idx+1)*nTapsY;
                upper(:,range1) = value(1:nChMx,  range0);
                lower(:,range2) = value(nChMx+1:end,range0);
            end
            value = [ upper ; lower ];
            value = AbstOvsdLpPuFb2dTypeIISystem.butterfly_(value,nChMx,nChMn)/2.0;
        end
        
        function value = butterfly_(x,nChMx,nChMn)
            %hChs_ = floor(size(x,1)/2);
            upper = x(1:nChMn,:);
            middle = x(nChMn+1:nChMx,:);
            lower = x(nChMx+1:end,:);
            value = [
                upper+lower ;
                1.414213562373095*middle;
                upper-lower ];
        end
    end
    
end
