classdef AbstOvsdLpPuFb2dTypeISystem < ...
        saivdr.dictionary.nsolt.AbstOvsdLpPuFb2dSystem %#codegen
    %ABSTOVSDLPPUFB2DTYPEISYSTEM Abstract class 2-D Type-I OLPPUFB 
    % 
    % SVN identifier:
    % $Id: AbstOvsdLpPuFb2dTypeISystem.m 683 2015-05-29 08:22:13Z sho $
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
        anglesT
        musT
    end
    
    properties (Access = protected,PositiveInteger)
        nStages
    end

    methods (Access = protected, Static = true, Abstract = true)
        value = getDefaultPolyPhaseOrder_()
    end    
    
    methods
        function obj = AbstOvsdLpPuFb2dTypeISystem(varargin)
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
            s.anglesT = obj.anglesT;
            s.musT = obj.musT;
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            obj.nStages = s.nStages;
            obj.matrixE0 = s.matrixE0;
            obj.anglesT = s.anglesT;
            obj.musT = s.musT;
            loadObjectImpl@saivdr.dictionary.nsolt.AbstOvsdLpPuFb2dSystem(obj,s,wasLocked);
        end
        
        function processTunedPropertiesImpl(obj)
            import saivdr.dictionary.nsolt.AbstOvsdLpPuFb2dTypeISystem
            import saivdr.dictionary.nsolt.ChannelGroup
            import saivdr.dictionary.utility.Direction
            ordX = obj.PolyPhaseOrder(Direction.HORIZONTAL);
            nChL = obj.NumberOfChannels(ChannelGroup.LOWER);
            if isscalar(obj.Angles) && obj.Angles == 0 && ...
                    mod(ordX,2) == 1 && mod(nchL,2) == 0
                [obj.anglesT,obj.musT] = ...
                    AbstOvsdLpPuFb2dTypeISystem.calcGvnRotT_(nChL);
            end
        end
        
        function updateProperties_(obj)
            import saivdr.dictionary.nsolt.ChannelGroup
            import saivdr.dictionary.utility.Direction
            import saivdr.dictionary.utility.ParameterMatrixSet
            import saivdr.dictionary.nsolt.AbstOvsdLpPuFb2dTypeISystem
            
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
            obj.nStages = uint32(1+ordX+ordY);
            obj.matrixE0 = getMatrixE0_(obj);
            
            % Check NumberOfChannels
            if length(obj.NumberOfChannels) > 2
                error('Dimension of NumberOfChannels must be less than or equal to two.');
            end
            if isempty(obj.NumberOfChannels)
                obj.NumberOfChannels = nHalfDecs * [ 1 1 ];
            elseif isscalar(obj.NumberOfChannels)
                if mod(obj.NumberOfChannels,2) ~= 0
                    id = 'SaivDr:IllegalArgumentException';
                    msg = '#Channels must be even.';
                    me = MException(id, msg);
                    throw(me);
                else
                    obj.NumberOfChannels = ...
                        obj.NumberOfChannels * [ 1 1 ]/2;
                end
            elseif obj.NumberOfChannels(ChannelGroup.UPPER) ~= ...
                    obj.NumberOfChannels(ChannelGroup.LOWER)
                id = 'SaivDr:IllegalArgumentException';
                msg = 'ps and pa must be the same as each other.';
                me = MException(id, msg);
                throw(me);
            end
            
            % Prepare ParameterMatrixSet
            paramMtxSizeTab = ...
                obj.NumberOfChannels(ChannelGroup.LOWER)*ones(obj.nStages+1,2);
            obj.ParameterMatrixSet = ParameterMatrixSet(...
                'MatrixSizeTable',paramMtxSizeTab);
            %
            nChL = obj.NumberOfChannels(ChannelGroup.LOWER);
            if isscalar(obj.Angles) && obj.Angles == 0 && ...
                    mod(ordX,2) == 1 && mod(nChL,2) == 0
                [obj.anglesT,obj.musT] = ...
                    AbstOvsdLpPuFb2dTypeISystem.calcGvnRotT_(nChL);
            end
            %
        end
        
        function updateAngles_(obj)
            import saivdr.dictionary.nsolt.ChannelGroup
            import saivdr.dictionary.utility.Direction
            nChL = obj.NumberOfChannels(ChannelGroup.LOWER);
            nAngsPerStg = nChL*(nChL-1)/2;
            sizeOfAngles = [nAngsPerStg obj.nStages+1];
            ordX = obj.PolyPhaseOrder(Direction.HORIZONTAL);
            if isscalar(obj.Angles) && obj.Angles==0
                obj.Angles = zeros(sizeOfAngles);
                if mod(ordX,2)==1 && mod(nChL,2)==0
                    obj.Angles(:,2) = obj.anglesT;
                    obj.Angles(:,3) = obj.anglesT;
                end
            end
            if size(obj.Angles,1) ~= sizeOfAngles(1) || ...
                    size(obj.Angles,2) ~= sizeOfAngles(2)
                id = 'SaivDr:IllegalArgumentException';
                coder.extrinsic('sprintf')
                msg = sprintf(...
                    'Size of angles must be [ %d %d ]',...
                    sizeOfAngles(1), sizeOfAngles(2));
                me = MException(id, msg);
                throw(me);
            end
        end
        
        function updateMus_(obj)
            import saivdr.dictionary.nsolt.ChannelGroup
            import saivdr.dictionary.utility.Direction
            nChL = obj.NumberOfChannels(ChannelGroup.LOWER);
            sizeOfMus = [ nChL obj.nStages+1 ];
            ordX = obj.PolyPhaseOrder(Direction.HORIZONTAL);
            ordY = obj.PolyPhaseOrder(Direction.VERTICAL);
            if isscalar(obj.Mus) && obj.Mus==1
                obj.Mus = -ones(sizeOfMus);
                obj.Mus(:,1:2) = ones(size(obj.Mus,1),2);
                if mod(ordX,2)==1 && ~isempty(obj.musT)
                    obj.Mus(:,2) = obj.musT;
                    obj.Mus(:,3) = obj.musT;
                end
                if mod(ordY,2)==1
                    obj.Mus(:,ordX+3) = ones(sizeOfMus(1),1);
                end
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
            import saivdr.dictionary.nsolt.ChannelGroup
            import saivdr.dictionary.utility.Direction
            import saivdr.dictionary.nsolt.AbstOvsdLpPuFb2dTypeISystem
            %
            nChs = obj.NumberOfChannels;
            dec  = obj.DecimationFactor;
            decX = dec(Direction.HORIZONTAL);
            decY = dec(Direction.VERTICAL);
            nHalfDecs = prod(dec)/2;
            ordX = obj.PolyPhaseOrder(Direction.HORIZONTAL);
            ordY = obj.PolyPhaseOrder(Direction.VERTICAL);
            %
            I = getMatrixI_(obj);
            E0 = obj.matrixE0;
            %
            cM_2 = ceil(nHalfDecs);
            W = step(obj.ParameterMatrixSet,[],uint32(1))*[ eye(cM_2) ;
                zeros(nChs(ChannelGroup.LOWER)-cM_2,cM_2)];
            fM_2 = floor(nHalfDecs);
            U = step(obj.ParameterMatrixSet,[],uint32(2))*[ eye(fM_2);
                zeros(nChs(ChannelGroup.LOWER)-fM_2,fM_2) ];
            R = blkdiag(W,U);
            E = R*E0;
            iParamMtx = uint32(3);
            % Horizontal extention
            for iOrdX = 1:ordX
                U = step(obj.ParameterMatrixSet,[],iParamMtx);
                R = blkdiag(I,U);
                E = R*AbstOvsdLpPuFb2dTypeISystem.processQx_(E,dec);
                iParamMtx = iParamMtx+1;
            end
            % Vertical extention
            for iOrdY = 1:ordY
                ord  = [iOrdY ordX];
                U = step(obj.ParameterMatrixSet,[],iParamMtx);
                R = blkdiag(I,U);
                E = R*AbstOvsdLpPuFb2dTypeISystem.processQy_(E,dec,ord);
                iParamMtx = iParamMtx+1;
            end
            nSubbands = size(E,1);
            lenY = decY*(ordY+1);
            lenX = decX*(ordX+1);
            value = zeros(lenY,lenX,nSubbands);
            for iSubband = 1:nSubbands
                value(:,:,iSubband) = reshape(E(iSubband,:),lenY,lenX);
            end
            
        end
        
        function value = getMatrixI_(obj)
            import saivdr.dictionary.nsolt.ChannelGroup
            value = eye(ceil(obj.NumberOfChannels(ChannelGroup.LOWER)));
        end

    end
    
    methods (Access = protected, Static = true)
        
        function [anglesT,musT] = calcGvnRotT_(nChL)
            import saivdr.dictionary.nsolt.ChannelGroup
            import saivdr.dictionary.utility.OrthonormalMatrixFactorizationSystem
            omfs = OrthonormalMatrixFactorizationSystem();
            nQ = nChL/2;
            matrixT = [zeros(nQ) eye(nQ) ; eye(nQ) zeros(nQ)];
            [anglesT,musT] = step(omfs,matrixT);
        end
        
        function value = processQx_(x,dec_)
            import saivdr.dictionary.nsolt.AbstOvsdLpPuFb2dTypeISystem
            nChs_ = size(x,1);
            hChs_ = nChs_/2;
            value = AbstOvsdLpPuFb2dTypeISystem.butterfly_(x);
            % Block delay lower Coefs. in Horizontal direction
            Z = zeros(hChs_,prod(dec_));
            value = [
                value(1:hChs_,:) Z;
                Z value(hChs_+1:end,:) ];
            value = AbstOvsdLpPuFb2dTypeISystem.butterfly_(value)/2.0;
        end
        
        function value = processQy_(x,dec,ord)
            import saivdr.dictionary.utility.Direction
            import saivdr.dictionary.nsolt.AbstOvsdLpPuFb2dTypeISystem
            decY_ = dec(Direction.VERTICAL);
            decX_ = dec(Direction.HORIZONTAL);
            ordY_ = ord(Direction.VERTICAL);
            ordX_ = ord(Direction.HORIZONTAL);
            nChs_ = size(x,1);
            hChs = nChs_/2;
            value = AbstOvsdLpPuFb2dTypeISystem.butterfly_(x);
            % Block delay lower Coefs. in Horizontal direction
            nTapsYp = decY_*ordY_;
            nTapsY = decY_*(ordY_+1);
            nTapsX = decX_*(ordX_+1);
            nTaps = nTapsY*nTapsX;
            upper = zeros(hChs,nTaps);
            lower = zeros(hChs,nTaps);
            for idx = 0:decX_*(ordX_+1)-1
                range0 = idx*nTapsYp+1:(idx+1)*nTapsYp;
                range1 = idx*nTapsY+1:(idx+1)*nTapsY-decY_;
                range2 = idx*nTapsY+decY_+1:(idx+1)*nTapsY;
                upper(:,range1) = value(1:hChs,    range0);
                lower(:,range2) = value(hChs+1:end,range0);
            end
            value = [ upper ; lower ];
            value = AbstOvsdLpPuFb2dTypeISystem.butterfly_(value)/2.0;
        end
        
        function value = butterfly_(x)
            hChs_ = size(x,1)/2;
            upper = x(1:hChs_,:);
            lower = x(hChs_+1:end,:);
            value = [
                upper+lower ;
                upper-lower ];
        end
        
    end
end
