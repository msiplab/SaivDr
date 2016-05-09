classdef AbstOvsdLpPuFb3dTypeIISystem < ...
        saivdr.dictionary.nsoltx.AbstOvsdLpPuFb3dSystem %#codegen
    %ABSTOVSDLPPUFB3DTYPEIISYSTEM Abstract class 3-D Type-II OLPPUFB
    %
    % SVN identifier:
    % $Id: AbstOvsdLpPuFb3dTypeIISystem.m 690 2015-06-09 09:37:49Z sho $
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

    properties (Access = protected)
        matrixE0
        mexFcn
    end

    properties (Access = protected,PositiveInteger)
        nStages
    end

    methods (Access = protected, Static = true, Abstract = true)
        value = getDefaultPolyPhaseOrder_()
    end

    methods
        function obj = AbstOvsdLpPuFb3dTypeIISystem(varargin)
            obj = obj@saivdr.dictionary.nsoltx.AbstOvsdLpPuFb3dSystem(...
                varargin{:});
            updateProperties_(obj);
            updateAngles_(obj);
            updateMus_(obj);
        end
    end

    methods (Access = protected)

        function s = saveObjectImpl(obj)
            s = saveObjectImpl@saivdr.dictionary.nsoltx.AbstOvsdLpPuFb3dSystem(obj);
            s.nStages = obj.nStages;
            s.matrixE0 = obj.matrixE0;
            s.mexFcn   = obj.mexFcn;
        end

        function loadObjectImpl(obj,s,wasLocked)
            obj.mexFcn  = s.mexFcn;
            obj.nStages = s.nStages;
            %TODO: 読み込みデータが複素対応したら修正
            [p_,~] = s.matrixE0;
            obj.matrixE0 = blkdiag(eye(floor(p_/2)),1i*eye(floor(p_/2)),1)*s.matrixE0;
            loadObjectImpl@saivdr.dictionary.nsoltx.AbstOvsdLpPuFb3dSystem(obj,s,wasLocked);
        end

        function resetImpl(obj)
            resetImpl@saivdr.dictionary.nsoltx.AbstOvsdLpPuFb3dSystem(obj);
            % Prepare MEX function
            import saivdr.dictionary.nsoltx.ChannelGroup
            import saivdr.dictionary.nsoltx.mexsrcs.fcn_autobuild_bb_type2
            %TODO: 引数を修正
            [obj.mexFcn, obj.mexFlag] = fcn_autobuild_bb_type2(...
                obj.NumberOfChannels(ChannelGroup.UPPER),...
                obj.NumberOfChannels(ChannelGroup.LOWER));
        end

        function setupImpl(obj,varargin)
            % Prepare MEX function
            import saivdr.dictionary.nsoltx.ChannelGroup
            import saivdr.dictionary.nsoltx.mexsrcs.fcn_autobuild_bb_type2
            %TODO: 引数を修正
            [obj.mexFcn, obj.mexFlag] = fcn_autobuild_bb_type2(...
                obj.NumberOfChannels(ChannelGroup.UPPER),...
                obj.NumberOfChannels(ChannelGroup.LOWER));
        end

        function updateProperties_(obj)
            import saivdr.dictionary.nsoltx.ChannelGroup
            import saivdr.dictionary.utility.Direction
            import saivdr.dictionary.utility.ParameterMatrixSet

            % Check DecimationFactor
            if length(obj.DecimationFactor) > 3
                error('Dimension of DecimationFactor must be less than or equal to three.');
            end
            nHalfDecs = prod(obj.DecimationFactor)/2;

            % Check PolyPhaseOrder
            if isempty(obj.PolyPhaseOrder)
                obj.PolyPhaseOrder = obj.getDefaultPolyPhaseOrder_();
            end
            if length(obj.PolyPhaseOrder) > 3
                error('Dimension of PolyPhaseOrder must be less than or equal to three.');
            end
            ordX = obj.PolyPhaseOrder(Direction.HORIZONTAL);
            ordY = obj.PolyPhaseOrder(Direction.VERTICAL);
            ordZ = obj.PolyPhaseOrder(Direction.DEPTH);
            if mod(ordX,2)~=0 || mod(ordY,2)~=0 || mod(ordZ,2)~=0
                error('Polyphase orders must be even.');
            end
            obj.nStages = uint32(1+double(ordX)/2+double(ordY)/2+double(ordZ)/2);
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
                %{
            elseif obj.NumberOfChannels(ChannelGroup.UPPER) <= ...
                    obj.NumberOfChannels(ChannelGroup.LOWER)
                id = 'SaivDr:IllegalArgumentException';
                msg = 'ps must be greater than pa. (not yet supported).';
                me = MException(id, msg);
                throw(me);
                %}
            elseif (obj.NumberOfChannels(ChannelGroup.UPPER) < ceil(nHalfDecs)) ||...
                    (obj.NumberOfChannels(ChannelGroup.LOWER) < floor(nHalfDecs))
                id = 'SaivDr:IllegalArgumentException';
                msg = 'Both of ps and pa must be greater than a half of #Decs.';
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

%             % Prepare MEX function
%             if ~obj.mexFlag
%                 import saivdr.dictionary.nsoltx.mexsrcs.fcn_autobuild_bb_type2
%                 [obj.mexFcn, obj.mexFlag] = fcn_autobuild_bb_type2(...
%                     obj.NumberOfChannels(ChannelGroup.UPPER),...
%                     obj.NumberOfChannels(ChannelGroup.LOWER));
%             end

        end

        function updateAngles_(obj)
            import saivdr.dictionary.nsoltx.ChannelGroup
            nAngsPerStg = zeros(2,1);
            %
            nChU = obj.NumberOfChannels(ChannelGroup.UPPER);
            nChL = obj.NumberOfChannels(ChannelGroup.LOWER);
            nAngsPerStg(ChannelGroup.UPPER) = ...
                nChU*double(nChU-1)/2;
            nAngsPerStg(ChannelGroup.LOWER) = ...
                nChL*double(nChL-1)/2;
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
            import saivdr.dictionary.nsoltx.ChannelGroup
            %
            nChU = obj.NumberOfChannels(ChannelGroup.UPPER);
            nChL = obj.NumberOfChannels(ChannelGroup.LOWER);
            sizeOfMus = [ (nChU+nChL) obj.nStages ];
            %
            if isscalar(obj.Mus) && obj.Mus == 1
                if nChU > nChL
                    obj.Mus = [
                        ones(nChU, obj.nStages);
                        -ones(nChL, obj.nStages) ];
                else
                    obj.Mus = [
                        -ones(nChU, obj.nStages);
                        ones(nChL, obj.nStages) ];
                end
                if mod(obj.nStages,2) == 1
                    obj.Mus(:,1) = ones(size(obj.Mus,1),1);
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
            import saivdr.dictionary.utility.Direction
            import saivdr.dictionary.nsoltx.ChannelGroup
            import saivdr.dictionary.nsoltx.AbstOvsdLpPuFb3dTypeIISystem
            %import saivdr.dictionary.nsoltx.mexsrcs.*

            %
            nChs = obj.NumberOfChannels;
            dec  = obj.DecimationFactor;
            decX = dec(Direction.HORIZONTAL);
            decY = dec(Direction.VERTICAL);
            decZ = dec(Direction.DEPTH);
            nHalfDecs = prod(dec)/2;
            ordX = obj.PolyPhaseOrder(Direction.HORIZONTAL);
            ordY = obj.PolyPhaseOrder(Direction.VERTICAL);
            ordZ = obj.PolyPhaseOrder(Direction.DEPTH);
            %
            E0 = obj.matrixE0;
            %
            %TODO:begin
            cM_2 = ceil(nHalfDecs);
            pmMtxSt_ = obj.ParameterMatrixSet;
            W = step(pmMtxSt_,[],uint32(1))*[
                eye(cM_2) ;
                zeros(nChs(ChannelGroup.UPPER)-cM_2,cM_2)];
            fM_2 = floor(nHalfDecs);
            %
            U = step(pmMtxSt_,[],uint32(2))*[
                eye(fM_2);
                zeros(nChs(ChannelGroup.LOWER)-fM_2,fM_2) ];
            R = blkdiag(W,U);
            %TODO:end
            %R = step(pmMtxSt_,[],uint32(1))*[eye(floor(nHalfDecs)),zeros(sum(nChs)-floor(nHalfDecs),1)]
            E = R*E0;
            iParamMtx = uint32(3);
            %iParamMtx = uint32(2);

            % Depth extention
            lenY = decY;
            lenX = decX;
            nShift = int32(lenY*(decZ*lenX));
            for iOrdZ = 1:uint32(double(ordZ)/2)
                %hW = step(pmMtxSt_,[],iParamMtx);
                hW = eye(ceil(nHalfDecs));
                %hU = step(pmMtxSt_,[],iParamMtx+1);
                hU = eye(ceil(nHalfDecs));
                %angles2 = step(pmMtxSt_,[],iParamMtx+2);
                angles2 = pi/4*ones(floor(nHalfDecs/2),1);
                %W = step(pmMtxSt_,[],iParamMtx+3);
                W = step(pmMtxSt_,[],iParamMtx);
                %U = step(pmMtxSt_,[],iParamMtx+4);
                U = step(pmMtxSt_,[],iParamMtx+1);
                %angles1 = step(pmMtxSt_,[],iParamMtx+5);
                angles1 = pi/4*ones(floor(nHalfDecs/2),1);
                if obj.mexFlag
                    %TODO:
                    E = obj.mexFcn(E, hW, hU, angles2, W, U, angles1, floor(hChs), nShift);
                else
                    import saivdr.dictionary.nsoltx.mexsrcs.Order2BuildingBlockTypeII
                    hObb = Order2BuildingBlockTypeII();
                    E = step(hObb, E, hW, hU, angles2, W, U, angles1, floor(hChs), nShift);
                end
                iParamMtx = iParamMtx+2;
                %iParamMtx = iParamMtx+6;
            end
            lenZ = decZ*(ordZ+1);

            % Horizontal extention
            E = permuteCoefs_(obj,E,lenY*lenX); % Y X Z -> Z Y X
            nShift = int32(lenZ*(decX*lenY));
            for iOrdX = 1:uint32(double(ordX)/2)
                %hW = step(pmMtxSt_,[],iParamMtx);
                hW = eye(ceil(nHalfDecs));
                %hU = step(pmMtxSt_,[],iParamMtx+1);
                hU = eye(ceil(nHalfDecs));
                %angles2 = step(pmMtxSt_,[],iParamMtx+2);
                angles2 = pi/4*ones(floor(nHalfDecs/2),1);
                %W = step(pmMtxSt_,[],iParamMtx+3);
                W = step(pmMtxSt_,[],iParamMtx);
                %U = step(pmMtxSt_,[],iParamMtx+4);
                U = step(pmMtxSt_,[],iParamMtx+1);
                %angles1 = step(pmMtxSt_,[],iParamMtx+5);
                angles1 = pi/4*ones(floor(nHalfDecs/2),1);
                if obj.mexFlag
                    %TODO:
                    E = obj.mexFcn(E, hW, hU, angles2, W, U, angles1, floor(hChs), nShift);
                else
                    import saivdr.dictionary.nsoltx.mexsrcs.Order2BuildingBlockTypeII
                    hObb = Order2BuildingBlockTypeII();
                    E = step(hObb, E, hW, hU, angles2, W, U, angles1, floor(hChs), nShift);
                end
                iParamMtx = iParamMtx+2;
                %iParamMtx = iParamMtx+6;
            end
            lenX = decX*(ordX+1);

            % Vertical extention
            E = permuteCoefs_(obj,E,lenZ*lenY); % Z Y X -> X Z Y
            nShift = int32(lenX*(decY*lenZ));
            for iOrdY = 1:uint32(double(ordY)/2)
                %hW = step(pmMtxSt_,[],iParamMtx);
                hW = eye(ceil(nHalfDecs));
                %hU = step(pmMtxSt_,[],iParamMtx+1);
                hU = eye(ceil(nHalfDecs));
                %angles2 = step(pmMtxSt_,[],iParamMtx+2);
                angles2 = pi/4*ones(floor(nHalfDecs/2),1);
                %W = step(pmMtxSt_,[],iParamMtx+3);
                W = step(pmMtxSt_,[],iParamMtx);
                %U = step(pmMtxSt_,[],iParamMtx+4);
                U = step(pmMtxSt_,[],iParamMtx+1);
                %angles1 = step(pmMtxSt_,[],iParamMtx+5);
                angles1 = pi/4*ones(floor(nHalfDecs/2),1);
                if obj.mexFlag
                    %TODO:
                    E = obj.mexFcn(E, hW, hU, angles2, W, U, angles1, floor(hChs), nShift);
                else
                    import saivdr.dictionary.nsoltx.mexsrcs.Order2BuildingBlockTypeII
                    hObb = Order2BuildingBlockTypeII();
                    E = step(hObb, E, hW, hU, angles2, W, U, angles1, floor(hChs), nShift);
                end
                iParamMtx = iParamMtx+2;
                %iParamMtx = iParamMtx+6;
            end
            lenY = decY*(ordY+1);

            %
            E = permuteCoefs_(obj,E,lenX*lenZ); % X Z Y -> Y X Z
            nSubbands = size(E,1);
            value = zeros(lenY,lenX,lenZ,nSubbands);
            for iSubband = 1:nSubbands
                value(:,:,:,iSubband) = reshape(E(iSubband,:),lenY,lenX,lenZ);
            end

        end

    end

end
