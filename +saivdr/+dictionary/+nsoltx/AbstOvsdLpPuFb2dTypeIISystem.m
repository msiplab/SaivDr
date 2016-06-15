classdef AbstOvsdLpPuFb2dTypeIISystem < ...
        saivdr.dictionary.nsoltx.AbstOvsdLpPuFb2dSystem %#codegen
    %AbstOvsdLpPuFb2dTypeIISystem Abstract class 2-D Type-II OLPPUFB
    %
    % SVN identifier:
    % $Id: AbstOvsdLpPuFb2dTypeIISystem.m 683 2015-05-29 08:22:13Z sho $
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
        function obj = AbstOvsdLpPuFb2dTypeIISystem(varargin)
            obj = obj@saivdr.dictionary.nsoltx.AbstOvsdLpPuFb2dSystem(...
                varargin{:});
            updateProperties_(obj);
            updateAngles_(obj);
            updateMus_(obj);
        end
    end

    methods (Access = protected)

        function s = saveObjectImpl(obj)
            s = saveObjectImpl@saivdr.dictionary.nsoltx.AbstOvsdLpPuFb2dSystem(obj);
            s.nStages  = obj.nStages;
            s.matrixE0 = obj.matrixE0;
            s.mexFcn   = obj.mexFcn;
        end

        function loadObjectImpl(obj,s,wasLocked)
            obj.mexFcn   = s.mexFcn;
            obj.nStages  = s.nStages;
            obj.matrixE0 = s.matrixE0;
            loadObjectImpl@saivdr.dictionary.nsoltx.AbstOvsdLpPuFb2dSystem(obj,s,wasLocked);
        end

        function resetImpl(obj)
            resetImpl@saivdr.dictionary.nsoltx.AbstOvsdLpPuFb2dSystem(obj);
            % Build MEX
            import saivdr.dictionary.nsoltx.mexsrcs.fcn_autobuild_bb_type2
            import saivdr.dictionary.nsoltx.ChannelGroup
            [obj.mexFcn, obj.mexFlag] = fcn_autobuild_bb_type2(...
                floor(obj.NumberOfChannels/2));
        end

        function setupImpl(obj,varargin)
            % Prepare MEX function
            import saivdr.dictionary.nsoltx.ChannelGroup
            import saivdr.dictionary.nsoltx.mexsrcs.fcn_autobuild_bb_type2
            [obj.mexFcn, obj.mexFlag] = fcn_autobuild_bb_type2(...
                floor(obj.NumberOfChannels/2));
        end

        function updateProperties_(obj)
            import saivdr.dictionary.nsoltx.ChannelGroup
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
                obj.NumberOfChannels = 2*floor(nHalfDecs)+1;
            elseif isvector(obj.NumberOfChannels)
                obj.NumberOfChannels = sum(obj.NumberOfChannels);
                % TODO: —áŠOˆ—‚ð³‚µ‚­ŽÀ‘•‚·‚é
                if mod(obj.NumberOfChannels,2) == 0
                    id = 'SaivDr:IllegalArgumentException';
                    msg = '#Channels must be odd.';
                    me = MException(id, msg);
                    throw(me);
                end
            end

            % Prepare ParameterMatrixSet
%             paramMtxSizeTab = repmat(...
%                 [ obj.NumberOfChannels(ChannelGroup.UPPER) ;
%                 obj.NumberOfChannels(ChannelGroup.LOWER) ],...
%                 obj.nStages,2);
            paramMtxSizeTab = [obj.NumberOfChannels*ones(1,2);
                repmat([floor(obj.NumberOfChannels/2)*ones(2,2);
                floor(obj.NumberOfChannels/4),1;
                ceil(obj.NumberOfChannels/2)*ones(2,2);
                floor(obj.NumberOfChannels/4),1], obj.nStages-1, 1)];
            obj.ParameterMatrixSet = ParameterMatrixSet(...
                'MatrixSizeTable',paramMtxSizeTab);

            % Prepare MEX function
            if ~obj.mexFlag
                import saivdr.dictionary.nsoltx.mexsrcs.fcn_autobuild_bb_type2
                [obj.mexFcn, obj.mexFlag] = fcn_autobuild_bb_type2(...
                    floor(obj.NumberOfChannels/2));
            end

        end

        function updateAngles_(obj)
            import saivdr.dictionary.nsoltx.ChannelGroup
            nAngsPerStg = zeros(3,1);
            %
            nAngsPerStg(1) = ...
                floor(obj.NumberOfChannels/2) ...
                *double(floor(obj.NumberOfChannels/2)-1);
            nAngsPerStg(2) = ...
                ceil(obj.NumberOfChannels/2) ...
                *double(ceil(obj.NumberOfChannels/2)-1);
            nAngsPerStg(3) = 2*floor(obj.NumberOfChannels/4);
            nAngsInit = obj.NumberOfChannels*(obj.NumberOfChannels-1)/2;
            sizeOfAngles = nAngsInit + sum(nAngsPerStg)*(obj.nStages-1);
            %

            if isscalar(obj.Angles) && obj.Angles == 0
                obj.Angles = zeros(sizeOfAngles,1);
            end
            obj.Angles = obj.Angles(:);
            % TODO : —áŠOˆ—
%             if size(obj.Angles,1) ~= sizeOfAngles(1) || ...
%                     size(obj.Angles,2) ~= sizeOfAngles(2)
            if size(obj.Angles) ~= sizeOfAngles
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
%             sizeOfMus = [ 2*sum(obj.NumberOfChannels) obj.nStages ];
            sizeOfMus = obj.NumberOfChannels*(2*obj.nStages-1);
            %
%             nChL = floor(obj.NumberOfChannels/2);
%             nChU = ceil(obj.NumberOfChannels/2);
%             if isscalar(obj.Mus) && obj.Mus == 1
                %TODO:obj.Mus‚ð“KØ‚ÉÝ’è‚·‚é
%                 if nChU > nChL
%                     obj.Mus = repmat([
%                         ones(nChU, obj.nStages);
%                         -ones(nChL, obj.nStages) ],2,1);
%                 else
%                     obj.Mus = repmat([
%                         -ones(nChU, obj.nStages);
%                         ones(nChL, obj.nStages) ],2,1);
%                 end
%                 if mod(obj.nStages,2) == 1
%                     obj.Mus(:,1) = ones(size(obj.Mus,1),1);
%                 end
%                 sizeOfMus = prod(sizeOfMus);
%             end
            obj.Mus = ones(sizeOfMus,1);
%             if size(obj.Mus,1) ~= sizeOfMus(1) || ...
%                     size(obj.Mus,2) ~= sizeOfMus(2)
            if size(obj.Mus) ~= sizeOfMus
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
            import saivdr.dictionary.nsoltx.AbstOvsdLpPuFb2dTypeIISystem
            import saivdr.dictionary.nsoltx.mexsrcs.*

            %
            nChs = obj.NumberOfChannels;
            dec  = obj.DecimationFactor;
            decX = dec(Direction.HORIZONTAL);
            decY = dec(Direction.VERTICAL);
            ordX = obj.PolyPhaseOrder(Direction.HORIZONTAL);
            ordY = obj.PolyPhaseOrder(Direction.VERTICAL);
            pmMtxSt_ = obj.ParameterMatrixSet;
            mexFcn_  = obj.mexFcn;
            mexFlag_ = obj.mexFlag;
            %
            E0 = obj.matrixE0;
            %
            V0 = step(pmMtxSt_,[],uint32(1));
            E = V0*[ E0 ; zeros(nChs-prod(dec),prod(dec))];
            iParamMtx = uint32(2);

            % Horizontal extention
            lenY = decY;
            lenX = decX;
            nShift = int32(lenY*lenX);
            for iOrdX = 1:uint32(double(ordX)/2)
                W = step(pmMtxSt_,[],iParamMtx);
                U = step(pmMtxSt_,[],iParamMtx+1);
                angsB1 = step(pmMtxSt_,[],iParamMtx+2);
                hW = step(pmMtxSt_,[],iParamMtx+3);
                hU = step(pmMtxSt_,[],iParamMtx+4);
                angsB2 = step(pmMtxSt_,[],iParamMtx+5);
                if mexFlag_
                    E = mexFcn_(E, W, U, angsB1, hW, hU, angsB2, floor(nChs/2), nShift);
                else
                    import saivdr.dictionary.nsoltx.mexsrcs.Order2BuildingBlockTypeII
                    hObb = Order2BuildingBlockTypeII();
                    E = step(hObb, E, W, U, angsB1, hW, hU, angsB2, floor(nChs/2), nShift);
                end
                iParamMtx = iParamMtx+6;
            end
           lenX = decX*(ordX+1);

            % Vertical extention
            if ordY > 0
                E = permuteCoefs_(obj,E,lenY);
                nShift = int32(lenX*lenY);
                for iOrdY = 1:uint32(double(ordY)/2)
                    W = step(pmMtxSt_,[],iParamMtx);
                    U = step(pmMtxSt_,[],iParamMtx+1);
                    angsB1 = step(pmMtxSt_,[],iParamMtx+2);
                    hW = step(pmMtxSt_,[],iParamMtx+3);
                    hU = step(pmMtxSt_,[],iParamMtx+4);
                    angsB2 = step(pmMtxSt_,[],iParamMtx+5);
                    if mexFlag_
                        E = mexFcn_(E, W, U, angsB1, hW, hU, angsB2, floor(nChs/2), nShift);
                    else
                        import saivdr.dictionary.nsoltx.mexsrcs.Order2BuildingBlockTypeII
                        hObb = Order2BuildingBlockTypeII();
                        E = step(hObb, E, W, U, angsB1, hW, hU, angsB2, floor(nChs/2), nShift);
                    end
                    iParamMtx = iParamMtx+6;
                end
                lenY = decY*(ordY+1);

                E = ipermuteCoefs_(obj,E,lenY);
            end
            %
            nSubbands = size(E,1);
            value = zeros(lenY,lenX,nSubbands);
            for iSubband = 1:nSubbands
                value(:,:,iSubband) = reshape(E(iSubband,:),lenY,lenX);
            end

        end


    end

end
