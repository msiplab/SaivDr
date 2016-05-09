classdef AbstOvsdLpPuFb3dTypeISystem < ...
        saivdr.dictionary.nsoltx.AbstOvsdLpPuFb3dSystem %#codegen
    %ABSTOVSDLPPUFB3DTYPEISYSTEM Abstract class 3-D Type-I OLPPUFB
    %
    % SVN identifier:
    % $Id: AbstOvsdLpPuFb3dTypeISystem.m 690 2015-06-09 09:37:49Z sho $
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
        function obj = AbstOvsdLpPuFb3dTypeISystem(varargin)
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
            obj.mexFcn   = s.mexFcn;
            obj.nStages = s.nStages;
            %TODO: 読み込みデータが複素対応したら修正
            [p_,~] = s.matrixE0;
            obj.matrixE0 = blkdiag(eye(p_/2),1i*eye(p_/2))*s.matrixE0;
            loadObjectImpl@saivdr.dictionary.nsoltx.AbstOvsdLpPuFb3dSystem(obj,s,wasLocked);
        end

        function resetImpl(obj)
            resetImpl@saivdr.dictionary.nsoltx.AbstOvsdLpPuFb3dSystem(obj);
            % Prepare MEX function
            import saivdr.dictionary.nsoltx.mexsrcs.fcn_autobuild_bb_type1
            [obj.mexFcn, obj.mexFlag] = fcn_autobuild_bb_type1(obj.NumberOfChannels(1));
        end

        function setupImpl(obj,varargin)
            % Prepare MEX function
            import saivdr.dictionary.nsoltx.mexsrcs.fcn_autobuild_bb_type1
            [obj.mexFcn, obj.mexFlag] = fcn_autobuild_bb_type1(obj.NumberOfChannels(1));
        end

        function updateProperties_(obj)
            import saivdr.dictionary.nsoltx.ChannelGroup
            import saivdr.dictionary.utility.Direction
            import saivdr.dictionary.utility.ParameterMatrixSet
           % import saivdr.dictionary.nsoltx.AbstOvsdLpPuFb3dTypeISystem

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
            obj.nStages = uint32(1+ordX+ordY+ordZ);
            obj.matrixE0 = getMatrixE0_(obj);

            % Check NumberOfChannels
            if length(obj.NumberOfChannels) > 2
                error('Dimension of NumberOfChannels must be less than or equal to two.');
            end
            id = 'SaivDr:IllegalArgumentException';
            if length(obj.NumberOfChannels) == 2 && ...
                (obj.NumberOfChannels(1) < nHalfDecs || ...
                    obj.NumberOfChannels(2) < nHalfDecs)
                msg = 'Both of NumberOfSymmetricChannels and NumberOfAntisymmetric channels shoud be greater than or equal to prod(DecimationFactor)/2.';
                me = MException(id, msg);
                throw(me);
            end
            if isempty(obj.NumberOfChannels)
                obj.NumberOfChannels = nHalfDecs * [ 1 1 ];
            elseif isscalar(obj.NumberOfChannels)
                if mod(obj.NumberOfChannels,2) ~= 0
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

            % Prepare MEX function
%            if ~obj.mexFlag
%                import saivdr.dictionary.nsoltx.mexsrcs.fcn_autobuild_bb_type1
%                [obj.mexFcn, obj.mexFlag] = fcn_autobuild_bb_type1(obj.NumberOfChannels(1));
%            end
        end

        function updateAngles_(obj)
            import saivdr.dictionary.nsoltx.ChannelGroup
            import saivdr.dictionary.utility.Direction
            nChL = obj.NumberOfChannels(ChannelGroup.LOWER);
            nAngsPerStg = nChL*(nChL-1)/2;
            sizeOfAngles = [nAngsPerStg obj.nStages+1];
            if isscalar(obj.Angles) && obj.Angles==0
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
            import saivdr.dictionary.utility.Direction
            %
            nChL = obj.NumberOfChannels(ChannelGroup.LOWER);
            sizeOfMus = [ nChL obj.nStages+1 ];
            %
            if isscalar(obj.Mus) && obj.Mus==1
                obj.Mus = -ones(sizeOfMus);
                obj.Mus(:,1:2) = ones(size(obj.Mus,1),2);
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
            import saivdr.dictionary.nsoltx.ChannelGroup
            import saivdr.dictionary.utility.Direction
            import saivdr.dictionary.nsoltx.AbstOvsdLpPuFb3dTypeISystem
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
            W = step(pmMtxSt_,[],uint32(1))*[ eye(cM_2) ;
                zeros(nChs(ChannelGroup.LOWER)-cM_2,cM_2)];
            fM_2 = floor(nHalfDecs);
            U = step(pmMtxSt_,[],uint32(2))*[ eye(fM_2);
                zeros(nChs(ChannelGroup.LOWER)-fM_2,fM_2) ];
            R = blkdiag(W,U);
            %TODO:end
            %R = step(pmMtxSt_,[],uint32(1))*[eye(nChs);zeros(nChs-dec,prod(dec))];
            E = R*E0;
            iParamMtx = uint32(3);
            %iParamMtx = uint32(2);
            hChs = nChs(1);

            % Depth extention
            lenY = decY;
            lenX = decX;
            nShift = int32(lenY*(decZ*lenX));
            for iOrdZ = 1:ordZ
                %W = step(pmMtxSt_,[],iParamMtx);
                W = eye(hChs);
                U = step(pmMtxSt_,[],iParamMtx);
                %angles = step(pmMtxSt_,[],iParamMtx);
                angles = pi/4*ones(floor(hChs/2),1);
                if obj.mexFlag
                    %TODO:mexFcnの修正
                    E = obj.mexFcn(E, W, U, angles, hChs, nShift);
                else
                    import saivdr.dictionary.nsoltx.mexsrcs.Order1BuildingBlockTypeI
                    hObb = Order1BuildingBlockTypeI();
                    E = step(hObb, E, W, U, angles, hChs, nShift);
                end
                iParamMtx = iParamMtx+1;
                %iParamMtx = iParamMtx+3;
            end
            lenZ = decZ*(ordZ+1);

            % Horizontal extention
            E = permuteCoefs_(obj,E,lenY*lenX); % Y X Z -> Z Y X
            nShift = int32(lenZ*(decX*lenY));
            for iOrdX = 1:ordX
                %W = step(pmMtxSt_,[],iParamMtx);
                W = eye(hChs);
                %U = step(pmMtxSt_,[],iParamMtx+1);
                U = step(pmMtxSt_,[],iParamMtx);
                %angles = step(pmMtxSt_,[],iParamMtx+2);
                angles = pi/4*ones(floor(hChs/2),1);
                if obj.mexFlag
                    %TODO:
                    E = obj.mexFcn(E, W, U, angles, hChs, nShift);
                else
                    import saivdr.dictionary.nsoltx.mexsrcs.Order1BuildingBlockTypeI
                    hObb = Order1BuildingBlockTypeI();
                    E = step(hObb, E, W, U, angles, hChs, nShift);
                end
                %iParamMtx = iParamMtx+1;
                %iParamMtx = iParamMtx+3;
            end
            lenX = decX*(ordX+1);

            % Vertical extention
            E = permuteCoefs_(obj,E,lenZ*lenY); % Z Y X -> X Z Y
            nShift = int32(lenX*(decY*lenZ));
            for iOrdY = 1:ordY
                %W = step(pmMtxSt_,[],iParamMtx);
                W = eye(hChs);
                %U = step(pmMtxSt_,[],iParamMtx+1);
                U = step(pmMtxSt_,[],iParamMtx);
                %angles = step(pmMtxSt_,[],iParamMtx+2);
                angles = pi/4*ones(floor(hChs/2),1);
                if obj.mexFlag
                    E = obj.mexFcn(E, W, U, angles, hChs, nShift);
                else
                    import saivdr.dictionary.nsoltx.mexsrcs.Order1BuildingBlockTypeI
                    hObb = Order1BuildingBlockTypeI();
                    E = step(hObb, E, W, U, angles, hChs, nShift);
                end
                iParamMtx = iParamMtx+1;
                %iParamMtx = iParamMtx+3;
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
