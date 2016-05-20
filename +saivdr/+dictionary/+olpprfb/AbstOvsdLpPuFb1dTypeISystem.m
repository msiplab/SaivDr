classdef AbstOvsdLpPuFb1dTypeISystem < ...
        saivdr.dictionary.olpprfb.AbstOvsdLpPuFb1dSystem %#codegen
    %ABSTOVSDLPPUFB1DTYPEISYSTEM Abstract class 2-D Type-I OLPPUFB
    %
    % SVN identifier:
    % $Id: AbstOvsdLpPuFb1dTypeISystem.m 690 2015-06-09 09:37:49Z sho $
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
        function obj = AbstOvsdLpPuFb1dTypeISystem(varargin)
            obj = obj@saivdr.dictionary.olpprfb.AbstOvsdLpPuFb1dSystem(...
                varargin{:});
            updateProperties_(obj);
            updateAngles_(obj);
            updateMus_(obj);
        end
    end

    methods (Access = protected)

        function s = saveObjectImpl(obj)
            s = saveObjectImpl@saivdr.dictionary.olpprfb.AbstOvsdLpPuFb1dSystem(obj);
            s.nStages = obj.nStages;
            s.matrixE0 = obj.matrixE0;
            s.mexFcn   = obj.mexFcn;
        end

        function loadObjectImpl(obj,s,wasLocked)
            obj.mexFcn   = s.mexFcn;
            obj.nStages = s.nStages;
            obj.matrixE0 = s.matrixE0;
            loadObjectImpl@saivdr.dictionary.olpprfb.AbstOvsdLpPuFb1dSystem(obj,s,wasLocked);
        end

        function resetImpl(obj)
            resetImpl@saivdr.dictionary.olpprfb.AbstOvsdLpPuFb1dSystem(obj);
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
            import saivdr.dictionary.utility.ParameterMatrixSet
           % import saivdr.dictionary.olpprfb.AbstOvsdLpPuFb1dTypeISystem

            % Check DecimationFactor
            if ~isscalar(obj.DecimationFactor)
                error('DecimationFactor must be a scaler.');
            end
            nHalfDecs = obj.DecimationFactor/2;

            % Check PolyPhaseOrder
            if isempty(obj.PolyPhaseOrder)
                obj.PolyPhaseOrder = obj.getDefaultPolyPhaseOrder_();
            end
            if ~isscalar(obj.PolyPhaseOrder)
                error('PolyPhaseOrder must be a scaler.');
            end
            ord = obj.PolyPhaseOrder;
            obj.nStages = uint32(1+ord);
            obj.matrixE0 = getMatrixE0_(obj);

            % Check NumberOfChannels
            if length(obj.NumberOfChannels) > 2
                error('Dimension of NumberOfChannels must be less than or equal to two.');
            end
            id = 'SaivDr:IllegalArgumentException';
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
                msg = 'ps and pa must be the same as each other.';
                me = MException(id, msg);
                throw(me);
            end

            % Prepare ParameterMatrixSet
            %paramMtxSizeTab = ...
            %    obj.NumberOfChannels(ChannelGroup.LOWER)*ones(obj.nStages+1,2);
            paramMtxSizeTab = repmat(obj.NumberOfChannels(ChannelGroup.LOWER)*ones(2,2), obj.nStages, 1);
            obj.ParameterMatrixSet = ParameterMatrixSet(...
                'MatrixSizeTable',paramMtxSizeTab);

        end

        function updateAngles_(obj)
            import saivdr.dictionary.nsoltx.ChannelGroup
            nChL = obj.NumberOfChannels(ChannelGroup.LOWER);
            nAngsPerStg = nChL*(nChL-1)/2;
            sizeOfAngles = [2*nAngsPerStg obj.nStages];
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
            nChL = obj.NumberOfChannels(ChannelGroup.LOWER);
            sizeOfMus = [ 2*nChL obj.nStages ];
            if isscalar(obj.Mus) && obj.Mus==1
                obj.Mus = -ones(sizeOfMus);
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
            import saivdr.dictionary.nsoltx.ChannelGroup
            import saivdr.dictionary.olpprfb.AbstOvsdLpPuFb1dTypeISystem
            %import saivdr.dictionary.nsoltx.mexsrcs.*
            %
            nChs = obj.NumberOfChannels;
            dec  = obj.DecimationFactor;
            nHalfDecs = dec/2;
            ord = obj.PolyPhaseOrder;
            pmMtxSet_  = obj.ParameterMatrixSet;
            mexFcn_ = obj.mexFcn;
            mexFlag_ = obj.mexFlag;
            %
            E0 = obj.matrixE0;
            %
            cM_2 = ceil(nHalfDecs);
            W = step(pmMtxSet_,[],uint32(1))*[ eye(cM_2) ;
                zeros(nChs(ChannelGroup.LOWER)-cM_2,cM_2)];
            fM_2 = floor(nHalfDecs);
            U = step(pmMtxSet_,[],uint32(2))*[ eye(fM_2);
                zeros(nChs(ChannelGroup.LOWER)-fM_2,fM_2) ];
            R = blkdiag(W,U);
            E = R*E0;
            iParamMtx = uint32(3);
            hChs = nChs(1);

            % Order extention
            if ord > 0
                nShift = int32(dec);
                for iOrd = 1:ord
                    %W = eye(hChs);
                    W = step(pmMtxSet_,[],iParamMtx);
                    %U = step(pmMtxSet_,[],iParamMtx);
                    U = step(pmMtxSet_,[],iParamMtx+1);
                    angles = pi/4*ones(floor(hChs/2),1);
                    %angles = step(pmMtxSet_,[],iParamMt+2);
                    if mexFlag_
                        E = mexFcn_(E, W, U, angles, hChs, nShift);
                    else
                        import saivdr.dictionary.nsoltx.mexsrcs.Order1BuildingBlockTypeI
                        hObb = Order1BuildingBlockTypeI();
                        E = step(hObb, E, W, U, angles, hChs, nShift);
                    end
                    %iParamMtx = iParamMtx+1;
                    iParamMtx = iParamMtx+2;
                end
            end
            %E = blkdiag(eye(hChs),-1i*eye(hChs))*E;
            value = E.';
        end

    end
end
