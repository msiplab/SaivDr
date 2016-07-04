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
            updateSymmetry_(obj);
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
            [obj.mexFcn, obj.mexFlag] = fcn_autobuild_bb_type1(obj.NumberOfChannels/2);
        end

        function setupImpl(obj,varargin)
            % Prepare MEX function
            import saivdr.dictionary.nsoltx.mexsrcs.fcn_autobuild_bb_type1
            [obj.mexFcn, obj.mexFlag] = fcn_autobuild_bb_type1(obj.NumberOfChannels/2);
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
                obj.NumberOfChannels = 2*ceil(nHalfDecs);
            elseif isvector(obj.NumberOfChannels)
                obj.NumberOfChannels = sum(obj.NumberOfChannels);
                if mod(obj.NumberOfChannels,2) ~= 0
                    msg = '#Channels must be even.';
                    me = MException(id, msg);
                    throw(me);
                end
            end

            % Prepare ParameterMatrixSet
            paramMtxSizeTab = [obj.NumberOfChannels*ones(1,2);
                repmat([obj.NumberOfChannels/2*ones(2,2);floor(obj.NumberOfChannels/4),1], obj.nStages-1, 1)];
            obj.ParameterMatrixSet = ParameterMatrixSet(...
                'MatrixSizeTable',paramMtxSizeTab);

        end

        function updateAngles_(obj)
            import saivdr.dictionary.nsoltx.ChannelGroup
            nChL = obj.NumberOfChannels/2;
            nAngsInitStg = obj.NumberOfChannels*(obj.NumberOfChannels-1)/2;
            nAngsPerStg = nChL*(nChL-1)+floor(nChL/2);
            sizeOfAngles = nAngsInitStg+nAngsPerStg*(obj.nStages-1);
            if isscalar(obj.Angles) && obj.Angles==0
                obj.Angles = zeros(sizeOfAngles,1);
            end
            obj.Angles = obj.Angles(:);
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
            nChL = obj.NumberOfChannels/2;
            %sizeOfMus = [ 2*nChL obj.nStages ];
            sizeOfMus = 2*nChL*obj.nStages;
            if isscalar(obj.Mus) && obj.Mus==1
                %TODO:obj.Mus‚ð“KØ‚É’è‚ß‚é
                obj.Mus = ones(sizeOfMus,1);
            end
            % TODO: —áŠOˆ—
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
            import saivdr.dictionary.nsoltx.ChannelGroup
            import saivdr.dictionary.olpprfb.AbstOvsdLpPuFb1dTypeISystem
            %import saivdr.dictionary.nsoltx.mexsrcs.*
            %
            nChs = obj.NumberOfChannels;
            dec  = obj.DecimationFactor;
            hChs = nChs/2;
            ord = obj.PolyPhaseOrder;
            pmMtxSet_  = obj.ParameterMatrixSet;
            mexFcn_ = obj.mexFcn;
            mexFlag_ = obj.mexFlag;
            %
            E0 = obj.matrixE0;
            %
            V0 = step(pmMtxSet_,[],uint32(1));
            E = V0*[ E0 ; zeros(nChs - dec, dec) ];
            iParamMtx = uint32(2);

            % Order extention
            if ord > 0
                nShift = int32(dec);
                for iOrd = 1:ord
                    W = step(pmMtxSet_,[],iParamMtx);
                    U = step(pmMtxSet_,[],iParamMtx+1);
                    angsB = step(pmMtxSet_,[],iParamMtx+2);
                    if mexFlag_
                        E = mexFcn_(E, W, U, angsB, hChs, nShift);
                    else
                        import saivdr.dictionary.nsoltx.mexsrcs.Order1BuildingBlockTypeI
                        hObb = Order1BuildingBlockTypeI();
                        E = step(hObb, E, W, U, angsB, hChs, nShift);
                    end
                    iParamMtx = iParamMtx+3;
                end
            end
            E = diag(obj.Symmetry)*E;
            value = E.';
        end

    end
end
