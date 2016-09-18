classdef AbstOvsdLpPuFb2dTypeISystem < ...
        saivdr.dictionary.nsoltx.AbstOvsdLpPuFb2dSystem %#codegen
    %ABSTOVSDLPPUFB2DTYPEISYSTEM Abstract class 2-D Type-I OLPPUFB
    %
    % SVN identifier:
    % $Id: AbstOvsdLpPuFb2dTypeISystem.m 690 2015-06-09 09:37:49Z sho $
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
        function obj = AbstOvsdLpPuFb2dTypeISystem(varargin)
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
            s.nStages = obj.nStages;
            s.matrixE0 = obj.matrixE0;
            s.mexFcn   = obj.mexFcn;
        end

        function loadObjectImpl(obj,s,wasLocked)
            obj.mexFcn   = s.mexFcn;
            obj.nStages = s.nStages;
            obj.matrixE0 = s.matrixE0;
            loadObjectImpl@saivdr.dictionary.nsoltx.AbstOvsdLpPuFb2dSystem(obj,s,wasLocked);
        end

        function resetImpl(obj)
            resetImpl@saivdr.dictionary.nsoltx.AbstOvsdLpPuFb2dSystem(obj);
            % Prepare MEX function
            import saivdr.dictionary.nsoltx.mexsrcs.fcn_autobuild_bb_type1
            [obj.mexFcn, obj.mexFlag] = fcn_autobuild_bb_type1(floor(obj.NumberOfChannels/2));
        end

        function setupImpl(obj,varargin)
            % Prepare MEX function
            import saivdr.dictionary.nsoltx.mexsrcs.fcn_autobuild_bb_type1
            [obj.mexFcn, obj.mexFlag] = fcn_autobuild_bb_type1(floor(obj.NumberOfChannels/2));
        end

        function updateProperties_(obj)
            import saivdr.dictionary.nsoltx.ChannelGroup
            import saivdr.dictionary.utility.Direction
            import saivdr.dictionary.utility.ParameterMatrixSet
           % import saivdr.dictionary.nsoltx.AbstOvsdLpPuFb2dTypeISystem

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

%             % Check NumberOfChannels
%             if length(obj.NumberOfChannels) > 2
%                 error('Dimension of NumberOfChannels must be less than or equal to two.');
%             end
%             if isempty(obj.NumberOfChannels)
%                 obj.NumberOfChannels = 2*floor(nHalfDecs)+1;
%             elseif isvector(obj.NumberOfChannels)
%                 obj.NumberOfChannels = sum(obj.NumberOfChannels);
%                 % TODO: 例外処理を正しく実装する
%                 if mod(obj.NumberOfChannels,2) == 0
%                     id = 'SaivDr:IllegalArgumentException';
%                     msg = '#Channels must be odd.';
%                     me = MException(id, msg);
%                     throw(me);
%                 end
%             end
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
            symmetricMtxSizeTab = obj.NumberOfChannels*ones(1,2);
            initParamMtxSizeTab = obj.NumberOfChannels*ones(1,2);
            propParamMtxSizeTab = [...
                ceil(obj.NumberOfChannels/2)*ones(1,2);
                floor(obj.NumberOfChannels/2)*ones(1,2);
                floor(sum(obj.NumberOfChannels)/4),1 ];
            paramMtxSizeTab = [...
                symmetricMtxSizeTab;
                initParamMtxSizeTab;
                repmat(propParamMtxSizeTab,obj.nStages-1,1)];
            obj.ParameterMatrixSet = ParameterMatrixSet(...
                'MatrixSizeTable',paramMtxSizeTab);

        end

        function updateAngles_(obj)
            import saivdr.dictionary.nsoltx.ChannelGroup
            import saivdr.dictionary.utility.Direction
            nCh = obj.NumberOfChannels;
            nAngsPerStg = nCh*(nCh-2)/4+floor(nCh/4);
            nAngsInit = nCh*(nCh-1)/2;
            nAngsSym = nCh;
            sizeOfAngles = nAngsSym + nAngsInit + (obj.nStages-1)*nAngsPerStg;
            if isscalar(obj.Angles) && obj.Angles==0
                angsSym = zeros(nAngsSym,1);
                angsInit = zeros(nAngsInit,1);
                angsPerStg = zeros(nAngsPerStg,obj.nStages-1);
                angsPerStg(end-floor(nCh/4)+1:end,:) = pi/2*ones(floor(nCh/4),obj.nStages-1);
                obj.Angles = [angsSym ; angsInit; angsPerStg(:)];
                %obj.Angles = zeros(sizeOfAngles,1);
            end
%             if size(obj.Angles,1) ~= sizeOfAngles(1) || ...
%                     size(obj.Angles,2) ~= sizeOfAngles(2)
            if size(obj.Angles) ~= sizeOfAngles
                id = 'SaivDr:IllegalArgumentException';
                %TODO: エラーメッセージの設定
                msg = sprintf(...
                    'Size of angles must be [ %d %d ]',...
                    sizeOfAngles(1), sizeOfAngles(2));
                me = MException(id, msg);
                throw(me);
            end
        end

        %TODO:　修正する
        function updateMus_(obj)
            %import saivdr.dictionary.nsoltx.ChannelGroup
            import saivdr.dictionary.utility.Direction
            %nChL = obj.NumberOfChannels(ChannelGroup.LOWER);
            nCh = obj.NumberOfChannels;
            sizeOfMus = [ nCh obj.nStages];
            if isscalar(obj.Mus) && obj.Mus==1
%                 obj.Mus = -ones(sizeOfMus);
%                 obj.Mus(:,1:2) = ones(size(obj.Mus,1),2);
                obj.Mus = ones(sizeOfMus);
                tmp = -ones(floor(nCh/2),floor((obj.nStages-1)/2));
                obj.Mus(floor(nCh/2)+1:end,2:2:obj.nStages) = tmp;
                %mus = obj.Mus;
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
            import saivdr.dictionary.nsoltx.AbstOvsdLpPuFb2dTypeISystem
            %import saivdr.dictionary.nsoltx.mexsrcs.*
            %
            nChs = obj.NumberOfChannels;
            dec  = obj.DecimationFactor;
            decX = dec(Direction.HORIZONTAL);
            decY = dec(Direction.VERTICAL);
            ordX = obj.PolyPhaseOrder(Direction.HORIZONTAL);
            ordY = obj.PolyPhaseOrder(Direction.VERTICAL);
            pmMtxSet_  = obj.ParameterMatrixSet;
            mexFcn_ = obj.mexFcn;
            mexFlag_ = obj.mexFlag;
            
            S = step(pmMtxSet_,[],uint32(1));
            %
            E0 = obj.matrixE0;
            %
            V0 = step(pmMtxSet_,[],uint32(2));
            E = V0*[ E0 ; zeros(nChs-prod(dec),prod(dec))];
            
            iParamMtx = uint32(3);
            %TODO: iParamMtx = uint32(2);
            hChs = nChs/2;

            %
            initOrdX = 1;
            lenX = decX;
            initOrdY = 1;
            lenY = decY;

            % Horizontal extention
            nShift = int32(lenY*lenX);
            for iOrdX = initOrdX:ordX
                %TODO:
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
            lenX = decX*(ordX+1);

            % Vertical extention
            if ordY > 0
                E = permuteCoefs_(obj,E,lenY);
                nShift = int32(lenX*lenY);
                for iOrdY = initOrdY:ordY
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
                lenY = decY*(ordY+1);

                E = ipermuteCoefs_(obj,E,lenY);
            end
            
            E = S*E;
            %
            nSubbands = size(E,1);
            value = zeros(lenY,lenX,nSubbands);
            for iSubband = 1:nSubbands
                value(:,:,iSubband) = reshape(E(iSubband,:),lenY,lenX);
            end

        end

    end
end
