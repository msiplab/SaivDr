classdef AbstCplxOvsdLpPuFb1dTypeIISystem < ...
        saivdr.dictionary.colpprfb.AbstCplxOvsdLpPuFb1dSystem %#codegen
    %AbstCplxOvsdLpPuFb1dTypeIISystem Abstract class 2-D Type-II OLPPUFB
    %
    % SVN identifier:
    % $Id: AbstCplxOvsdLpPuFb1dTypeIISystem.m 653 2015-02-04 05:21:08Z sho $
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
        function obj = AbstCplxOvsdLpPuFb1dTypeIISystem(varargin)
            obj = obj@saivdr.dictionary.colpprfb.AbstCplxOvsdLpPuFb1dSystem(...
                varargin{:});
            updateProperties_(obj);
            updateAngles_(obj);
            updateMus_(obj);
            updateSymmetry_(obj);
        end
    end

    methods (Access = protected)

        function s = saveObjectImpl(obj)
            s = saveObjectImpl@saivdr.dictionary.colpprfb.AbstCplxOvsdLpPuFb1dSystem(obj);
            s.nStages  = obj.nStages;
            s.matrixE0 = obj.matrixE0;
            s.mexFcn   = obj.mexFcn;
        end

        function loadObjectImpl(obj,s,wasLocked)
            obj.mexFcn   = s.mexFcn;
            obj.nStages  = s.nStages;
            obj.matrixE0 = s.matrixE0;
            loadObjectImpl@saivdr.dictionary.colpprfb.AbstCplxOvsdLpPuFb1dSystem(obj,s,wasLocked);
        end

        function resetImpl(obj)
            resetImpl@saivdr.dictionary.colpprfb.AbstCplxOvsdLpPuFb1dSystem(obj);
            % Build MEX
            import saivdr.dictionary.cnsoltx.mexsrcs.fcn_autobuild_cbb_type2
            [obj.mexFcn, obj.mexFlag] = fcn_autobuild_cbb_type2(...
                floor(obj.NumberOfChannels/2));
        end

        function setupImpl(obj,varargin)
            % Prepare MEX function
            import saivdr.dictionary.cnsoltx.mexsrcs.fcn_autobuild_cbb_type2
            [obj.mexFcn, obj.mexFlag] = fcn_autobuild_cbb_type2(...
                floor(obj.NumberOfChannels/2));
        end

        function updateProperties_(obj)
            import saivdr.dictionary.utility.ParameterMatrixContainer

            % Check DecimationFactor
            if ~isscalar(obj.DecimationFactor)
                error('DecimationFactor must be scalar.');
            end
            nHalfDecs = obj.DecimationFactor/2;

            % Check PolyPhaseOrder
            if isempty(obj.PolyPhaseOrder)
                obj.PolyPhaseOrder = obj.getDefaultPolyPhaseOrder_();
            end
            if ~isscalar(obj.PolyPhaseOrder)
                error('PolyPhaseOrder must be scalar.');
            end
            ord = obj.PolyPhaseOrder;
            if mod(ord,2)~=0
                error('Polyphase order must be even.');
            end
            obj.nStages = uint32(1+double(ord)/2);
            obj.matrixE0 = getMatrixE0_(obj);

            % Check NumberOfChannels
            if length(obj.NumberOfChannels) > 2
                error('Dimension of NumberOfChannels must be less than or equal to two.');
            end
            if isempty(obj.NumberOfChannels)
                obj.NumberOfChannels = 2*floor(nHalfDecs)+1;
            elseif isvector(obj.NumberOfChannels)
                obj.NumberOfChannels = sum(obj.NumberOfChannels);
                if mod(obj.NumberOfChannels,2) == 0
                    id = 'SaivDr:IllegalArgumentException';
                    msg = '#Channels must be odd.';
                    me = MException(id, msg);
                    throw(me);
                end
            end

            % Prepare ParameterMatrixContainer
            paramMtxSizeTab = [obj.NumberOfChannels*ones(1,2);
                repmat([floor(obj.NumberOfChannels/2)*ones(2,2);
                floor(obj.NumberOfChannels/4),1;
                ceil(obj.NumberOfChannels/2)*ones(2,2);
                floor(obj.NumberOfChannels/4),1], obj.nStages-1, 1)];
            obj.ParameterMatrixSet = ParameterMatrixContainer(...
                'MatrixSizeTable',paramMtxSizeTab);

            % Prepare MEX function
            if ~obj.mexFlag
                import saivdr.dictionary.cnsoltx.mexsrcs.fcn_autobuild_cbb_type2
                [obj.mexFcn, obj.mexFlag] = fcn_autobuild_cbb_type2(...
                    floor(obj.NumberOfChannels/2));
            end
        end

        function updateAngles_(obj)
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
            if length(obj.Angles) ~= sizeOfAngles
                id = 'SaivDr:IllegalArgumentException';
                msg = sprintf(...
                    'Length of angles must be %d',...
                    sizeOfAngles);
                me = MException(id, msg);
                throw(me);
            end
        end

        function updateMus_(obj)
            %
            sizeOfMus = obj.NumberOfChannels*(2*obj.nStages-1);
            %
            nChL = floor(obj.NumberOfChannels/2);
            nChU = ceil(obj.NumberOfChannels/2);
            if isscalar(obj.Mus) && obj.Mus == 1
                obj.Mus = [ ones(1,obj.NumberOfChannels),...
                    repmat([ ones(1,nChL), -1*ones(1,nChL),...
                    ones(1,nChU), -1*ones(1,nChL), 1 ], 1, obj.nStages-1)];
            end
            obj.Mus = obj.Mus(:);
            if length(obj.Mus) ~= sizeOfMus
                id = 'SaivDr:IllegalArgumentException';
                msg = sprintf(...
                    'Length of mus must be %d',...
                    sizeOfMus);
                me = MException(id, msg);
                throw(me);
            end
        end

        function value = getAnalysisFilterBank_(obj)
            import saivdr.dictionary.colpprfb.AbstCplxOvsdLpPuFb1dTypeIISystem
            import saivdr.dictionary.cnsoltx.mexsrcs.*

            %
            nChs = obj.NumberOfChannels;
            dec  = obj.DecimationFactor;
            ord = obj.PolyPhaseOrder;
            pmMtxSt_ = obj.ParameterMatrixSet;
            mexFcn_  = obj.mexFcn;
            mexFlag_ = obj.mexFlag;
            
            %
            E0 = obj.matrixE0;
            %
            V0 = step(pmMtxSt_,[],uint32(1));
            E = V0*[ E0 ; zeros(nChs-dec,dec) ];
            iParamMtx = uint32(2);

            % Order extension
            if ord > 0
                nShift = int32(dec);
                for iOrd = 1:uint32(double(ord)/2)
                    W = step(pmMtxSt_,[],iParamMtx);
                    U = step(pmMtxSt_,[],iParamMtx+1);
                    angsB1 = step(pmMtxSt_,[],iParamMtx+2);
                    HW = step(pmMtxSt_,[],iParamMtx+3);
                    HU = step(pmMtxSt_,[],iParamMtx+4);
                    angsB2 = step(pmMtxSt_,[],iParamMtx+5);
                    if mexFlag_
                        E = mexFcn_(E, W, U, angsB1, HW, HU, angsB2, floor(nChs/2), nShift);
                    else
                        import saivdr.dictionary.cnsoltx.mexsrcs.Order2CplxBuildingBlockTypeII
                        hObb = Order2CplxBuildingBlockTypeII();
                        E = step(hObb, E, W, U, angsB1, HW, HU, angsB2, floor(nChs/2), nShift);
                    end
                    iParamMtx = iParamMtx+6;
                end
                len = dec*(ord+1);
            end
            Phi = diag(exp(1i*obj.Symmetry));
            E = Phi*E;
            value = E.';
        end

    end

end
