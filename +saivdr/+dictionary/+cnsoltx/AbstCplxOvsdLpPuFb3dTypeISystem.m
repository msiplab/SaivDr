classdef AbstCplxOvsdLpPuFb3dTypeISystem < ...
        saivdr.dictionary.cnsoltx.AbstCplxOvsdLpPuFb3dSystem %#codegen
    %ABSTOVSDLPPUFB3DTYPEISYSTEM Abstract class 3-D Type-I OLPPUFB
    %
    % Requirements: MATLAB R2013b
    %
    % Copyright (c) 2014-2016, Shogo MURAMATSU
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
        function obj = AbstCplxOvsdLpPuFb3dTypeISystem(varargin)
            obj = obj@saivdr.dictionary.cnsoltx.AbstCplxOvsdLpPuFb3dSystem(...
                varargin{:});
            updateProperties_(obj);
            updateAngles_(obj);
            updateMus_(obj);
            updateSymmetry_(obj);
        end
    end

    methods (Access = protected)

        function s = saveObjectImpl(obj)
            s = saveObjectImpl@saivdr.dictionary.cnsoltx.AbstCplxOvsdLpPuFb3dSystem(obj);
            s.nStages = obj.nStages;
            s.matrixE0 = obj.matrixE0;
            s.mexFcn   = obj.mexFcn;
        end

        function loadObjectImpl(obj,s,wasLocked)
            obj.nStages = s.nStages;
            obj.matrixE0 = s.matrixE0;
            loadObjectImpl@saivdr.dictionary.cnsoltx.AbstCplxOvsdLpPuFb3dSystem(obj,s,wasLocked);
            %
            if ~isempty(s.mexFcn)
                if exist(func2str(s.mexFcn),'file') == 3
                    obj.mexFcn  = s.mexFcn;
                else
                    import saivdr.dictionary.cnsoltx.mexsrcs.fcn_Order1CplxBuildingBlockTypeI
                    obj.mexFcn = @fcn_Order1CplxBuildingBlockTypeI;
                end
            end
        end

        function resetImpl(obj)
            resetImpl@saivdr.dictionary.cnsoltx.AbstCplxOvsdLpPuFb3dSystem(obj);
%             % Prepare MEX function
%             import saivdr.dictionary.cnsoltx.mexsrcs.fcn_autobuild_cbb_type1
%             [obj.mexFcn, obj.mexFlag] = fcn_autobuild_cbb_type1(floor(obj.NumberOfChannels/2));
        end

        function setupImpl(obj,varargin)
            % Prepare MEX function
            if exist('fcn_Order1CplxBuildingBlockTypeI_mex','file')==3
                obj.mexFcn = @fcn_Order1CplxBuildingBlockTypeI_mex;
                obj.mexFlag = true;
            else
                import saivdr.dictionary.cnsoltx.mexsrcs.fcn_Order1CplxBuildingBlockTypeI
                obj.mexFcn = @fcn_Order1CplxBuildingBlockTypeI;
                obj.mexFlag = false;
            end
        end

        function updateProperties_(obj)
            import saivdr.dictionary.utility.Direction
            import saivdr.dictionary.utility.ParameterMatrixContainer
           % import saivdr.dictionary.cnsoltx.AbstCplxOvsdLpPuFb3dTypeISystem

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
            paramMtxSizeTab = [
                obj.NumberOfChannels*ones(1,2);
                repmat([
                    obj.NumberOfChannels/2*ones(2,2);
                    floor(obj.NumberOfChannels/4),1],...
                    obj.nStages-1, 1)];
            obj.ParameterMatrixSet = ParameterMatrixContainer(...
                'MatrixSizeTable',paramMtxSizeTab);

            % Prepare MEX function
%            if ~obj.mexFlag
%                import saivdr.dictionary.cnsoltx.mexsrcs.fcn_autobuild_bb_type1
%                [obj.mexFcn, obj.mexFlag] = fcn_autobuild_bb_type1(obj.NumberOfChannels(1));
%            end
        end

        function updateAngles_(obj)
            import saivdr.dictionary.utility.Direction
            nChL = obj.NumberOfChannels/2;
            nAngsInitStg = obj.NumberOfChannels*(obj.NumberOfChannels-1)/2;
            nAngsPerStg = nChL*(nChL-1)+floor(nChL/2);
            sizeOfAngles = nAngsInitStg+nAngsPerStg*(obj.nStages-1);
            if isscalar(obj.Angles) && obj.Angles == 0
                obj.Angles = zeros(sizeOfAngles,1);
            end
            obj.Angles = obj.Angles(:);
            if length(obj.Angles) ~= sizeOfAngles
                id = 'SaivDr:IllegalArgumentException';
                msg = sprintf(...
                    'Length of angles must be %d',sizeOfAngles);
                me = MException(id, msg);
                throw(me);
            end
        end
        
        function updateMus_(obj)
            nChL = obj.NumberOfChannels/2;
            sizeOfMus = 2*nChL*obj.nStages;
            if isscalar(obj.Mus) && obj.Mus == 1
                musMat = ones(2*nChL,obj.nStages);
                musMat(nChL+1:end,2:end) = -1*ones(nChL,obj.nStages-1);
                obj.Mus = musMat(:);
            end
            obj.Mus = obj.Mus(:);
            if length(obj.Mus) ~= sizeOfMus
                id = 'SaivDr:IllegalArgumentException';
                msg = sprintf(...
                    'Length of mus must be %d',sizeOfMus);
                me = MException(id, msg);
                throw(me);
            end
        end

        function value = getAnalysisFilterBank_(obj)
            import saivdr.dictionary.utility.Direction
            import saivdr.dictionary.cnsoltx.AbstCplxOvsdLpPuFb3dTypeISystem
            %
            nChs = obj.NumberOfChannels;
            dec  = obj.DecimationFactor;
            decX = dec(Direction.HORIZONTAL);
            decY = dec(Direction.VERTICAL);
            decZ = dec(Direction.DEPTH);
            %nHalfDecs = prod(dec)/2;
            ordX = obj.PolyPhaseOrder(Direction.HORIZONTAL);
            ordY = obj.PolyPhaseOrder(Direction.VERTICAL);
            ordZ = obj.PolyPhaseOrder(Direction.DEPTH);
            pmMtxSet_ = obj.ParameterMatrixSet;
            mexFcn_ = obj.mexFcn;
            mexFlag_ = obj.mexFlag;
            %
            E0 = obj.matrixE0;
            %
            V0 = step(pmMtxSet_,[],uint32(1));
            E = V0*[ E0 ; zeros(nChs-prod(dec),prod(dec))];
            iParamMtx = uint32(2);
            hChs = nChs/2;

            % Depth extention
            lenY = decY;
            lenX = decX;
            nShift = int32(lenY*(decZ*lenX));
            for iOrdZ = 1:ordZ
                W = step(pmMtxSet_,[],iParamMtx);
                U = step(pmMtxSet_,[],iParamMtx+1);
                angsB = step(pmMtxSet_,[],iParamMtx+2);
                if mexFlag_
                    E = mexFcn_(E, W, U, angsB, hChs, nShift);
                else
                    import saivdr.dictionary.cnsoltx.mexsrcs.Order1CplxBuildingBlockTypeI
                    hObb = Order1CplxBuildingBlockTypeI();
                    E = step(hObb, E, W, U, angsB, hChs, nShift);
                end
                iParamMtx = iParamMtx+3;
            end
            lenZ = decZ*(ordZ+1);

            % Horizontal extention
            E = permuteCoefs_(obj,E,lenY*lenX); % Y X Z -> Z Y X
            nShift = int32(lenZ*(decX*lenY));
            for iOrdX = 1:ordX
                W = step(pmMtxSet_,[],iParamMtx);
                U = step(pmMtxSet_,[],iParamMtx+1);
                angsB = step(pmMtxSet_,[],iParamMtx+2);
                if mexFlag_
                    E = mexFcn_(E, W, U, angsB, hChs, nShift);
                else
                    import saivdr.dictionary.cnsoltx.mexsrcs.Order1CplxBuildingBlockTypeI
                    hObb = Order1CplxBuildingBlockTypeI();
                    E = step(hObb, E, W, U, angsB, hChs, nShift);
                end
                iParamMtx = iParamMtx+3;
            end
            lenX = decX*(ordX+1);

            % Vertical extention
            E = permuteCoefs_(obj,E,lenZ*lenY); % Z Y X -> X Z Y
            nShift = int32(lenX*(decY*lenZ));
            for iOrdY = 1:ordY
                W = step(pmMtxSet_,[],iParamMtx);
                U = step(pmMtxSet_,[],iParamMtx+1);
                angsB = step(pmMtxSet_,[],iParamMtx+2);
                if mexFlag_
                    E = mexFcn_(E, W, U, angsB, hChs, nShift);
                else
                    import saivdr.dictionary.cnsoltx.mexsrcs.Order1CplxBuildingBlockTypeI
                    hObb = Order1CplxBuildingBlockTypeI();
                    E = step(hObb, E, W, U, angsB, hChs, nShift);
                end
                iParamMtx = iParamMtx+3;
            end
            lenY = decY*(ordY+1);

            %
            E = permuteCoefs_(obj,E,lenX*lenZ); % X Z Y -> Y X Z
            Phi = diag(exp(1i*obj.Symmetry));
            E = Phi*E;
            nSubbands = size(E,1);
            value = zeros(lenY,lenX,lenZ,nSubbands);
            for iSubband = 1:nSubbands
                value(:,:,:,iSubband) = reshape(E(iSubband,:),lenY,lenX,lenZ);
            end

        end

    end

end
