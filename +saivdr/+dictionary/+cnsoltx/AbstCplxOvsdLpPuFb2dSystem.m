classdef AbstCplxOvsdLpPuFb2dSystem < matlab.System %#codegen
    %ABSTOVSDLPPUFB2DSYSTEM Abstract class 2-D OLPPUFB
    %
    % SVN identifier:
    % $Id: AbstCplxOvsdLpPuFb2dSystem.m 866 2015-11-24 04:29:42Z sho $
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

    properties (Nontunable)
        DecimationFactor = [ 2 2 ];
        PolyPhaseOrder   = [];
        NumberOfChannels = [];
        OutputMode = 'Coefficients'
    end

    properties (Hidden, Transient)
        OutputModeSet = ...
            matlab.system.StringSet({...
            'Coefficients',...
            'AnalysisFilters',...
            'AnalysisFilterAt',...
            'SynthesisFilters',...
            'SynthesisFilterAt',...
            'ParameterMatrixSet',...
            'Char'});
    end

    properties (Hidden)
        Symmetry = 0;
        Angles = 0;
        Mus    = 1;
    end

    properties (GetAccess = public, SetAccess = protected)
        ParameterMatrixSet
    end

    properties (Access = protected)
        Coefficients
    end

    properties(Access = protected, Logical)
        mexFlag = false
    end

    methods (Access = protected, Abstract = true)
        value = getAnalysisFilterBank_(obj)
        updateParameterMatrixSet_(obj)
        updateAngles_(obj)
        updateMus_(obj)
    end

    methods
        function obj = AbstCplxOvsdLpPuFb2dSystem(varargin)
            setProperties(obj,nargin,varargin{:});
        end

        function atmimshow(obj)
            % Show Atomic Images
            updateParameterMatrixSet_(obj);
            obj.mexFlag = false;
            H = getAnalysisFilterBank_(obj);
            for ib=1:obj.NumberOfChannels
                % Real Part
                subplot(2,obj.NumberOfChannels,ib);
                imshow(real(H(:,:,ib))+0.5);
                % Imaginary Part
                subplot(2,obj.NumberOfChannels,ib+obj.NumberOfChannels);
                imshow(imag(H(:,:,ib))+0.5);
            end
        end

    end

    methods (Access = protected)

        function s = saveObjectImpl(obj)
            s = saveObjectImpl@matlab.System(obj);
            s.ParameterMatrixSet = matlab.System.saveObject(obj.ParameterMatrixSet);
            s.Coefficients = obj.Coefficients;
            s.mexFlag = obj.mexFlag;
        end

        function loadObjectImpl(obj,s,wasLocked)
            import saivdr.dictionary.utility.ParameterMatrixSet
            obj.mexFlag = s.mexFlag;
            obj.Coefficients = s.Coefficients;
            loadObjectImpl@matlab.System(obj,s,wasLocked);
            %
            obj.ParameterMatrixSet = matlab.System.loadObject(s.ParameterMatrixSet);
        end

        function validatePropertiesImpl(obj)
            id = 'SaivDr:IllegalPropertyException';
            if prod(obj.DecimationFactor) > sum(obj.NumberOfChannels)
                error('%s:\n sum(NumberOfChannels) must be greater than or equal to prod(DecimationFactor).',...
                    id);
            end
        end

        function resetImpl(obj)
            obj.mexFlag = false;
        end

        function output = stepImpl(obj,varargin)

            if ~isempty(varargin{1})
                obj.Angles = varargin{1};
                updateAngles_(obj);
            end
            if ~isempty(varargin{2})
                obj.Mus = varargin{2};
                updateMus_(obj);
            end

            updateParameterMatrixSet_(obj);
            updateCoefficients_(obj);

            if strcmp(obj.OutputMode,'Coefficients')
                output = obj.Coefficients;
            elseif strcmp(obj.OutputMode,'ParameterMatrixSet')
                output = clone(obj.ParameterMatrixSet);
            elseif strcmp(obj.OutputMode,'AnalysisFilterAt')
                idx = varargin{3};
                H = getAnalysisFilterBank_(obj);
                output = H(:,:,idx);
            elseif strcmp(obj.OutputMode,'AnalysisFilters')
                output = getAnalysisFilterBank_(obj);
            elseif strcmp(obj.OutputMode,'SynthesisFilterAt')
                idx = varargin{3};
                H = getAnalysisFilterBank_(obj);
                H = conj(H);
                output = rot90(H(:,:,idx),2);
            elseif strcmp(obj.OutputMode,'SynthesisFilters')
                H = getAnalysisFilterBank_(obj);
                H = conj(H);
                H = flip(H,1);
                output = flip(H,2);
            else
                output = [];
            end
        end

        function N = getNumInputsImpl(obj)
            if strcmp(obj.OutputMode,'Coefficients') || ...
                    strcmp(obj.OutputMode,'Char') || ...
                    strcmp(obj.OutputMode,'ParameterMatrixSet') || ...
                    strcmp(obj.OutputMode,'AnalysisFilters') || ...
                    strcmp(obj.OutputMode,'SynthesisFilters')
                N = 2;
            else
                N = 3;
            end
        end

        function N = getNumOutputsImpl(~)
            N = 1;
        end

        function updateCoefficients_(obj)
            import saivdr.dictionary.utility.Direction
            decX = obj.DecimationFactor(Direction.HORIZONTAL);
            decY = obj.DecimationFactor(Direction.VERTICAL);
            ordX = obj.PolyPhaseOrder(Direction.HORIZONTAL);
            ordY = obj.PolyPhaseOrder(Direction.VERTICAL);
            nChs = obj.NumberOfChannels;
            %
            H = getAnalysisFilterBank_(obj);
            nSubbands = sum(nChs);
            coefs = complex(zeros(nSubbands,decY*decX,ordY+1,ordX+1));
            for iSubband = 1:nSubbands
                hi = H(:,:,iSubband);
                for iOrdX = 0:ordX
                    idxX = iOrdX*decX;
                    for iOrdY = 0:ordY
                        idxY = iOrdY*decY;
                        b = hi(...
                            idxY+1:idxY+decY,...
                            idxX+1:idxX+decX);
                        coefs(iSubband,:,iOrdY+1,iOrdX+1) = b(:).';
                    end
                end
            end
            obj.Coefficients = coefs;
        end
        
        function updateSymmetry_(obj)
            if isscalar(obj.Symmetry) && obj.Symmetry == 0
                obj.Symmetry = zeros(obj.NumberOfChannels,1);
            end
            if length(obj.Symmetry) ~= obj.NumberOfChannels
                %TODO: ó·äOèàóù
            end
        end

        function value = getMatrixE0_(obj)
            import saivdr.dictionary.utility.Direction
            import saivdr.utility.HermitianSymmetricDFT
            nRows = obj.DecimationFactor(Direction.VERTICAL);
            nCols = obj.DecimationFactor(Direction.HORIZONTAL);
            nElmBi = nRows*nCols;
            hsdftY = HermitianSymmetricDFT.hsdftmtx(nRows);
            hsdftX = HermitianSymmetricDFT.hsdftmtx(nCols);
            coefs = complex(zeros(nElmBi));
            iElm = 1;
            for iCol = 1:nCols
                for iRow = 1:nRows
                    hsdftCoef = complex(zeros(nRows,nCols));
                    hsdftCoef(iRow,iCol) = 1;
                    basisImage = conj(hsdftY).'*hsdftCoef*conj(hsdftX);
                    coefs(iElm,:) = basisImage(:).';
                    iElm = iElm + 1;
                end
            end
            value = coefs;

        end

        function value = permuteCoefs_(~,arr_,phs_)
            len_ = size(arr_,2)/phs_;
            value = zeros(size(arr_));
            for idx = 0:phs_-1
                value(:,idx*len_+1:(idx+1)*len_) = arr_(:,idx+1:phs_:end);
            end
        end

        function value = ipermuteCoefs_(~,arr_,phs_)
            len_ = size(arr_,2)/phs_;
            value = zeros(size(arr_));
            for idx = 0:phs_-1
                value(:,idx+1:phs_:end) = arr_(:,idx*len_+1:(idx+1)*len_);
            end
        end
        
        function [initAngles,propAngles] = splitAngles_(obj)
            nCh = obj.NumberOfChannels;
            nInitAngles = nCh*(nCh-1)/2;
            
            initAngles = obj.Angles(1:nInitAngles);
            
            propAngles = obj.Angles(nInitAngles+1:end);
        end

    end

end
