classdef AbstOvsdLpPuFb2dSystem < matlab.System %#codegen
    %ABSTOVSDLPPUFB2DSYSTEM Abstract class 2-D OLPPUFB
    %
    % SVN identifier:
    % $Id: AbstOvsdLpPuFb2dSystem.m 683 2015-05-29 08:22:13Z sho $
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
            'ParameterMatrixSet',...
            'Char'});            
    end
    
    properties (Hidden)
        ParameterMatrixSet
        Angles = 0;
        Mus    = 1;   
    end
        
    properties (Access = protected)
        Coefficients
    end

    methods (Access = protected, Abstract = true)
        value = getAnalysisFilterBank_(obj)
        updateParameterMatrixSet_(obj)
        updateAngles_(obj)
        updateMus_(obj)
    end
    
    methods 
        function obj = AbstOvsdLpPuFb2dSystem(varargin)
            setProperties(obj,nargin,varargin{:});
        end
        
        function atmimshow(obj)
            % Show Atomic Images
            updateParameterMatrixSet_(obj);
            H = getAnalysisFilterBank_(obj);
            coder.extrinsic('imshow')
            for ib=1:sum(obj.NumberOfChannels)
                subplot(2,obj.NumberOfChannels(1),ib);
                imshow(rot90(H(:,:,ib),2)+0.5);
            end
        end

    end
    
    methods (Access = protected)

        function s = saveObjectImpl(obj)
            s = saveObjectImpl@matlab.System(obj);
            s.ParameterMatrixSet = matlab.System.saveObject(obj.ParameterMatrixSet);
            s.Coefficients = obj.Coefficients;
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            import saivdr.dictionary.utility.ParameterMatrixSet
            obj.Coefficients = s.Coefficients;
            loadObjectImpl@matlab.System(obj,s,wasLocked);
            %
            obj.ParameterMatrixSet = matlab.System.loadObject(s.ParameterMatrixSet);
        end        
        
        function validatePropertiesImpl(obj)
            id = 'SaivDr:IllegalPropertyException';
            if prod(obj.DecimationFactor) > sum(obj.NumberOfChannels)
                error('%s:\n sum(NumberOfChannels) must be greater than or equalto prod(DecimationFactor).',...
                    id);
            end
        end        
        
        %function setupImpl(obj,varargin)
        %end
        
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
            elseif strcmp(obj.OutputMode,'Char')
                import saivdr.dictionary.utility.PolyPhaseCoefOperation2dSystem
                ppm = PolyPhaseCoefOperation2dSystem('Operation','MTimes');
                vectorH = step(ppm,obj.Coefficients,delayChain_(obj));                
                ppc = PolyPhaseCoefOperation2dSystem('Operation','Char');
                output = step(ppc,vectorH);
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
                output = rot90(H(:,:,idx),2);
            elseif strcmp(obj.OutputMode,'SynthesisFilters')     
                H = getAnalysisFilterBank_(obj);          
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
            coefs = zeros(nSubbands,decY*decX,ordY+1,ordX+1);
            for iSubband = 1:nSubbands
                hi = H(:,:,iSubband);
                for iOrdX = 0:ordX
                    for iOrdY = 0:ordY
                        b = hi(...
                            iOrdY*decY+1:(iOrdY+1)*decY,...
                            iOrdX*decX+1:(iOrdX+1)*decX);
                        coefs(iSubband,:,iOrdY+1,iOrdX+1) = b(:).';
                    end
                end
            end
            obj.Coefficients = coefs;
        end
        
        function value = getMatrixE0_(obj)
            import saivdr.dictionary.utility.Direction
            nRows = obj.DecimationFactor(Direction.VERTICAL);
            nCols = obj.DecimationFactor(Direction.HORIZONTAL);
            nElmBi = nRows*nCols;
            %dctY = dct(eye(nRows)); %dctmtx(nRows);
            %dctX = dct(eye(nCols)); %dctmtx(nCols);
            coefs = zeros(nElmBi);
            iElm = 1; % E0.'= [ Bee Boo Boe Beo ] % Byx
            for iCol = 1:2:nCols % x-e
                for iRow = 1:2:nRows % y-e
                    dctCoef = zeros(nRows,nCols);
                    dctCoef(iRow,iCol) = 1;
                    %basisImage = dctY.'*dctCoef*dctX;
                    basisImage = idct2(dctCoef);
                    coefs(iElm,:) = basisImage(:).';
                    iElm = iElm + 1;
                end
            end
            for iCol = 2:2:nCols % x-o
                for iRow = 2:2:nRows % y-e
                    dctCoef = zeros(nRows,nCols);
                    dctCoef(iRow,iCol) = 1;
                    %basisImage = dctY.'*dctCoef*dctX;
                    basisImage = idct2(dctCoef);
                    coefs(iElm,:) = basisImage(:).';
                    iElm = iElm + 1;
                end
            end
            for iCol = 1:2:nCols % x-e
                for iRow = 2:2:nRows % y-o
                    dctCoef = zeros(nRows,nCols);
                    dctCoef(iRow,iCol) = 1;
                    %basisImage = dctY.'*dctCoef*dctX;
                    basisImage = idct2(dctCoef);
                    coefs(iElm,:) = -basisImage(:).';
                    iElm = iElm + 1;
                end
            end
            for iCol = 2:2:nCols % x-o
                for iRow = 1:2:nRows % y-e
                    dctCoef = zeros(nRows,nCols);
                    dctCoef(iRow,iCol) = 1;
                    %basisImage = dctY.'*dctCoef*dctX;
                    basisImage = idct2(dctCoef);
                    coefs(iElm,:) = -basisImage(:).';
                    iElm = iElm + 1;
                end
            end
            value = coefs;
        end
        
        function value = delayChain_(obj)
            import saivdr.dictionary.utility.Direction
            decY = obj.DecimationFactor(Direction.VERTICAL);
            decX = obj.DecimationFactor(Direction.HORIZONTAL);            
            idx = 1;
            delay = zeros(decX*decY,1,decY,decX);
            for iBndX = 1:decX
                for iBndY = 1:decY
                    delay(idx,:,iBndY,iBndX) = 1;
                    idx = idx + 1;
                end
            end
            value = delay;
        end

    end
    
end
