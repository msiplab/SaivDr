classdef Synthesis3dOlaWrapper < saivdr.dictionary.AbstSynthesisSystem
    %SYNTHESIS3DOLAWRAPPER OLA wrapper for 3-D synthesis system
    %
    % Reference:
    %   Shogo Muramatsu and Hitoshi Kiya,
    %   ''Parallel Processing Techniques for Multidimensional Sampling
    %   Lattice Alteration Based on Overlap-Add and Overlap-Save Methods,''
    %   IEICE Trans. on Fundamentals, Vol.E78-A, No.8, pp.939-943, Aug. 1995
    %
    % Requirements: MATLAB R2018a
    %
    % Copyright (c) 2018, Shogo MURAMATSU
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
        Synthesizer
        BoundaryOperation
        PadSize = [0 0 0]
        InputType = 'Vector'
        SplitFactor = []
    end
    
    properties (Logical)
        UseParallel = false
        IsIntegrityTest = true
    end
    
    properties (Nontunable, PositiveInteger, Hidden)
        VerticalSplitFactor = 1
        HorizontalSplitFactor = 1
        DepthSplitFactor = 1
    end
    
    properties (Hidden, Transient)
        BoundaryOperationSet = ...
            matlab.system.StringSet({'Circular'});
        InputTypeSet = ...
            matlab.system.StringSet({'Vector','Cell'});
    end
    
    properties (Access = private, Nontunable)
        refSize
        refSubSize
        subPadSize
        subPadArrays
        synthesizers
        nWorkers
    end
    
    methods
        
        % Constractor
        function obj = Synthesis3dOlaWrapper(varargin)
            import saivdr.dictionary.utility.Direction
            setProperties(obj,nargin,varargin{:})
            if ~isempty(obj.Synthesizer)
                obj.BoundaryOperation = obj.Synthesizer.BoundaryOperation;
            end
            if ~isempty(obj.SplitFactor)
                obj.VerticalSplitFactor = obj.SplitFactor(Direction.VERTICAL);
                obj.HorizontalSplitFactor = obj.SplitFactor(Direction.HORIZONTAL);
                obj.DepthSplitFactor = obj.SplitFactor(Direction.DEPTH);
            end
        end
        %{
        function setFrameBound(obj,frameBound)
            obj.FrameBound = frameBound;
        end
        %}
    end
    
    methods (Access = protected)
        
        function flag = isInactivePropertyImpl(obj,propertyName)
            if strcmp(propertyName,'VerticalSplitFactor') || ...
                    strcmp(propertyName,'HorizontalSplitFactor') || ...
                    strcmp(propertyName,'DepthSplitFactor')
                flag = ~isempty(obj.SplitFactor);
            else
                flag = false;
            end
        end
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@saivdr.dictionary.AbstSynthesisSystem(obj);
            s.Synthesizer = matlab.System.saveObject(obj.Synthesizer);
            s.nWorkers = obj.nWorkers;
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            obj.nWorkers = s.nWorkers;
            obj.Synthesizer = matlab.System.loadObject(s.Synthesizer);
            loadObjectImpl@saivdr.dictionary.AbstSynthesisSystem(obj,s,wasLocked);
        end
        
        function setupImpl(obj,coefs,scales)
            obj.Synthesizer.release();
            refSynthesizer = obj.Synthesizer.clone();
            %
            verticalSplitFactor = obj.VerticalSplitFactor;
            horizontalSplitFactor = obj.HorizontalSplitFactor;
            depthSplitFactor = obj.DepthSplitFactor;
            nSplit = verticalSplitFactor*horizontalSplitFactor*depthSplitFactor;
            nChs = size(scales,1);
            %
            if strcmp(obj.InputType,'Cell')
                subCoefArrays = obj.merge_(coefs,scales);
                tmpArrays = cell(verticalSplitFactor,horizontalSplitFactor,depthSplitFactor);
                refCoefs = [];
                for iCh = 1:nChs
                    iSplit = 0;
                    for iLay = 1:depthSplitFactor
                        for iCol = 1:horizontalSplitFactor
                            for iRow = 1:verticalSplitFactor
                                iSplit = iSplit + 1;
                                tmpArrays{iRow,iCol,iLay} = subCoefArrays{iSplit,iCh};
                            end
                        end
                    end
                    tmpArray = cell2mat(tmpArrays);
                    tmpVec = tmpArray(:).';
                    refCoefs = [ refCoefs tmpVec ];
                end
                refScales = scales*diag(...
                    [verticalSplitFactor horizontalSplitFactor depthSplitFactor]);
            else
                refCoefs = coefs;
                refScales = scales;
            end
            recImg = step(refSynthesizer,refCoefs,refScales);
            obj.refSize = size(recImg);
            obj.refSubSize = obj.refSize*...
                diag(1./[obj.VerticalSplitFactor,...
                obj.HorizontalSplitFactor,...
                obj.DepthSplitFactor ]);
            %
            scaleRatio = refScales*diag(1./obj.refSize);
            obj.subPadSize = scaleRatio*diag(obj.PadSize);
            %
            obj.synthesizers = cell(nSplit,1);
            if obj.UseParallel
                obj.nWorkers = Inf;
                for iSplit=1:nSplit
                    obj.synthesizers{iSplit} = clone(obj.Synthesizer);
                end
            else
                obj.nWorkers = 0;
                for iSplit=1:nSplit
                    obj.synthesizers{iSplit} = obj.Synthesizer;
                end
            end
            
            % Evaluate
            % Check if scales are divisible by split factors
            exceptionId = 'SaivDr:IllegalSplitFactorException';
            message = 'Split factor must be a divisor of array size.';
            if sum(mod(obj.refSubSize,1)) ~= 0
                throw(MException(exceptionId,message))
            end
            % Check if subPadSizes are integer
            %exceptionId = 'SaivDr:IllegalSplitFactorException';
            %message = 'Split factor must be a divisor of array size.';
            if sum(mod(obj.subPadSize,1)) ~= 0
                throw(MException('SaivDr','Illegal Pad Size.'))
            end
            % Allocate memory for zero padding of arrays
            obj.subPadArrays = cell(nChs,1);
            nCoefs = 0;
            for iCh = 1:nChs
                subScale = refScales(iCh,:) * ...
                    diag(1./[obj.VerticalSplitFactor,...
                    obj.HorizontalSplitFactor,...
                    obj.DepthSplitFactor]);
                nDim = subScale + 2*obj.subPadSize(iCh,:);
                obj.subPadArrays{iCh} = zeros(subScale+2*obj.subPadSize(iCh,:));
                nCoefs = nCoefs + prod(nDim);
            end
            % Check integrity
            if obj.IsIntegrityTest
                exceptionId = 'SaivDr:ReconstructionFailureException';
                message = 'Failure occurs in reconstruction. Please check the split and padding size.';
                newImg = stepImpl(obj,coefs,scales);
                diffImg = recImg - newImg;
                if norm(diffImg(:))/numel(diffImg) > 1e-6
                    throw(MException(exceptionId,message))
                end
            end
            % Delete reference synthesizer
            refSynthesizer.delete()
        end
        
        function recImg = stepImpl(obj,coefs,scales)
            % 1. Split
            if strcmp(obj.InputType,'Cell')
                subCoefArrays = obj.merge_(coefs,scales);
            else
                [subCoefs,subScales] = obj.split_(coefs,scales);
                subCoefArrays = obj.merge_(subCoefs,subScales);
            end
            % 2. Zero Padding
            subCoefArrays = padding_(obj,subCoefArrays);
            [subCoefs,subScales] = convert_(obj,subCoefArrays);
            % 3. Synthesize
            nSplit = length(subCoefs);
            subRecImg = cell(nSplit,1);
            %
            synthesizers_ = obj.synthesizers;
            nWorkers_ = obj.nWorkers;
            parfor (iSplit=1:nSplit,nWorkers_)
                subCoefs_ = subCoefs{iSplit};
                subRecImg{iSplit} = step(synthesizers_{iSplit},subCoefs_,subScales);
            end
            % 4. Overlap add (Circular)
            recImg = circular_ola_(obj,subRecImg);
        end
        
    end
    
    methods(Access = private)
        
        function recImg = circular_ola_(obj,subRecImg)
            import saivdr.dictionary.utility.Direction
            verticalSplitFactor = obj.VerticalSplitFactor;
            horizontalSplitFactor = obj.HorizontalSplitFactor;
            depthSplitFactor = obj.DepthSplitFactor;
            stepsize = obj.refSubSize;
            overlap = size(subRecImg{1})-stepsize;
            recImg = zeros(obj.refSize+overlap);
            % Overlap add
            iSplit = 0;
            tIdxDep = 0;
            for iDepSplit = 1:depthSplitFactor
                sIdxDep = tIdxDep + 1;
                tIdxDep = sIdxDep + stepsize(Direction.DEPTH) - 1;
                eIdxDep = tIdxDep + overlap(Direction.DEPTH);
                tIdxHor = 0;
                for iHorSplit = 1:horizontalSplitFactor
                    sIdxHor = tIdxHor + 1;
                    tIdxHor = sIdxHor + stepsize(Direction.HORIZONTAL) - 1;
                    eIdxHor = tIdxHor + overlap(Direction.HORIZONTAL);
                    tIdxVer = 0;
                    for iVerSplit = 1:verticalSplitFactor
                        iSplit = iSplit + 1;
                        sIdxVer = tIdxVer + 1;
                        tIdxVer = sIdxVer + stepsize(Direction.VERTICAL) - 1;
                        eIdxVer = tIdxVer + overlap(Direction.VERTICAL);
                        recImg(sIdxVer:eIdxVer,sIdxHor:eIdxHor,sIdxDep:eIdxDep) = ...
                            recImg(sIdxVer:eIdxVer,sIdxHor:eIdxHor,sIdxDep:eIdxDep) + ...
                            subRecImg{iSplit};
                    end
                end
            end
            % Folding
            recImg(1:overlap(Direction.VERTICAL),:,:) = ...
                recImg(1:overlap(Direction.VERTICAL),:,:) + ...
                recImg(end-overlap(Direction.VERTICAL)+1:end,:,:);
            recImg(:,1:overlap(Direction.HORIZONTAL),:) = ...
                recImg(:,1:overlap(Direction.HORIZONTAL),:) + ...
                recImg(:,end-overlap(Direction.HORIZONTAL)+1:end,:);
            recImg(:,:,1:overlap(Direction.DEPTH)) = ...
                recImg(:,:,1:overlap(Direction.DEPTH)) + ...
                recImg(:,:,end-overlap(Direction.DEPTH)+1:end);
            % Cropping & circular shift
            recImg = circshift(recImg(...
                1:obj.refSize(Direction.VERTICAL),...
                1:obj.refSize(Direction.HORIZONTAL),...
                1:obj.refSize(Direction.DEPTH)),-overlap/2);
        end
        
        function [subCoefs,subScales] = convert_(~,subCoefArrays)
            nSplit = size(subCoefArrays,1);
            nChs = size(subCoefArrays,2);
            subScales = zeros(nChs,3);
            subCoefs = cell(nSplit,1);
            for iSplit = 1:nSplit
                tmpCoefs_ = cell(1,nChs);
                for iCh = 1:nChs
                    tmpArray = subCoefArrays{iSplit,iCh};
                    if iSplit == 1
                        subScales(iCh,:) = size(tmpArray);
                    end
                    tmpCoefs_{1,iCh} = tmpArray(:).';
                end
                subCoefs{iSplit} = cell2mat(tmpCoefs_);
            end
        end
        
        function subCoefArrays = padding_(obj,subCoefArrays)
            import saivdr.dictionary.utility.Direction
            nSplit = size(subCoefArrays,1);
            nChs = size(subCoefArrays,2);
            subPadSize_ = obj.subPadSize;
            subPadArrays_ = obj.subPadArrays;
            for iCh = 1:nChs
                sRowIdx = subPadSize_(iCh,Direction.VERTICAL)+1;
                eRowIdx = sRowIdx + size(subCoefArrays{1,iCh},Direction.VERTICAL)-1;
                sColIdx = subPadSize_(iCh,Direction.HORIZONTAL)+1;
                eColIdx = sColIdx + size(subCoefArrays{1,iCh},Direction.HORIZONTAL)-1;
                sLayIdx = subPadSize_(iCh,Direction.DEPTH)+1;
                eLayIdx = sLayIdx + size(subCoefArrays{1,iCh},Direction.DEPTH)-1;
                for iSplit = 1:nSplit
                    tmpArray = subPadArrays_{iCh};
                    tmpArray(sRowIdx:eRowIdx,sColIdx:eColIdx,sLayIdx:eLayIdx) ...
                        = subCoefArrays{iSplit,iCh};
                    subCoefArrays{iSplit,iCh} = tmpArray;
                end
            end
        end
        
        function [subCoefs,subScales] = split_(obj,coefs,scales)
            import saivdr.dictionary.utility.Direction
            verticalSplitFactor = obj.VerticalSplitFactor;
            horizontalSplitFactor = obj.HorizontalSplitFactor;
            depthSplitFactor = obj.DepthSplitFactor;
            nSplit = verticalSplitFactor*horizontalSplitFactor*depthSplitFactor;
            % # of channels
            nChs = size(scales,1);
            subScales = scales*diag(...
                [1/verticalSplitFactor, 1/horizontalSplitFactor, 1/depthSplitFactor]);
            subCoefs = cell(nSplit,1);
            %
            eIdx = 0;
            for iCh = 1:nChs
                sIdx = eIdx + 1;
                eIdx = sIdx + prod(scales(iCh,:)) - 1;
                nRows = scales(iCh,Direction.VERTICAL);
                nCols = scales(iCh,Direction.HORIZONTAL);
                nLays = scales(iCh,Direction.DEPTH);
                coefArrays = reshape(coefs(sIdx:eIdx),[nRows nCols nLays]);
                %
                nSubRows = subScales(iCh,Direction.VERTICAL);
                nSubCols = subScales(iCh,Direction.HORIZONTAL);
                nSubLays = subScales(iCh,Direction.DEPTH);
                iSplit = 0;
                for iDepSplit = 1:depthSplitFactor
                    sLayIdx = (iDepSplit-1)*nSubLays + 1;
                    eLayIdx = iDepSplit*nSubLays;
                    for iHorSplit = 1:horizontalSplitFactor
                        sColIdx = (iHorSplit-1)*nSubCols + 1;
                        eColIdx = iHorSplit*nSubCols;
                        for iVerSplit = 1:verticalSplitFactor
                            iSplit = iSplit + 1;
                            sRowIdx = (iVerSplit-1)*nSubRows + 1;
                            eRowIdx = iVerSplit*nSubRows;
                            tmpArray = ...
                                coefArrays(sRowIdx:eRowIdx,sColIdx:eColIdx,sLayIdx:eLayIdx);
                            %
                            tmpVec = tmpArray(:).';
                            subCoefs{iSplit} = [ subCoefs{iSplit} tmpVec ];
                        end
                    end
                end
            end
        end
        
        
        function subCoefArrays = merge_(obj,subCoefs,subScales)
            verticalSplitFactor = obj.VerticalSplitFactor;
            horizontalSplitFactor = obj.HorizontalSplitFactor;
            depthSplitFactor = obj.DepthSplitFactor;
            nSplit = verticalSplitFactor*horizontalSplitFactor*depthSplitFactor;
            % # of channels
            nChs = size(subScales,1);
            subCoefArrays = cell(nSplit,nChs);
            %
            for iSplit = 1:nSplit
                coefsSplit = subCoefs{iSplit};
                eIdx = 0;
                for iCh = 1:nChs
                    sIdx = eIdx + 1;
                    eIdx = sIdx + prod(subScales(iCh,:)) - 1;
                    tmpSubCoefs = coefsSplit(sIdx:eIdx);
                    subCoefArrays{iSplit,iCh} = ...
                        reshape(tmpSubCoefs,subScales(iCh,:));
                end
            end
        end
        
    end
end