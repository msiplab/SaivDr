classdef OlsOlaProcess2d < matlab.System
    %OLSOLAPROCESS2D OLS/OLA wrapper for 2-D analysis and synthesis system
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
        Analyzer
        Synthesizer
        BoundaryOperation
        PadSize = [0 0]
        SplitFactor = []
        CoefsManipulator = []
        InitialState
    end
    
    properties (Logical, Nontunable)
        UseGpu = false
        UseParallel = false
        IsIntegrityTest = true
    end
    
    properties (DiscreteState)
        iteration
    end
    
    properties (Access = private)
        states
    end
    
    properties (Nontunable, PositiveInteger, Hidden)
        VerticalSplitFactor = 1
        HorizontalSplitFactor = 1
    end
    
    properties (Hidden, Transient)
        BoundaryOperationSet = ...
            matlab.system.StringSet({'Circular'});
    end
    
    properties (Access = private, Nontunable)
        refSize
        refSubSize
        refScales
        subPadSize
        subPadArrays
        nWorkers
    end
    
    properties (Access = private, Logical)
        isSpmd = false;
    end
    
    methods
        
        % Constractor
        function obj = OlsOlaProcess2d(varargin)
            import saivdr.dictionary.utility.Direction
            setProperties(obj,nargin,varargin{:})
            if ~isempty(obj.Analyzer)
                obj.BoundaryOperation = obj.Analyzer.BoundaryOperation;
            end
            if ~isempty(obj.SplitFactor)
                obj.VerticalSplitFactor = obj.SplitFactor(Direction.VERTICAL);
                obj.HorizontalSplitFactor = obj.SplitFactor(Direction.HORIZONTAL);
            else
                obj.SplitFactor(Direction.VERTICAL) = obj.VerticalSplitFactor;
                obj.SplitFactor(Direction.HORIZONTAL) = obj.HorizontalSplitFactor;
            end
            if isempty(obj.CoefsManipulator)
                import saivdr.utility.CoefsManipulator
                obj.CoefsManipulator = CoefsManipulator();
            end
        end
        
        function coefsSet = analyze(obj,srcImg)
            % Preperation
            verticalSplitFactor = obj.VerticalSplitFactor;
            horizontalSplitFactor = obj.HorizontalSplitFactor;
            nSplit = prod(obj.SplitFactor);
            obj.Analyzer.release();
            refAnalyzer = obj.Analyzer.clone();
            [~,refScales_] = refAnalyzer.step(srcImg);            
            
            % Parameters
            obj.refSize = size(srcImg);
            obj.refSubSize = obj.refSize*...
                diag(1./[verticalSplitFactor,horizontalSplitFactor]);
            scaleRatio = refScales_*diag(1./obj.refSize);
            obj.subPadSize = scaleRatio*diag(obj.PadSize);
            obj.refScales = refScales_;
            
            % Analyzer
            analyzer_ = obj.Analyzer;
            
            % Define support functions 
            extract_ols = @(c,s) obj.extract_ols_(c,s);
            
            % Circular global padding
            srcImg_ = padarray(srcImg,obj.PadSize,'circular');
            
            % Overlap save split
            subImgs = obj.split_ols_(srcImg_);
            
            % Analyze
            coefsSet = cell(nSplit,1);
            for iSplit=1:nSplit
                [subCoefs, subScales] = analyzer_.step(subImgs{iSplit});
                coefsSet{iSplit} = extract_ols(subCoefs,subScales);
            end
        end
        
    end
    
    methods(Access = protected)
        
        function flag = isInactivePropertyImpl(obj,propertyName)
            if strcmp(propertyName,'VerticalSplitFactor') || ...
                    strcmp(propertyName,'HorizontalSplitFactor')
                flag = ~isempty(obj.SplitFactor);
            else
                flag = false;
            end
        end
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@matlab.System(obj);
            s.nWorkers = obj.nWorkers;
            s.refSize = obj.refSize;
            s.refScales = obj.refScales;
            s.subPadSize = obj.subPadSize;
            s.refSubSize = obj.refSubSize;
            s.subPadArrays = obj.subPadArrays;
            s.states = obj.states;
            s.isSpmd = obj.isSpmd;
            if isLocked(obj)
                s.iteration = obj.iteration;
            end            
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            if wasLocked
                obj.iteration = s.iteration;
            end                     
            obj.isSpmd = s.isSpmd;            
            obj.states = s.states;
            obj.subPadArrays = s.subPadArrays;            
            obj.refSubSize = s.refSubSize;            
            obj.subPadSize = s.subPadSize;            
            obj.refScales = s.refScales;
            obj.refSize = s.refSize;            
            obj.nWorkers = s.nWorkers;
            loadObjectImpl@matlab.System(obj,s,wasLocked);
        end
        
        function setupImpl(obj,srcImg)
            if isa(srcImg,'gpuArray')
                obj.UseGpu = true;
            end
            
            % Preperation
            verticalSplitFactor = obj.VerticalSplitFactor;
            horizontalSplitFactor = obj.HorizontalSplitFactor;
            nSplit = verticalSplitFactor*horizontalSplitFactor;
            
            % Analyzers
            obj.Analyzer.release();
            refAnalyzer = obj.Analyzer.clone();
            if obj.IsIntegrityTest
                [refCoefs,refScales_] = refAnalyzer.step(srcImg);
            else
                [~,refScales_] = refAnalyzer.step(srcImg);
            end
            
            % Synthesizers
            obj.Synthesizer.release();
            refSynthesizer = obj.Synthesizer.clone();
            
            % Manipulators
            obj.CoefsManipulator.release();
            refCoefsManipulator = obj.CoefsManipulator.clone();
    
            % Parameters
            obj.refSize = size(srcImg);
            obj.refSubSize = obj.refSize*...
                diag(1./[verticalSplitFactor,horizontalSplitFactor]);            
            scaleRatio = refScales_*diag(1./obj.refSize);    
            obj.subPadSize = scaleRatio*diag(obj.PadSize);   
            obj.refScales = refScales_;
            
            % Workers
            if obj.UseParallel 
                pool = gcp('nocreate');
                if isempty(pool)
                    parpool();
                    pool = gcp();
                end
                if pool.NumWorkers >= nSplit
                    obj.nWorkers = pool.NumWorkers;
                    obj.isSpmd = true;
                else
                    obj.nWorkers = Inf;
                    obj.isSpmd = false;                    
                end
            else
                obj.nWorkers = 0;
                obj.isSpmd = false;
            end            
            
            % Evaluate
            % Check if srcImg is divisible by split factors
            exceptionId = 'SaivDr:IllegalSplitFactorException';
            message = 'Split factor must be a divisor of array size.';
            if sum(mod(obj.refSubSize,1)) ~= 0
                throw(MException(exceptionId,message))
            end
            % Check if scales are divisible by split factors
            if sum(mod(obj.subPadSize,1)) ~= 0
                throw(MException('SaivDr','Illegal Pad Size.'))                
            end

            % Allocate memory for zero padding of arrays
            nChs = size(refScales_,1);
            obj.subPadArrays = cell(nChs,1);
            nCoefs = 0;
            for iCh = 1:nChs
                subScale = refScales_(iCh,:) * ...
                    diag(1./[verticalSplitFactor,horizontalSplitFactor]);
                nDim = subScale+2*obj.subPadSize(iCh,:);
                obj.subPadArrays{iCh} = zeros(nDim,'like',srcImg);
                nCoefs = nCoefs + prod(nDim);
            end
    
            % Check integrity
            if obj.IsIntegrityTest
                exceptionId = 'SaivDr:ReconstructionFailureException';
                message = 'Failure occurs in reconstruction. Please check the split and padding size.';
                %
                refCoefsOut = refCoefsManipulator.step(refCoefs,0);
                imgExpctd = refSynthesizer.step(refCoefsOut,refScales_);
                %
                obj.states = num2cell(zeros(nSplit,1,'like',srcImg));
                imgActual = obj.stepImpl(srcImg);
                %
                diffImg = imgExpctd - imgActual;
                if norm(diffImg(:))/numel(diffImg) > 1e-6
                    throw(MException(exceptionId,message))
                end
                %
                if verLessThan('matlab','9.4')
                    obj.CoefsManipulator.release();
                end
            end
                
            % Delete reference analyzer and synthesizer
            refAnalyzer.delete()
            refSynthesizer.delete()
            refCoefsManipulator.delete()
            
            % Initialization of state for CoefsManipulator                       
            if isempty(obj.InitialState)
                for iSplit = 1:nSplit
                    state = num2cell(zeros(1,nChs,'like',srcImg));
                    obj.states{iSplit} = state;
                end
            elseif isscalar(obj.InitialState)
                for iSplit = 1:nSplit
                    state = num2cell(...
                        cast(obj.InitialState,'like',srcImg)*...
                        ones(1,nChs,'like',srcImg));
                    obj.states{iSplit} = state;
                end
            else
                for iSplit = 1:nSplit
                    initState = obj.InitialState{iSplit};
                    state = cellfun(@(x) cast(x,'like',srcImg),...
                        initState,'UniformOutput',false);
                    obj.states{iSplit} = state;
                end
            end
            
        end
        
        function recImg = stepImpl(obj,srcImg)
            
            if obj.isSpmd
                recImg = obj.stepImpl_spmd(srcImg);     % SPMD        
            else
                recImg = obj.stepImpl_parfor(srcImg);   % PARFOR
            end
            
        end
        
        function recImg = stepImpl_spmd(obj,srcImg)
            obj.iteration = obj.iteration + 1;
            
            % Support function handles
            analyze     = @(x)   obj.Analyzer.step(x);
            manipulate  = @(x,s) obj.CoefsManipulator.step(x,s);                        
            synthesize  = @(x,s) obj.Synthesizer.step(x,s);
            extract_ols = @(c,s) obj.extract_ols_(c,s);
            padding_ola = @(c)   obj.padding_ola_(c);
            arr2vec     = @(a)   obj.arr2vec_(a);
            
            % Circular global padding
            srcImg_ = padarray(srcImg,obj.PadSize,'circular');
            
            % Overlap save split
            subImgs = obj.split_ols_(srcImg_);
            
            % Initialize
            nSplit = length(subImgs);
            subImgCmp = Composite(nSplit);
            statesCmp = Composite(nSplit);
            for iSplit=1:nSplit 
                subImgCmp{iSplit} = subImgs{iSplit};
                statesCmp{iSplit} = obj.states{iSplit};
            end

            % Parallel processing
            usegpu_ = obj.UseGpu;
            spmd(nSplit)
                %iSplit = labindex;
                if usegpu_
                    subImgCmp = gpuArray(subImgCmp);
                end
                
                % Analyze
                [subCoefs, subScales] = analyze(subImgCmp);
                
                % Extract significant coefs.
                coefs = extract_ols(subCoefs,subScales);
                
                % Process for coefficients
                if usegpu_ && iscell(statesCmp)
                    stateGpu = cellfun(@gpuArray,statesCmp,...
                        'UniformOutput',false);
                    [coefs,stateGpu] = manipulate(coefs,stateGpu);                
                    statesCmp = cellfun(@gather,stateGpu,...
                        'UniformOutput',false);
                else
                    [coefs,statesCmp] = manipulate(coefs,statesCmp);                
                end
                
                % Zero padding for convolution
                subCoefArray = padding_ola(coefs);
                
                % Synthesis
                [subCoefs,subScales] = arr2vec(subCoefArray);
                subRecImgCmp = synthesize(subCoefs,subScales);
                
                if usegpu_
                    subRecImgCmp = gather(subRecImgCmp);
                end
            end
            
            % Update
            obj.states = statesCmp;            
            
            % Overlap add (Circular)
            subRecImgs = cell(nSplit,1);
            for iSplit = 1:nSplit
                subRecImgs{iSplit} = subRecImgCmp{iSplit};
            end
            recImg = obj.circular_ola_(subRecImgs);
        end
                
        function recImg = stepImpl_parfor(obj,srcImg)
            obj.iteration = obj.iteration + 1;
            states_ = obj.states;
            nWorkers_ = obj.nWorkers;
            
            % Support function handles
            analyze     = @(x)   obj.Analyzer.step(x);
            manipulate  = @(x,s) obj.CoefsManipulator.step(x,s);                        
            synthesize  = @(x,s) obj.Synthesizer.step(x,s);
            extract_ols = @(c,s) obj.extract_ols_(c,s);
            padding_ola = @(c)   obj.padding_ola_(c);
            arr2vec     = @(a)   obj.arr2vec_(a);
            
            % Circular global padding
            srcImg_ = padarray(srcImg,obj.PadSize,'circular');
            
            % Overlap save split
            subImgs = obj.split_ols_(srcImg_);
            
            % Initialize
            nSplit = length(subImgs);
            subRecImgs = cell(nSplit,1);
            
            % Parallel processing
            usegpu_ = obj.UseGpu;
            parfor (iSplit=1:nSplit,nWorkers_)
                if usegpu_
                    subImg = gpuArray(subImgs{iSplit});
                else
                    subImg = subImgs{iSplit};
                end
                
                % Analyze
                [subCoefs, subScales] = analyze(subImg);
                
                % Extract significant coefs.
                coefs = extract_ols(subCoefs,subScales);
                
                % Process for coefficients
                state = states_{iSplit};
                if usegpu_ && iscell(state)
                    state = cellfun(@gpuArray,state,'UniformOutput',false);
                end
                [coefs,state] = manipulate(coefs,state);
                if usegpu_ && iscell(state)
                    state = cellfun(@gather,state,'UniformOutput',false);
                end
                states_{iSplit} = state;
                
                % Zero padding for convolution
                subCoefArray = padding_ola(coefs);
                
                % Synthesis
                [subCoefs,subScales] = arr2vec(subCoefArray);
                subRecImg = synthesize(subCoefs,subScales);
                
                if usegpu_
                    subRecImgs{iSplit} = gather(subRecImg);
                else
                    subRecImgs{iSplit} = subRecImg;
                end
            end
            
            % Update
            obj.states = states_;
            
            % Overlap add (Circular)
            recImg = obj.circular_ola_(subRecImgs);
        end
        
        
        function resetImpl(obj)
            obj.iteration = 0;
        end
    end
    
    methods (Access = private)
        
        function recImg = circular_ola_(obj,subRecImg)
            import saivdr.dictionary.utility.Direction
            verticalSplitFactor = obj.VerticalSplitFactor;
            horizontalSplitFactor = obj.HorizontalSplitFactor;
            stepsize = obj.refSubSize;
            overlap = size(subRecImg{1})-stepsize;
            recImg = zeros(obj.refSize+overlap,'like',subRecImg{1});
            % Overlap add
            iSplit = 0;
            tIdxHor = 0;
            for iHorSplit = 1:horizontalSplitFactor
                sIdxHor = tIdxHor + 1;
                tIdxHor= sIdxHor + stepsize(Direction.HORIZONTAL) - 1;
                eIdxHor = tIdxHor + overlap(Direction.HORIZONTAL);
                tIdxVer = 0;
                for iVerSplit = 1:verticalSplitFactor
                    iSplit = iSplit + 1;
                    sIdxVer = tIdxVer + 1;
                    tIdxVer = sIdxVer + stepsize(Direction.VERTICAL) - 1;
                    eIdxVer = tIdxVer + overlap(Direction.VERTICAL);
                    recImg(sIdxVer:eIdxVer,sIdxHor:eIdxHor) = ...
                        recImg(sIdxVer:eIdxVer,sIdxHor:eIdxHor) + ...
                        subRecImg{iSplit};
                end
            end
            % Folding
            recImg(1:overlap(Direction.VERTICAL),:) = ...
                recImg(1:overlap(Direction.VERTICAL),:) + ...
                recImg(end-overlap(Direction.VERTICAL)+1:end,:);
            recImg(:,1:overlap(Direction.HORIZONTAL)) = ...
                recImg(:,1:overlap(Direction.HORIZONTAL)) + ...
                recImg(:,end-overlap(Direction.HORIZONTAL)+1:end);
            % Cropping & circular shift
            recImg = circshift(imcrop(recImg,[1 1 fliplr(obj.refSize-1)]),...
                -overlap/2);
        end
        
        function subCoefArrayOut = padding_ola_(obj,subCoefArrayIn)
            import saivdr.dictionary.utility.Direction
            nChs = size(subCoefArrayIn,2);
            subPadSize_ = obj.subPadSize;
            subPadArrays_ = obj.subPadArrays;
            subCoefArrayOut = cell(size(subCoefArrayIn));
            for iCh = 1:nChs
                sRowIdx = subPadSize_(iCh,Direction.VERTICAL)+1;
                eRowIdx = sRowIdx + size(subCoefArrayIn{iCh},Direction.VERTICAL)-1;
                sColIdx = subPadSize_(iCh,Direction.HORIZONTAL)+1;
                eColIdx = sColIdx + size(subCoefArrayIn{iCh},Direction.HORIZONTAL)-1;
                if isa(subCoefArrayIn{iCh},'gpuArray')
                    subCoefArrayOut{iCh} = gpuArray(subPadArrays_{iCh});
                else
                    subCoefArrayOut{iCh} = subPadArrays_{iCh};
                end
                subCoefArrayOut{iCh}(sRowIdx:eRowIdx,sColIdx:eColIdx) ...
                    = subCoefArrayIn{iCh};
            end
        end
        
        
        function [coefsCrop,scalesCrop] = ...
                extract_ols_(obj,coefsSplit,scalesSplit)
            import saivdr.dictionary.utility.Direction
            verticalSplitFactor = obj.VerticalSplitFactor;
            horizontalSplitFactor = obj.HorizontalSplitFactor;
            refSubScales = obj.refScales*...
                diag(1./[verticalSplitFactor,horizontalSplitFactor]);
            nChs = size(refSubScales,1);
            %
            coefsCrop = cell(1,nChs);
            eIdx = 0;
            for iCh = 1:nChs
                stepsize = refSubScales(iCh,:);
                sIdx = eIdx + 1;
                eIdx = sIdx + prod(scalesSplit(iCh,:)) - 1;
                tmpVec = coefsSplit(sIdx:eIdx);
                tmpArray = reshape(tmpVec,scalesSplit(iCh,:));
                %
                offset = (scalesSplit(iCh,:) - refSubScales(iCh,:))/2;
                sRowIdx = offset(Direction.VERTICAL) + 1;
                eRowIdx = sRowIdx + stepsize(Direction.VERTICAL) - 1;
                sColIdx = offset(Direction.HORIZONTAL) + 1;
                eColIdx = sColIdx + stepsize(Direction.HORIZONTAL) - 1;
                %
                tmpArrayCrop = tmpArray(sRowIdx:eRowIdx,sColIdx:eColIdx);
                coefsCrop{iCh} = tmpArrayCrop;
            end
            if nargout > 1
                scalesCrop = refSubScales;
            end
        end
                

        function subImgs = split_ols_(obj,srcImg)
            import saivdr.dictionary.utility.Direction
            verticalSplitFactor = obj.VerticalSplitFactor;
            horizontalSplitFactor = obj.HorizontalSplitFactor;
            nSplit = prod(obj.SplitFactor);
            stepsize = obj.refSubSize;
            overlap = 2*obj.PadSize;
            %
            subImgs = cell(nSplit,1);
            idx = 0;
            for iHorSplit = 1:horizontalSplitFactor
                sColIdx = (iHorSplit-1)*stepsize(Direction.HORIZONTAL) + 1;
                eColIdx = iHorSplit*stepsize(Direction.HORIZONTAL) + ...
                    overlap(Direction.HORIZONTAL);
                for iVerSplit = 1:verticalSplitFactor
                    idx = idx + 1;
                    sRowIdx = (iVerSplit-1)*stepsize(Direction.VERTICAL) + 1;
                    eRowIdx = iVerSplit*stepsize(Direction.VERTICAL) + ...
                        overlap(Direction.VERTICAL);
                    subImgs{idx} = srcImg(sRowIdx:eRowIdx,sColIdx:eColIdx);
                end
            end
        end
        
    end
    
    methods (Access = private, Static)

        function [subCoefs,subScales] = arr2vec_(subCoefArray)
            nChs = size(subCoefArray,2);
            subScales = zeros(nChs,2);
            tmpCoefs_ = cell(1,nChs);
            for iCh = 1:nChs
                tmpArray = subCoefArray{iCh};
                subScales(iCh,:) = size(tmpArray);
                tmpCoefs_{iCh} = tmpArray(:).';
            end
            if isa(tmpCoefs_{1},'gpuArray')
                tmpCoefs_ = cellfun(@gather,tmpCoefs_,'UniformOutput',false);
                subCoefs = gpuArray(cell2mat(tmpCoefs_));
            else
                subCoefs = cell2mat(tmpCoefs_);
            end
        end
        
    end
end

