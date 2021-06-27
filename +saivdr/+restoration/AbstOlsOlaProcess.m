classdef (Abstract) AbstOlsOlaProcess < matlab.System
    %ABSTOLSOLAPROCESS Abstract class of OLS/OLA wrapper for analysis and synthesis system
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
    
    properties (Abstract, Access = protected, Constant = true)
        DATA_DIMENSION
    end
    
    methods (Abstract)
        [coefs,scales] = getCoefficients(obj)
    end
    
    properties (Nontunable, Logical, Hidden)
        Debug = false
    end
    
    methods (Abstract, Access = protected)
        recImg = circular_ola_(obj,subRecImg)
        subCoefArrayOut = padding_ola_(obj,subCoefArrayIn)
        [coefsCrop,scalesCrop] = extract_ols_(obj,coefsSplit,scalesSplit)
        subImgs =  split_ols_(obj,srcImg)
        setupSplitFactor(obj)
    end
    
    properties (Nontunable)
        Analyzer
        Synthesizer
        BoundaryOperation
        PadSize = []
        SplitFactor = []
        CoefsManipulator = []
        InitialState
    end
    
    properties (Logical)
        UseGpu = false
        UseParallel = false
        IsIntegrityTest = true
    end
    
    properties (DiscreteState)
        iteration
    end
    
    properties (Hidden)
        States
    end
    
    properties (Hidden, Transient)
        BoundaryOperationSet = ...
            matlab.system.StringSet({'Circular'});
    end
    
    properties (Access = protected, Nontunable)
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
        function obj = AbstOlsOlaProcess(varargin)
            import saivdr.dictionary.utility.Direction
            setProperties(obj,nargin,varargin{:})
            if isempty(obj.CoefsManipulator)
                import saivdr.restoration.CoefsManipulator
                obj.CoefsManipulator = CoefsManipulator();
            end
            if isempty(obj.PadSize)
                obj.PadSize = zeros(1,obj.DATA_DIMENSION);
            end
        end
        
        function coefsSet = analyze(obj,srcImg)
            % Preperation
            obj.setupSplitFactor();
            splitFactor = obj.SplitFactor;
            nSplit = prod(obj.SplitFactor);
            obj.Analyzer.release();
            refAnalyzer = obj.Analyzer.clone();
            [~,refScales_] = refAnalyzer.step(srcImg);
            
            % Parameters
            obj.refSize = size(srcImg);
            obj.refSubSize = obj.refSize*diag(1./splitFactor);
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
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@matlab.System(obj);
            s.nWorkers = obj.nWorkers;
            s.refSize = obj.refSize;
            s.refScales = obj.refScales;
            s.subPadSize = obj.subPadSize;
            s.refSubSize = obj.refSubSize;
            s.subPadArrays = obj.subPadArrays;
            %s.States = obj.States;
            if isLocked(obj)
                s.iteration = obj.iteration;
            end
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            if wasLocked
                obj.iteration = s.iteration;
            end
            %obj.States = s.States;
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
            if obj.UseGpu && gpuDeviceCount < 1 
                obj.UseGpu = false;
                warning('No GPU is available.')
            end
            
            % Preperation
            splitFactor = obj.SplitFactor;
            nSplit = prod(splitFactor);
            
            % Analyzers
            obj.BoundaryOperation = obj.Analyzer.BoundaryOperation;
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
            obj.refSubSize = obj.refSize*diag(1./splitFactor);
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
                    if obj.Debug
                        obj.nWorkers = 'debug';
                    else
                        obj.nWorkers = Inf;
                    end
                    obj.isSpmd = false;
                end
            else
                obj.nWorkers = 0;
                obj.isSpmd = false;
            end
            
            %Evaluate
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
                subScale = refScales_(iCh,:)*diag(1./splitFactor);
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
                obj.States = num2cell(zeros(nSplit,1,'like',srcImg));
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
            obj.States = cell(nSplit,1);
            if isempty(obj.InitialState)
                for iSplit = 1:nSplit
                    state = num2cell(zeros(1,nChs,'like',srcImg));
                    obj.States{iSplit} = state;
                end
            elseif isscalar(obj.InitialState) && ~iscell(obj.InitialState)
                for iSplit = 1:nSplit
                    state = num2cell(...
                        cast(obj.InitialState,'like',srcImg)*...
                        ones(1,nChs,'like',srcImg));
                    obj.States{iSplit} = state;
                end
            else
                for iSplit = 1:nSplit
                    initState = obj.InitialState{iSplit};
                    state = cellfun(@(x) cast(x,'like',srcImg),...
                        initState,'UniformOutput',false);
                    obj.States{iSplit} = state;
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
            arr2vec     = @(a)   obj.arr2vec_(a,obj.DATA_DIMENSION);
            
            % Circular global padding
            srcImg_ = padarray(srcImg,obj.PadSize,'circular');
            
            % Overlap save split
            subImgs = obj.split_ols_(srcImg_);
            
            % Initialize
            nSplit = length(subImgs);
            subImgCmp = Composite(nSplit);
            stateCmp = Composite(nSplit);
            for iSplit=1:nSplit
                subImgCmp{iSplit} = subImgs{iSplit};
                stateCmp{iSplit} = obj.States{iSplit};
            end
            
            % Parallel processing
            usegpu_ = obj.UseGpu;
            spmd(nSplit)
                %iSplit = labindex;
                if usegpu_
                    subImg = gpuArray(subImgCmp);
                else
                    subImg = subImgCmp;
                end
                
                % Analyze
                [subCoefs, subScales] = analyze(subImg);
                
                % Extract significant coefs.
                coefs = extract_ols(subCoefs,subScales);
                
                % Process for coefficients
                state = stateCmp;
                if usegpu_ && iscell(state)
                    state = cellfun(@gpuArray,state,'UniformOutput',false);
                end
                coefs = manipulate(coefs,state);
                
                % Zero padding for convolution
                subCoefArray = padding_ola(coefs);
                if  usegpu_ 
                    coefs = cellfun(@gather,coefs,'UniformOutput',false);
                end
                stateCmp = coefs;
                
                % Synthesis
                [subCoefs,subScales] = arr2vec(subCoefArray);
                subRecImg = synthesize(subCoefs,subScales);
                
                if usegpu_
                    subRecImgCmp = gather(subRecImg);
                else
                    subRecImgCmp = subRecImg;
                end
            end
            
            % Update
            for iSplit=1:nSplit
                obj.States{iSplit} = stateCmp{iSplit};
            end
            
            % Overlap add (Circular)
            subRecImgs = cell(nSplit,1);
            for iSplit = 1:nSplit
                subRecImgs{iSplit} = subRecImgCmp{iSplit};
            end
            recImg = obj.circular_ola_(subRecImgs);
        end
        
        function recImg = stepImpl_parfor(obj,srcImg)
            obj.iteration = obj.iteration + 1;
            
            % Support function handles
            analyze     = @(x)   obj.Analyzer.step(x);
            manipulate  = @(x,s) obj.CoefsManipulator.step(x,s);
            synthesize  = @(x,s) obj.Synthesizer.step(x,s);
            extract_ols = @(c,s) obj.extract_ols_(c,s);
            padding_ola = @(c)   obj.padding_ola_(c);
            arr2vec     = @(a)   obj.arr2vec_(a,obj.DATA_DIMENSION);
            
            % Circular global padding
            srcImg_ = padarray(srcImg,obj.PadSize,'circular');
            
            % Overlap save split
            subImgs = obj.split_ols_(srcImg_);
            
            % Initialize
            nSplit = length(subImgs);
            states_ = obj.States;
            
            % Parallel processing
            nWorkers_ = obj.nWorkers;
            subRecImgs = cell(nSplit,1);            
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
                coefs = manipulate(coefs,state);
                
                % Zero padding for convolution
                subCoefArray = padding_ola(coefs);
                if usegpu_ 
                    coefs = cellfun(@gather,coefs,'UniformOutput',false);
                end
                states_{iSplit} = coefs;
                
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
            obj.States = states_;
            
            % Overlap add (Circular)
            recImg = obj.circular_ola_(subRecImgs);
        end
        
        function resetImpl(obj)
            obj.iteration = 0;
        end
    end
    
    methods (Access = protected, Static)
        
        function [subCoefs,subScales] = arr2vec_(subCoefArray,ndim)
            import saivdr.restoration.AbstOlsOlaProcess
            nChs = size(subCoefArray,2);
            subScales = zeros(nChs,ndim);
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

