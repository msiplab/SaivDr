classdef NsoltDictionaryLearning < matlab.System
    %NSOLTDICTIONARYLERNING NSOLT dictionary learning
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2014-2020, Shogo MURAMATSU
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
        TrainingImages
        DecimationFactor = []
        NumbersOfPolyphaseOrder = [ 4 4 ]
        OptimizationFunction = @fminunc            
        SparseApproximation = 'IterativeHardThresholding'
        DictionaryUpdater = 'NsoltDictionaryUpdateGaFmin'
        OrderOfVanishingMoment = 1
        StepMonitor
        NumberOfDimensions = 'Two'
        NumberOfUnfixedInitialSteps = 0
        GradObj = 'off'
    end
    
    properties (Hidden, Transient)
        GradObjSet = ...
            matlab.system.StringSet({'on','off'});    
        SparseApproximationSet = ...
            matlab.system.StringSet({'IterativeHardThresholding','GradientPursuit'});
        DictionaryUpdaterSet = ...
            matlab.system.StringSet({'NsoltDictionaryUpdateGaFmin','NsoltDictionaryUpdateSgd'});
        NumberOfDimensionsSet = ...
            matlab.system.StringSet({'Two','Three'});
    end
    
    properties (Nontunable,PositiveInteger)
        NumberOfSymmetricChannel = 4
        NumberOfAntisymmetricChannel = 4
        NumberOfSparseCoefficients = 1
        NumberOfLevels = 1
        MaxIterOfHybridFmin = 400
        GenerationFactorForMus = 10
    end
    
    properties (Nontunable, Logical)    
        IsOptimizationOfMus = false        
        IsFixedCoefs = false
        IsRandomInit = false
        IsVerbose    = false
    end
    
    properties (Hidden)
        StdOfAngRandomInit    = 1e-2;
        PrbOfFlipMuRandomInit = 0      
        Step = 'Constant'
        StepStart = 1
        StepFinal = 1e-4
        AdaGradEta = 1e-2
        AdaGradEps = 1e-8
        GaAngInit = 'off'
    end
    
    properties(DiscreteState, PositiveInteger)
        Count
    end
   
    properties(DiscreteState, Logical)
        IsPreviousStepFixed
    end    
    
    properties(GetAccess=public,SetAccess=private)
        OvsdLpPuFb
    end
    
    properties (Access = protected, Nontunable)
        %nDecs 
        sparseAprx
        dicUpdate
        nImgs
    end

    methods
        function obj = NsoltDictionaryLearning(varargin)
            setProperties(obj,nargin,varargin{:}); 
            import saivdr.dictionary.nsoltx.NsoltFactory
            if strcmp(obj.NumberOfDimensions,'Three')
                %obj.nDecs = [ 2 2 2 ];
                if isempty(obj.DecimationFactor)
                    obj.DecimationFactor = [ 2 2 2 ];
                end
                obj.OvsdLpPuFb = NsoltFactory.createOvsdLpPuFb3dSystem(...
                    'DecimationFactor',obj.DecimationFactor,...%obj.nDecs,...
                    'NumberOfChannels', [obj.NumberOfSymmetricChannel obj.NumberOfAntisymmetricChannel ],...
                    'PolyPhaseOrder',obj.NumbersOfPolyphaseOrder,...
                    'NumberOfVanishingMoments',obj.OrderOfVanishingMoment,...
                    'OutputMode','ParameterMatrixSet');
            else
                %obj.nDecs = [ 2 2 ];
                if isempty(obj.DecimationFactor)
                    obj.DecimationFactor = [ 2 2 ];
                end                
                obj.OvsdLpPuFb = NsoltFactory.createOvsdLpPuFb2dSystem(...
                    'DecimationFactor',obj.DecimationFactor,...%obj.nDecs,...
                    'NumberOfChannels', [obj.NumberOfSymmetricChannel obj.NumberOfAntisymmetricChannel ],...
                    'PolyPhaseOrder',obj.NumbersOfPolyphaseOrder,...
                    'NumberOfVanishingMoments',obj.OrderOfVanishingMoment,...
                    'OutputMode','ParameterMatrixSet');
            end
            obj.nImgs = length(obj.TrainingImages);
        end
    end
    
    methods (Access=protected)
        
        function s = saveObjectImpl(obj)
            % Call the base class method
            s = saveObjectImpl@matlab.System(obj);
            % Save the child System objects
            s.OvsdLpPuFb = matlab.System.saveObject(obj.OvsdLpPuFb);
            s.sparseAprx = matlab.System.saveObject(obj.sparseAprx);
            s.dicUpdate = matlab.System.saveObject(obj.dicUpdateAprx);            
            % Save the protected & private properties
            s.nImgs = obj.nImgs;
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            % Load protected and private properties
            obj.nImgs = s.nImgs;
            % Load the child System objects
            obj.sparseAprx = matlab.System.loadObject(s.sparseAprx);
            obj.dicUpdate = matlab.System.loadObject(s.dicUpdateAprx);                        
            obj.OvsdLpPuFb = matlab.System.loadObject(s.OvsdLpPuFb);
            % Call base class method to load public properties
            loadObjectImpl@matlab.System(obj,s,wasLocked);             
        end
        
        function flag = isInactivePropertyImpl(obj,propertyName)
            if strcmp(propertyName,'NumberOfUnfixedInitialSteps')
                flag = ~obj.IsFixedCoefs; 
            elseif strcmp(propertyName,'StdOfAngRandomInit') || ...
                    strcmp(propertyName,'PrbOfFlipMuRandomInit')
                flag = ~obj.IsRandomInit;
            elseif strcmp(propertyName,'Step') || ...
                    strcmp(propertyName,'StepStart') || ...
                    strcmp(propertyName,'StepFinal') || ...
                strcmp(propertyName,'GaAngInit')
                flag = ~strcmp(obj.DictionaryUpdater,'NsoltDictionaryUpdateSgd');
            else
                flag = false;
            end
        end
        
        function validatePropertiesImpl(obj)
            if isempty(obj.TrainingImages)
                error('Source images should be provided');
            elseif ~iscell(obj.TrainingImages)
                error('Source images should be provided as cell data');
            end
        end
        
        function setupImpl(obj, ~, ~)
            
            obj.Count = 1;
            obj.IsPreviousStepFixed = obj.IsFixedCoefs;
            
            % Instantiation of synthesizer and analyzer
            import saivdr.dictionary.nsoltx.*
            if strcmp(obj.NumberOfDimensions,'Three')
                synthesizer = NsoltFactory.createSynthesis3dSystem(...
                    obj.OvsdLpPuFb,...
                    'IsCloneLpPuFb',false);    
                analyzer = NsoltFactory.createAnalysis3dSystem(...
                    obj.OvsdLpPuFb,...
                    'IsCloneLpPuFb',false);    
                analyzer.NumberOfLevels = obj.NumberOfLevels;
            else
                synthesizer = NsoltFactory.createSynthesis2dSystem(...
                    obj.OvsdLpPuFb,...
                    'IsCloneLpPuFb',false);                    
                analyzer = NsoltFactory.createAnalysis2dSystem(...
                    obj.OvsdLpPuFb,...
                    'IsCloneLpPuFb',false);
                analyzer.NumberOfLevels = obj.NumberOfLevels;                
            end
            
            % Instantiation of sparse approximation
            if strcmp(obj.SparseApproximation,'GradientPursuit')
                import saivdr.sparserep.GradientPursuit
                obj.sparseAprx = GradientPursuit(...
                    'Dictionary', { synthesizer, analyzer },...
                    'StepMonitor',obj.StepMonitor);
            else
                import saivdr.sparserep.IterativeHardThresholding
                obj.sparseAprx = IterativeHardThresholding(...
                    'Dictionary', { synthesizer, analyzer },...                    
                    'StepMonitor',obj.StepMonitor);
            end
            
            % Instantiation of Dictionary Updater
            if strcmp(obj.DictionaryUpdater,'NsoltDictionaryUpdateSgd')
                import saivdr.dictionary.nsoltx.design.NsoltDictionaryUpdateSgd
                obj.dicUpdate = NsoltDictionaryUpdateSgd(...
                    'TrainingImages', obj.TrainingImages,...
                    'NumberOfLevels',obj.NumberOfLevels,...
                    'GenerationFactorForMus',obj.GenerationFactorForMus,...
                    'IsFixedCoefs',obj.IsFixedCoefs,...                    
                    'IsVerbose',obj.IsVerbose,...
                    'GradObj',obj.GradObj,...
                    'Step',obj.Step,...
                    'StepStart',obj.StepStart,...
                    'StepFinal',obj.StepFinal,...
                    'AdaGradEta',obj.AdaGradEta,...
                    'AdaGradEps',obj.AdaGradEps,...                    
                    'GaAngInit',obj.GaAngInit);
            else
                import saivdr.dictionary.nsoltx.design.NsoltDictionaryUpdateGaFmin
                obj.dicUpdate = NsoltDictionaryUpdateGaFmin(...
                    'TrainingImages', obj.TrainingImages,...
                    'NumberOfLevels',obj.NumberOfLevels,...
                    'OptimizationFunction',obj.OptimizationFunction,...
                    'MaxIterOfHybridFmin',obj.MaxIterOfHybridFmin,...
                    'GenerationFactorForMus',obj.GenerationFactorForMus,...
                    'IsFixedCoefs',obj.IsFixedCoefs,...
                    'IsVerbose',obj.IsVerbose,...
                    'GradObj',obj.GradObj);
            end
            
            % Random initialization
            if obj.IsRandomInit
                %                
                sdv = obj.StdOfAngRandomInit;
                angles_ = get(obj.OvsdLpPuFb,'Angles');
                sizeAngles = size(angles_);
                angles_ = angles_ + sdv*randn(sizeAngles);
                set(obj.OvsdLpPuFb,'Angles',angles_);
                %
                thr = obj.PrbOfFlipMuRandomInit;
                mus_ = get(obj.OvsdLpPuFb,'Mus');
                sizeMus = size(mus_);
                mus_ = mus_ .* (2*(rand(sizeMus)>=thr)-1);
                set(obj.OvsdLpPuFb,'Mus',mus_);                
                %
            end
            
        end
        
        function resetImpl(obj)
            obj.Count = 1;
            obj.IsPreviousStepFixed = obj.IsFixedCoefs;
        end
        
        function [ lppufb, fval, exitflag ] = stepImpl(obj,options,isOptMus)
            
            if isa(options,'optim.options.Fminunc') || ... % TODO for SGD
                    isa(options,'optim.options.Fmincon')
                options = optimoptions(options,'GradObj',obj.GradObj);
            end
            
            if isempty(isOptMus) || ...
                    ~strcmp(char(obj.OptimizationFunction),'ga')
                isOptMus = false;
            end
                        
            % Sparse Approximation
            sprsCoefs   = cell(obj.nImgs,1);
            setOfScales = cell(obj.nImgs,1);
            obj.sparseAprx.NumberOfSparseCoefficients = obj.NumberOfSparseCoefficients;
            for iImg = 1:obj.nImgs
                set(obj.StepMonitor,'SourceImage',obj.TrainingImages{iImg});
                [~, sprsCoefs{iImg}, setOfScales{iImg}] = ...
                    obj.sparseAprx.step(obj.TrainingImages{iImg});
            end
            
            % Dictionary Update
            set(obj.dicUpdate,'IsOptimizationOfMus',isOptMus);
            set(obj.dicUpdate,'SparseCoefficients',sprsCoefs);
            set(obj.dicUpdate,'SetOfScales',setOfScales);
            if strcmp(obj.DictionaryUpdater,'NsoltDictionaryUpdateSgd') && ...
                    obj.Count > 1
                set(obj.dicUpdate,'GaAngInit','off');
            end
            if obj.Count <= obj.NumberOfUnfixedInitialSteps
                cloneDicUpdate = clone(obj.dicUpdate);
                set(cloneDicUpdate,'IsFixedCoefs',false);
                [ lppufb, fval, exitflag ] = ...
                    step(cloneDicUpdate,obj.OvsdLpPuFb,options);
                obj.IsPreviousStepFixed = get(cloneDicUpdate,'IsFixedCoefs');
                delete(cloneDicUpdate);
            else
                [ lppufb, fval, exitflag ] = ...
                    step(obj.dicUpdate,obj.OvsdLpPuFb,options);
                obj.IsPreviousStepFixed = get(obj.dicUpdate,'IsFixedCoefs');
            end
            angs = get(lppufb,'Angles');
            mus  = get(lppufb,'Mus');
            set(obj.OvsdLpPuFb,'Angles',angs,'Mus',mus);

            % Increment Count
            obj.Count = obj.Count + 1;
                    
        end
        
        function N = getNumInputsImpl(~)
            N = 2; 
        end
        
        function N = getNumOutputsImpl(~)
            N = 3;
        end
     
    end

end

