classdef NsoltDictionaryLearning < matlab.System
    %NSOLTDICTIONARYLERNING NSOLT dictionary learning
    %
    % SVN identifier:
    % $Id: NsoltDictionaryLearning.m 683 2015-05-29 08:22:13Z sho $
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
        SourceImages
        NumberOfSymmetricChannel = 4
        NumberOfAntisymmetricChannel = 4
        NumbersOfPolyphaseOrder = [ 4 4 ]
        NumberOfSparseCoefficients = 1
        NumberOfTreeLevels = 1
        OptimizationFunction = @fminunc            
        MaxIterOfHybridFmin = 400
        GenerationFactorForMus = 10
        IsOptimizationOfMus = false
        OrderOfVanishingMoment = 1
        IsFixedCoefs = false
        IsRandomInit = false
        SparseCoding = 'IterativeHardThresholding'
        StepMonitor
    end
    
    properties(GetAccess=public,SetAccess=private)
         LpPuFb2d
    end
    
    properties (Access = protected, Nontunable)
        nDecs = [ 2 2 ]
        sparseCoder
        dicUpdate
        nImgs
    end
    
    properties (Hidden, Transient)
        SparseCodingSet = ...
            matlab.system.StringSet({'IterativeHardThresholding','GradientPursuit'});
    end
    
    methods
        function obj = NsoltDictionaryLearning(varargin)
            setProperties(obj,nargin,varargin{:}); 
            import saivdr.dictionary.nsolt.NsoltFactory
            obj.LpPuFb2d = NsoltFactory.createOvsdLpPuFb2dSystem(...
                'DecimationFactor',obj.nDecs,...
                'NumberOfChannels', [obj.NumberOfSymmetricChannel obj.NumberOfAntisymmetricChannel ],...
                'PolyPhaseOrder',obj.NumbersOfPolyphaseOrder,...
                'NumberOfVanishingMoments',obj.OrderOfVanishingMoment,...
                'OutputMode','ParameterMatrixSet');
            obj.nImgs = length(obj.SourceImages);
        end
    end
    
    methods (Access=protected)
        
        function validatePropertiesImpl(obj)
            if isempty(obj.SourceImages)
                error('Source images should be provided');
            elseif ~iscell(obj.SourceImages)
                error('Source images should be provided as cell data');
            end
        end
        
        function setupImpl(obj, ~, ~)
            if strcmp(obj.SparseCoding,'GradientPursuit')
                import saivdr.sparserep.GradientPursuit
                obj.sparseCoder = GradientPursuit(...
                    'NumberOfTreeLevels',obj.NumberOfTreeLevels,...
                    'StepMonitor',obj.StepMonitor);
            else
                import saivdr.sparserep.IterativeHardThresholding
                obj.sparseCoder = IterativeHardThresholding(...
                    'NumberOfTreeLevels',obj.NumberOfTreeLevels,...
                    'StepMonitor',obj.StepMonitor);
            end
            import saivdr.dictionary.nsolt.design.NsoltDictionaryUpdate
            obj.dicUpdate = NsoltDictionaryUpdate(...
                'SourceImages', obj.SourceImages,...
                'NumberOfTreeLevels',obj.NumberOfTreeLevels,...
                'OptimizationFunction',obj.OptimizationFunction,...
                'MaxIterOfHybridFmin',obj.MaxIterOfHybridFmin,...
                'GenerationFactorForMus',obj.GenerationFactorForMus,...
                'IsFixedCoefs',obj.IsFixedCoefs);
            
            % Random initialization
            if obj.IsRandomInit
                angles_ = getAngles(obj.LpPuFb2d);
                sizeAngles = size(angles_);
                angles_ = angles_ + pi*randn(sizeAngles);
                obj.LpPuFb2d  = setAngles(obj.LpPuFb2d,angles_);
            end
            
        end
        
        function resetImpl(~)
        end
        
        function [ lppufb, fval, exitflag ] = stepImpl(obj,options,isOptMus)

            % Sparse Coding 
            import saivdr.dictionary.nsolt.*
            synthesizer = NsoltFactory.createSynthesisSystem(obj.LpPuFb2d);
            analyzer = NsoltFactory.createAnalysisSystem(obj.LpPuFb2d);
            clnSparseCoder = clone(obj.sparseCoder);
            set(clnSparseCoder,'Synthesizer',synthesizer);
            set(clnSparseCoder,'AdjOfSynthesizer',analyzer);
            sprsCoefs   = cell(obj.nImgs,1);
            setOfScales = cell(obj.nImgs,1);
            for iImg = 1:obj.nImgs
                [~, sprsCoefs{iImg}, setOfScales{iImg}] = ...
                    step(clnSparseCoder,...
                    obj.SourceImages{iImg},obj.NumberOfSparseCoefficients);
            end
            
            % Dictionary Update
            set(obj.dicUpdate,'IsOptimizationOfMus',isOptMus);
            set(obj.dicUpdate,'SparseCoefficients',sprsCoefs);
            set(obj.dicUpdate,'SetOfScales',setOfScales);
            [ obj.LpPuFb2d, fval, exitflag ] = ...
                step(obj.dicUpdate,obj.LpPuFb2d,options);

            % Output
            lppufb = obj.LpPuFb2d;
        end
        
        function N = getNumInputsImpl(~)
            N = 2; 
        end
        
        function N = getNumOutputsImpl(~)
            N = 3;
        end
     
    end

end
