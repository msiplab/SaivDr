classdef AbstNsoltDesigner < matlab.System %#~codegen
    %ABSTNSOLTDESIGNER Abstract class of NSOLT Designer
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
    
    properties
        TrainingImages
        SparseCoefficients
        SetOfScales
        %
        IsOptimizationOfMus    = false
        GenerationFactorForMus = 2
        GradObj = 'off'
    end
    
    properties (Hidden, Transient)
        GradObjSet = ...
            matlab.system.StringSet({'on','off'});
    end
    
    properties (Nontunable, Access = protected)
        sizeAngles
        sizeMus
    end
    
    methods
        function obj = AbstNsoltDesigner(varargin)
            setProperties(obj,nargin,varargin{:});
        end
    end
    
    methods (Abstract = true)
        value  = costFcnAng(obj, lppufb, angles)
        value  = costFcnMus(obj, lppufb, mus)
        value  = isConstrained(obj, lppufb)
        [c,ce] = nonlconFcn(obj, lppufb, angles)
    end
    
    methods (Access = protected)
        
        function flag = isInactiveSubPropertyImpl(~,propertyName)
            if strcmp(propertyName,'GenerationFactorForMus')
                flag = ~strcmp(propertyName,'IsOptimizationOfMus');
            else
                flag = false;
            end
        end
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@matlab.System(obj);
            s.sizeAngles = obj.sizeAngles;
            s.sizeMus    = obj.sizeMus;
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            obj.sizeAngles = s.sizeAngles;
            obj.sizeMus    = s.sizeMus;
            loadObjectImpl@matlab.System(obj,s,wasLocked);
        end
        
        function setupImpl(obj,lppufb_,~)
            angles_ = get(lppufb_,'Angles');
            obj.sizeAngles = size(angles_);
            mus_ = get(lppufb_,'Mus');
            obj.sizeMus = size(mus_);
        end
        
        function resetImpl(~)
        end
        
        function N = getNumInputsImpl(~)
            % Specify number of System inputs
            N = 2;
        end
        
        function N = getNumOutputsImpl(~)
            % Specify number of System outputs
            N = 3;
        end
        
        function lppufb = stepOptMus(obj,lppufb,options)
            if obj.IsOptimizationOfMus
                optionsMus = ...
                    gaoptimset(options,'PopulationType','bitstring');
                %
                populationSize_ = gaoptimget(optionsMus,'PopulationSize');
                nMus = prod(obj.sizeMus);
                mus_ = get(lppufb,'Mus');
                initMus = (rand(populationSize_,nMus)>0.5);
                initMus(1,:) = (mus_(:).'+1)/2;
                optionsMus = ...
                    gaoptimset(optionsMus,'InitialPopulation',initMus);
                %
                optionsMus = ...
                    gaoptimset(optionsMus,...
                    'Generations', obj.GenerationFactorForMus*nMus);
                optionsMus = ...
                    gaoptimset(optionsMus,...
                    'StallGenLimit',(obj.GenerationFactorForMus-1)*nMus);
                problemMus.fitnessfcn = @(x)costFcnMus(obj,lppufb, x);
                problemMus.nvars    = nMus;
                problemMus.Aineq    = [];
                problemMus.bineq    = [];
                problemMus.Aeq      = [];
                problemMus.beq      = [];
                problemMus.lb       = [];
                problemMus.ub       = [];
                problemMus.nonlcon  = [];
                problemMus.rngstate = [];
                problemMus.intcon   = [];
                problemMus.solver   = 'ga';
                problemMus.options  = optionsMus;
                optMus = ga(problemMus);
                set(lppufb,'Mus',2*(reshape(optMus,obj.sizeMus))-1);
            end
        end
    end
end
