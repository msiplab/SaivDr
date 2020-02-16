classdef AbstNsoltDesigner < matlab.System %#~codegen
    %ABSTNSOLTDESIGNER Abstract class of NSOLT Designer
    %
    % SVN identifier:
    % $Id: AbstNsoltDesigner.m 683 2015-05-29 08:22:13Z sho $
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
        OptimizationFunction   = @fminunc            
    end
    
    properties
        IsOptimizationOfMus    = false;
        %
        MaxIterOfHybridFmin    = 200;
        GenerationFactorForMus = 2;
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
        
        function flag = isInactiveSubPropertyImpl(obj,propertyName)
            if strcmp(propertyName,'MaxIterOfHybridFmin')
                flag = (obj.OptimizationFunction ~= @ga);
            elseif strcmp(propertyName,'GenerationFactorForMus')
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
        
        function setupImpl(obj,lppufb,~)
            angles_ = get(lppufb,'Angles');
            obj.sizeAngles = size(angles_);
            mus_ = get(lppufb,'Mus');
            obj.sizeMus = size(mus_);
        end
        
        function resetImpl(~)
        end        
        
        function [ lppufb, fval, exitflag ] = stepImpl(obj,lppufb,options)

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
            if strcmp(char(obj.OptimizationFunction),'ga')
                populationSize_ = gaoptimget(options,'PopulationSize');
                nAngs = prod(obj.sizeAngles); % Number of angles
                angles_ = get(lppufb,'Angles');
                initAngs = repmat(angles_(:).',populationSize_,1)...
                    +2*pi*(rand(populationSize_,nAngs)-0.5);
                initAngs(1,:) = angles_(:).';
                options = ...
                    gaoptimset(options,'InitialPopulation',initAngs);
                %
                if isConstrained(obj,lppufb)
                    nonlcon = @(x)nonlconFcn(obj,lppufb,x);
                    options = setHybridFmincon_(obj,options);
                else
                    nonlcon = [];
                    options = setHybridFminunc_(obj,options);
                end
                %
                problem.fitnessfcn = @(x)costFcnAng(obj,lppufb,x);
                problem.nvars    = nAngs;
                problem.Aineq    = [];
                problem.bineq    = [];
                problem.Aeq      = [];
                problem.beq      = [];
                problem.lb       = [];
                problem.ub       = [];
                problem.nonlcon  = nonlcon;
                problem.rngstate = [];
                problem.intcon   = [];
                problem.solver   = 'ga';
                problem.options  = options;
            else
                initAngs = get(lppufb,'Angles');
                %
                if strcmp(char(obj.OptimizationFunction),'fmincon')
                    problem = createOptimProblem('fmincon',...
                        'objective',@(x)costFcnAng(obj,lppufb,x),...
                        'x0',initAngs,...
                        'nonlcon', @(x)nonlconFcn(obj,lppufb,x),...
                        'options', options);
                else
                    problem = createOptimProblem('fminunc',...
                        'objective',@(x)costFcnAng(obj,lppufb,x),...                        
                        'x0',initAngs,...
                        'options', options);
                end
            end
            [optAngs, fval, exitflag] = obj.OptimizationFunction(problem);
            set(lppufb,'Angles',reshape(optAngs,obj.sizeAngles));
        end
        
        function N = getNumInputsImpl(~)
            % Specify number of System inputs
            N = 2; 
        end
        
        function N = getNumOutputsImpl(~)
            % Specify number of System outputs
            N = 3; 
        end
        
    end
    
    methods (Access = private)
        
        function value = setHybridFmincon_(obj,options)
            hybridopts = optimoptions(@fmincon);
            if strcmp(gaoptimget(options,'UseParallel'),'always')
                hybridopts = optimoptions(hybridopts,...
                    'UseParallel','always');
                hybridopts = optimoptions(hybridopts,...
                    'GradObj','off');
                hybridopts = optimoptions(hybridopts,...
                    'GradConstr','off');
                %hybridopts = optimoptions(hybridopts,...
                %    'TolCon',0);
            end
            if ~isempty(gaoptimget(options,'PlotFcn'))
                hybridopts = optimoptions(hybridopts,...
                    'PlotFcns',@optimplotfval);
            end
            hybridopts = optimoptions(hybridopts,'Display',...
                gaoptimget(options,'Display'));
            hybridopts = optimoptions(hybridopts,'MaxFunEvals',...
                2000*prod(obj.sizeAngles));
            hybridopts = optimoptions(hybridopts,'MaxIter',...
                obj.MaxIterOfHybridFmin);
            value = ...
                gaoptimset(options,'HybridFcn',...
                {@fmincon,hybridopts});
        end
        
        function value = setHybridFminunc_(obj,options)
            hybridopts = optimoptions(@fminunc);
            if ~isempty(gaoptimget(options,'PlotFcn'))
                hybridopts = optimoptions(hybridopts,...
                    'PlotFcns',@optimplotfval);
            end
            hybridopts = optimoptions(hybridopts,'Display',...
                gaoptimget(options,'Display'));
            hybridopts = optimoptions(hybridopts,'MaxFunEvals',...
                2000*prod(obj.sizeAngles));
            hybridopts = optimoptions(hybridopts,'MaxIter',...
                obj.MaxIterOfHybridFmin);
            value = ...
                gaoptimset(options,'HybridFcn',...
                {@fminunc,hybridopts});
        end
        
    end
end
