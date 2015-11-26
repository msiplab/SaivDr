classdef AbstNsoltDesignerGaFmin < ...
        saivdr.dictionary.nsoltx.design.AbstNsoltDesigner
    %ABSTNSOLTDESIGNER Abstract class of NSOLT Designer
    %
    % SVN identifier:
    % $Id: AbstNsoltDesignerGaFmin.m 868 2015-11-25 02:33:11Z sho $
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
        OptimizationFunction   = @fminunc            
    end
        
    properties
        MaxIterOfHybridFmin    = 200;
    end
    
    methods
        function obj = AbstNsoltDesignerGaFmin(varargin)
            obj = obj@saivdr.dictionary.nsoltx.design.AbstNsoltDesigner(...
                varargin{:});
        end
    end
    
    methods (Access = protected)
        
        function flag = isInactiveSubPropertyImpl(obj,propertyName)
            if strcmp(propertyName,'MaxIterOfHybridFmin') 
                flag = (obj.OptimizationFunction ~= @ga);
            else
                flag = isInactiveSubPropertyImpl...
                    @saivdr.dictionary.nsoltx.design.AbstNsoltDesigner(...
                    obj,propertyName);
            end            
        end        
        
        function [ lppufb, fval, exitflag ] = stepImpl(obj,lppufb_,options)

            % Cloning
            lppufb = clone(lppufb_);
            
            % Optimization of Mus            
            lppufb = stepOptMus(obj,lppufb,options);
            
            % Optimization of Angs            
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
        
    end
    
    methods (Access = private)
        
        function value = setHybridFmincon_(obj,options)
            hybridopts = optimoptions(@fmincon);
            if strcmp(gaoptimget(options,'UseParallel'),'always')
                hybridopts = optimoptions(hybridopts,...
                    'UseParallel','always');
                hybridopts = optimoptions(hybridopts,...
                    'GradObj', obj.GradObj);
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
            hybridopts = optimoptions(hybridopts,'GradObj',...
                obj.GradObj);
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
            hybridopts = optimoptions(hybridopts,'GradObj',...
                obj.GradObj);            
            value = ...
                gaoptimset(options,'HybridFcn',...
                {@fminunc,hybridopts});
        end
        
    end
end
