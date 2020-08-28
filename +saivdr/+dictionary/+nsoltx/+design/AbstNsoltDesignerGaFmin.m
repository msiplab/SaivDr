classdef AbstNsoltDesignerGaFmin < ...
        saivdr.dictionary.nsoltx.design.AbstNsoltDesigner
    %ABSTNSOLTDESIGNERGAFMIN Abstract class of NSOLT Designer
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
                problem.x0 = initAngs;
                problem.options = options;                
                problem.objective = @(x)costFcnAng(obj,lppufb,x);                
                if strcmp(char(obj.OptimizationFunction),'fmincon')
                    problem.solver = 'fmincon';
                    problem.nonlcon = @(x)nonlconFcn(obj,lppufb,x);
                else
                    problem.solver = 'fminunc';
                end
            end
            %[optAngs, fval, exitflag] = obj.OptimizationFunction(problem);
            [optAngs, fval, exitflag] = ...
                eval(sprintf('%s(problem)',problem.solver));
            set(lppufb,'Angles',reshape(optAngs,obj.sizeAngles));
        end
        
    end
    
    methods (Access = private)
        
        function value = setHybridFmincon_(obj,options)
            hybridopts = optimoptions(@fmincon);
            if gaoptimget(options,'UseParallel')
                hybridopts = optimoptions(hybridopts,...
                    'UseParallel',true);
                %
                strver = version('-release');
                if strcmp(strver,'2016b') || strcmp(strver,'2017a')
                    % Bug ID: 1563824
                    hybridopts = optimoptions(hybridopts,...
                        ...'Algorithm','sqp-legacy');...
                        'Algorithm','interior-point');
                end
                if verLessThan('optim','7.5')
                    hybridopts = optimoptions(hybridopts,...
                        'GradObj', obj.GradObj);
                    hybridopts = optimoptions(hybridopts,...
                        'GradConstr','off');
                else
                    hybridopts = optimoptions(hybridopts,...
                        'SpecifyObjectiveGradient',...
                        strcmp(obj.GradObj,'on'));
                    hybridopts = optimoptions(hybridopts,...
                        'SpecifyConstraintGradient',false);
                end
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
            if verLessThan('optim','7.5')
                hybridopts = optimoptions(hybridopts,'GradObj',...
                    obj.GradObj);
            else
                hybridopts = optimoptions(hybridopts,...
                    'SpecifyObjectiveGradient',strcmp(obj.GradObj,'on'));
                
            end
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
            if verLessThan('optim','7.5')
                hybridopts = optimoptions(hybridopts,...
                    'GradObj', obj.GradObj);
            else
                hybridopts = optimoptions(hybridopts,...
                    'SpecifyObjectiveGradient',...
                    strcmp(obj.GradObj,'on'));
            end
            value = ...
                gaoptimset(options,'HybridFcn',...
                {@fminunc,hybridopts});
        end
        
    end
end

