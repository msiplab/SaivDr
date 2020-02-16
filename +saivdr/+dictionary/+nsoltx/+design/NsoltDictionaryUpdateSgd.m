classdef NsoltDictionaryUpdateSgd < ...
        saivdr.dictionary.nsoltx.design.AbstNsoltDesigner %#~codegen
    %NSOLTDICTIONARYUPDATESGD Update step of NSOLT dictionary learning
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2015-2020, Shogo MURAMATSU and Genki FUJII
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
        Step = 'Reciprocal'
    end
    
    properties (Hidden, Transient)
        StepSet = ...
            matlab.system.StringSet(...
            {'LineSearch','Reciprocal','Constant','Exponential','AdaGrad'});
        GaAngInitSet = ...
            matlab.system.StringSet(...
            {'on','off'});
    end

    properties (Hidden, Nontunable)
        NumberOfLevels = 1
        IsFixedCoefs       = true
        StepStart = 1
        StepFinal = 1e-4
        AdaGradEta = 1e-2;
        AdaGradEps = 1e-8;
    end    

    properties (Hidden)
        GaAngInit = 'off'
    end

    properties (Logical)
        IsVerbose = false
    end
    
    properties (Access = protected)
        aprxError
    end
    
    methods
        
        function obj = NsoltDictionaryUpdateSgd(varargin)
            obj = obj@saivdr.dictionary.nsoltx.design.AbstNsoltDesigner(...
                varargin{:});
        end
        
        function [value, stcgrad] = costFcnAng(obj, lppufb, angles)
            clnlppufb = clone(lppufb);
            clnaprxer = clone(obj.aprxError);
            angles = reshape(angles,obj.sizeAngles);
            set(clnlppufb,'Angles',angles);
            nsamples_ = length(obj.TrainingImages);
            iImg = randi(nsamples_);
            % Gradient is evaluated for a randomly selected image 
            [~,stcgrad] = step(clnaprxer,clnlppufb,...
                obj.SparseCoefficients,obj.SetOfScales,iImg);
            % Cost is evaluated for all images
            value = step(clnaprxer,clnlppufb,... 
                obj.SparseCoefficients,obj.SetOfScales,[]);            
        end
        
        function value = costFcnMus(obj, lppufb, mus)
            clnlppufb = clone(lppufb);
            clnaprxer = clone(obj.aprxError);
            mus = 2*(reshape(mus,obj.sizeMus))-1;
            set(clnlppufb, 'Mus', mus);
            % Cost is evaluated for all images            
            value = step(clnaprxer,clnlppufb,...
                obj.SparseCoefficients,obj.SetOfScales,[]);
        end
        
        function value = isConstrained(~,~)
            value = false;
        end
        
        function [c,ce] = nonlconFcn(~,~,~)
            c  = [];
            ce = [];
        end
        
    end
    
    methods (Access=protected)
        
        function validatePropertiesImpl(obj)
            if isempty(obj.TrainingImages)
                error('Training images should be provided');
            elseif ~iscell(obj.TrainingImages)
                error('Training images should be provided as cell data');
            end
            %
            if obj.NumberOfLevels ~= 1
                error('NumberOfLevels should be 1.');   
            end
            %
            if ~obj.IsFixedCoefs
                error('IsFixedCoefs should be true.');
            end            
            %
            if ~obj.GradObj
                error('GradObj should be on.');
            end
        end
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@saivdr.dictionary.nsoltx.design.AbstNsoltDesigner(obj);
            s.aprxError = matlab.System.saveObject(obj.aprxError);
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            obj.aprxError = matlab.System.loadObject(s.aprxError);
            loadObjectImpl@saivdr.dictionary.nsoltx.design.AbstNsoltDesigner(obj,s,wasLocked);
        end
        
        function flag = isInactiveSubPropertyImpl(obj,propertyName)
            flag = isInactiveSubPropertyImpl...
                @saivdr.dictionary.nsoltx.design.AbstNsoltDesigner(...
                obj,propertyName);
        end
        
        function setupImpl(obj,lppufb_,options)
            setupImpl@saivdr.dictionary.nsoltx.design.AbstNsoltDesigner(...
                obj,lppufb_,options);
            import saivdr.dictionary.nsoltx.design.AprxErrorWithSparseRep
            obj.aprxError = AprxErrorWithSparseRep(...
                'TrainingImages', obj.TrainingImages,...
                'NumberOfLevels',obj.NumberOfLevels,...
                'GradObj',obj.GradObj,...
                'IsFixedCoefs',obj.IsFixedCoefs,...
                'Stochastic','on');
            if obj.IsVerbose
                fprintf('(NsoltDictionaryUpdateSgd) IsFixedCoefs = %d\n', ...
                    get(obj.aprxError,'IsFixedCoefs'));
            end            
        end
        
        function [ lppufb, fval, exitflag ] = stepImpl(obj,lppufb_,options)

            lppufb = clone(lppufb_);
            
            % Optimization of Mus
            lppufb = stepOptMus(obj,lppufb,options);
            
            % Initalization of Angs with GA
            if strcmp(obj.GaAngInit,'on')
                lppufb = stepGaAngInit_(obj,lppufb);
            end
            
            % Optimization of Angs
            initAngs = get(lppufb,'Angles');
            %
            problem.objective = @(x)costFcnAng(obj,lppufb,x);
            problem.x0        = initAngs;
            problem.options   = options;
            problem.step      = obj.Step;
            problem.stepStart = obj.StepStart;
            problem.stepFinal = obj.StepFinal;          
            problem.adaGradEta = obj.AdaGradEta;
            problem.adaGradEps = obj.AdaGradEps;
            
            %
            [optAngs, fval, exitflag] = obj.fminsgd_(problem);
            set(lppufb,'Angles',reshape(optAngs,obj.sizeAngles));
            
        end
        
    end
    
    methods (Access = private)
            
        % GA for Angs
        function lppufb = stepGaAngInit_(obj,lppufb)
            populationSize_   = 16;
            mfgscale          = pi/6;    % Standard deviation for mutation
            mfgshrink         = 1; 
            gaoptions = gaoptimset(...
                'PopulationSize',populationSize_,...
                'EliteCount',2,...
                'MutationFcn',{@mutationgaussian,mfgscale,mfgshrink},...
                'Generations', 32,...
                'StallGenLimit',16,...
                'PlotFcn',@gaplotbestf);
            %
            nAngs = prod(obj.sizeAngles); % Number of angles
            angles_ = get(lppufb,'Angles');
            initAngs = repmat(angles_(:).',populationSize_,1)...
                +2*pi*(rand(populationSize_,nAngs)-0.5);
            initAngs(1,:) = angles_(:).';
            gaoptions = ...
                gaoptimset(gaoptions,'InitialPopulation',initAngs);
            %
            problem.fitnessfcn = @(x)costFcnAngGa_(obj,lppufb,x);
            problem.nvars    = nAngs;
            problem.Aineq    = [];
            problem.bineq    = [];
            problem.Aeq      = [];
            problem.beq      = [];
            problem.lb       = [];
            problem.ub       = [];
            problem.nonlcon  = [];
            problem.rngstate = [];
            problem.intcon   = [];
            problem.solver   = 'ga';
            problem.options  = gaoptions;
            %
            optAngs = ga(problem);
            set(lppufb,'Angles',reshape(optAngs,obj.sizeAngles));
        end
        
        function value = costFcnAngGa_(obj, lppufb, angles)
            clnlppufb = clone(lppufb);
            clnaprxer = clone(obj.aprxError);
            angles = reshape(angles,obj.sizeAngles);
            set(clnlppufb,'Angles',angles);
            value = step(clnaprxer,clnlppufb,...
                obj.SparseCoefficients,obj.SetOfScales,[]);
        end        
    end
    
    methods (Access = private, Static = true)
            
        function [optAngs,fval,exitflag] = fminsgd_(problem)
            options_  = problem.options;
            if isempty(options_.MaxIter)
                maxIter_ = 400;
            else
                maxIter_  = options_.MaxIter;
            end
            isDisplay  = strcmp(options_.Display,'iter');
            optAngs    = problem.x0;
            step_      = problem.step;
            stepStart_ = problem.stepStart;            
            stepFinal_ = problem.stepFinal;
            agEta_     = problem.adaGradEta;
            agEps_     = problem.adaGradEps;
            %
            TolX_     = options_.TolX;
            TolFun_   = options_.TolFun;
            message   = '';
            %
            preFval  = problem.objective(optAngs);
            if strcmp(char(options_.PlotFcns),'optimplotfval')
                figPlotFval = findobj(get(groot,'Children'),...
                    'Name','OptimPlotFval',...
                    'type','figure');
                if isempty(figPlotFval)
                    figPlotFval = figure;
                    set(figPlotFval,'Name','OptimPlotFval');
                else
                    figure(figPlotFval)
                end
                optimValues.fval = preFval;
                optimValues.iteration = 0;
                options_.PlotFcns([],optimValues,'iter');
                drawnow
            end
            eta0 = stepStart_;
            etaf = stepFinal_;
            sumgrdAng = 0;
            for iItr = 1:maxIter_
                % Cost is evaluated for all images
                % Gradient is evaluated for a randomly selected image
                [fval,stcgrad] = problem.objective(optAngs);
                %
                if strcmp(char(options_.PlotFcns),'optimplotfval')
                    figure(figPlotFval)
                    optimValues.fval = fval;
                    optimValues.iteration = iItr;
                    stop = options_.PlotFcns([],optimValues,'iter');
                    if stop
                        break
                    end
                    drawnow
                end
                
                % Step value
                grdAngs = reshape(stcgrad,size(optAngs));
                if  strcmp(step_,'Constant')
                    eta  = eta0;
                elseif strcmp(step_,'Exponential')
                    eta  = eta0*(etaf/eta0)^(iItr/maxIter_);
                elseif strcmp(step_,'LineSearch')
                    fun = @(x) problem.objective(optAngs-x*grdAngs);
                    stepoptions = optimoptions('fminunc',...
                        'Algorithm','quasi-newton');
                    eta = fminunc(fun,eta0,stepoptions);
                    eta0 = eta;
                elseif  strcmp(step_,'AdaGrad')
                    grdAng02 = grdAngs.^2;
                    sumgrdAng = sumgrdAng + grdAng02;
                    sqgradAng = sqrt(sumgrdAng) + agEps_;
                    eta = agEta_./sqgradAng;
                else
                    eta = stepStart_/iItr;
                end

                % Update
                dltAngs = eta.*grdAngs;
                optAngs = optAngs - dltAngs;
                %
                if isDisplay
                    if iItr == 1
                        fprintf('         \t     \t          \t First-order\n')
                        fprintf('Iteration\t f(x)\t Step-size\t optimality\n')
                    end
                    fprintf('\t % 4d',iItr) 
                    fprintf('\t % 7.4g',fval)
                    fprintf('\t % 7.4g',norm(dltAngs(:)))
                    fprintf('\t % 7.4g',max(abs(stcgrad(:))))
                    fprintf('\n')
                end
                %
                difX = norm(dltAngs);
                if (difX < TolX_)
                    message = sprintf('TolX: %g < %g\n',difX,TolX_);
                    break
                end
                %
                difFun = norm(fval - preFval);
                if (difFun < TolFun_*(1+norm(stcgrad(:))))
                    message = sprintf('TolFun: %g < %g\n',difFun,TolFun_);
                    break
                end
            end
            if iItr >= maxIter_
                message = sprintf('MaxIter: %d\n',maxIter_);
            end
            %
            if isDisplay
                fprintf('\n')
                fprintf(message)
                fprintf('\t iterations: %d\n',iItr)
                fprintf('\t fval: %g\n',fval)
            end
            %
            exitflag = [];
        end
    end
end

