classdef DalImRestoration < matlab.System
    properties (Nontunable)
        Synthesizer
        AdjOfSynthesizer
        LinearProcess
        NumberOfTreeLevels = 1
    end
    
    properties (Hidden,Nontunable)
        NumberOfComponents
    end
    
    properties (Logical)
        UseParallel = false;
    end
    
    properties (PositiveInteger)
        MaxIter = 100
    end
    
    properties (Access = protected,Nontunable)
        AdjLinProcess
    end
    
    properties
        Eps0 = 1e-6
        Lambda = 0.00185 % TODO:
    end
    
    properties (Access = private)
        nItr
        gamma
        scales
        y
        eta0
        eta
    end
    
    methods
        function obj = DalImRestoration(varargin)
            setProperties(obj,nargin,varargin{:})
        end
    end
    
    methods (Access = protected)
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@matlab.System(obj);
            % TODO:
        end
        
        function loadObjectImpl(obj, s, wasLocked)
            loadObjectImpl@matlab.System(obj,s,wasLocked);
        end
        
%         function validatePropertiesImpl(obj)
%         end

        function setupImpl(obj,srcImg)
        end
        
        function resetImpl(~)
        end
        
        function [resImg,coefvec,scales] = stepImpl(obj,srcImg)
            % Initialization
            obj.nItr = 0;
            obj.eta0 = 0.01/obj.Lambda;
            obj.gamma = 1; % TODO:???????
            [obj.y,obj.scales] = step(obj.Analyzer,srcImg,obj.NumberOfTreeLevels);
            alpha = zeros(size(srcImg)); % TODO: ???????????
            err = Inf;
                      
            while err > obj.Eps0 && obj.nItr < obj.MaxIter
                % setup optimfunc
                obj.eta = 2^obj.nItr*obj.eta0;
                
                objFunc = getObjectiveFunction(obj);
                outputFunc = getOutputFunction(obj);
                
                opt = optimoptions(@fminunc,...
                    'OutputFcn',outputFunc,...
                    'SpecifyObjectiveGradient',false,...
                    'UseParallel',true);
                
                alpha = fminunc(objFunc,alpha,opt);
                
                % optimize y
                ypre = obj.y;
                obj.y = hogehoge_(alpha);
                ypst = obj.y;
                err = norm(ypst(:)-ypre(:))^2/norm(ypst(:))^2;
                
                obj.nItr = obj.nItr + 1;
            end
            resImg = step(obj.Synthesizer,obj.y,obj.scales);
            coefvec = obj.y;
            scales = obj.scales;
        end
        
        function N = getNumInputsImpl(~)
            N = 1;
        end
        
        function N = getNumOutputsImpl(~)
            N = 1;
        end
    end
    
    methods (Access = private)
        function func = getObjectiveFunction(obj)
            function [value, grad] = objFunc(alpha)
                ccfl = norm(alpha,2)^2/2 - real(dot(alpha,obj.y));
                py = hogehoge_(obj,alpha);
                value = ccfl + norm(py,2)^2/(2*obj.eta);
                if nargout > 1
                    gccfl = alpha - obj.y;
                    grad = -gccfl + step(obj.Synthesizer,py,obj.scales);
                end
            end
            func = @objFunc;
        end
        
        function func = getOutputFunction(obj) % stop condition
            function stopFlag = outputFunc(alpha,optimValues,~)
                gf = norm(optimValues.gradient);
                stopFlag = gf <= sqrt(obj.gamma/obj.eta)*norm(hogehoge_(obj,alpha)-obj.y,2)^2;
            end
            func = @outputFunc;
        end
        
        % TODO: ????????
        function outputcf = hogehoge_(obj,alpha)
            v = step(obj.AdjOfSynthesizer,alpha,obj.NumberOfTreeLevels);
            outputcf = DalImRestoration.softshrink_(obj.y+obj.eta*v,obj.Lambda*obj.eta);
        end
    end
    
    methods (Static = true, Access = private)
        % Soft shrink
        function outputcf = softshrink_(inputcf,threshold)
            % Soft-thresholding shrinkage
            ln = abs(inputcf);
            
            outputcf = zeros(size(inputcf));
            for idx = 1:length(inputcf)
                if ln(idx) > threshold
                    outputcf(idx) = (ln(idx)-threshold)/ln(idx)*inputcf(idx);
                end
            end
%             outputcf = (ln > threshold).*(ln-threshold)./ln.*inputcf;
        end

    end
end