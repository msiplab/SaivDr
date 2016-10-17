classdef AdmmImRestoration < matlab.System
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
        x
        y
        ypre
        alpha
        eta0
        eta
    end
    
    methods
        function obj = AdmmImRestoration(varargin)
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
            obj.eta = obj.eta0;
            obj.gamma = 1; % TODO:???????
            obj.x = srcImg;
            [obj.y,obj.scales] = step(obj.AdjOfSynthesizer,srcImg,obj.NumberOfTreeLevels);
            obj.ypre = obj.y;
            obj.alpha = zeros(size(srcImg)); % TODO: ???????????
            err = Inf;
            
            while err > obj.Eps0 && obj.nItr < obj.MaxIter
                hx = step(obj.Synthesizer,2*obj.y-obj.ypre,obj.scales);
                obj.alpha = (obj.x-hx+obj.eta*obj.alpha)/(obj.Lambda+obj.eta);
                
                % optimize y
                obj.ypre = obj.y;
                hy = step(obj.AdjOfSynthesizer,obj.alpha,obj.NumberOfTreeLevels);
                obj.y = AdmmImRestoration.softshrink_(...
                    obj.y + obj.eta*hy, obj.eta);
                %ypst = obj.y;
                err = norm(obj.y(:)-obj.ypre(:))^2/norm(obj.y(:))^2;
                
                obj.nItr = obj.nItr + 1;
            end
            resImg = step(obj.Synthesizer,obj.y,obj.scales);
            coefvec = obj.y;
            scales = obj.scales;
        end
        
        function N = getNumInputsImpl(~)
            N = 1;
        end
        
%         function N = getNumOutputsImpl(~)
%             N = 1;
%         end
    end
    
    methods (Access = private)
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