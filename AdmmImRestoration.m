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
        StepMonitor
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
            for iCmp = 1:obj.NumberOfComponents
                [obj.y(:,iCmp),obj.scales(:,:,iCmp)] = step(obj.AdjOfSynthesizer,srcImg(:,:,iCmp),obj.NumberOfTreeLevels);
            end
            
            if ~isempty(obj.StepMonitor)
                reset(obj.StepMonitor);
            end
            obj.ypre = obj.y;
            obj.alpha = zeros(size(srcImg)); % TODO: ???????????
            err = Inf;
            
            while err > obj.Eps0 && obj.nItr < obj.MaxIter
                hx = zeros(size(srcImg));
                for iCmp = 1:obj.NumberOfComponents
                    hx(:,:,iCmp) = step(obj.Synthesizer,2*obj.y-obj.ypre,obj.scales);
                    obj.alpha(:,:,iCmp) = (obj.x(:,:,iCmp)-hx(:,:,iCmp)+obj.eta*obj.alpha(:,:,iCmp))/(obj.Lambda+obj.eta);
                end
                % optimize y
                obj.ypre = obj.y;
                hy = zeros(size(obj.y));
                for iCmp = 1:obj.NumberOfComponents
                    hy(:,iCmp) = step(obj.AdjOfSynthesizer,obj.alpha(:,:,iCmp),obj.NumberOfTreeLevels);
                    obj.y(:,iCmp) = AdmmImRestoration.softshrink_(...
                        obj.y(:,iCmp) + obj.eta*hy(:,iCmp), obj.eta);
                    %ypst = obj.y;
                end
                err = norm(obj.y(:)-obj.ypre(:))^2/norm(obj.y(:))^2;
                
                if ~isempty(obj.StepMonitor)
                    step(obj.StepMonitor,hx);
                end
                obj.nItr = obj.nItr + 1;
                
            end
            resImg = zeros(size(srcImg));
            for iCmp = 1:obj.NumberOfComponents
                resImg(:,:,iCmp) = step(obj.Synthesizer,obj.y(:,iCmp),obj.scales);
            end
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
    end
    
    methods (Static = true, Access = private)
        % Soft shrink
        function outputcf = softshrink_(inputcf,threshold)
            % Soft-thresholding shrinkage
            outputcf = max(1.0-threshold./abs(inputcf),0).*inputcf;
        end
        
    end
end