classdef GaussianDenoiserSfth < saivdr.restoration.denoiser.AbstGaussianDenoiseSystem % codegen
    % GAUSSIANDENOIZERSFTH Gaussian denoizer with soft thresholding
    %

    properties (Nontunable,Access = private)
        threshold
    end
    
    properties(DiscreteState)

    end

    % Pre-computed constants
    properties(Access = private)

    end

    
    methods
        function obj = GaussianDenoiserSfth(varargin)
            obj = obj@saivdr.restoration.denoiser.AbstGaussianDenoiseSystem(varargin{:})
            setProperties(obj,nargin,varargin{:});
        end
            
    end
    
    methods(Access = protected)
        

        function s = saveObjectImpl(obj)
            s = saveObjectImpl@saivdr.restoration.denoiser.AbstGaussianDenoiseSystem(obj);
            %s.Synthesizer = matlab.System.saveObject(obj.Synthesizer);
            s.threshold = obj.threshold;
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            obj.threshold = s.threshold;
            %obj.Synthesizer = matlab.System.loadObject(s.Synthesizer);
            loadObjectImpl@saivdr.restoration.denoiser.AbstGaussianDenoiseSystem(obj,s,wasLocked);
        end
        
        
        function setupImpl(obj)
            % Perform one-time calculations, such as computing constants
            obj.threshold = obj.Sigma.^2;
        end

        function y = stepImpl(obj,u)
            % Implement algorithm. Calculate y as a function of input u and
            % discrete states.
            nc = abs(u)-obj.threshold;
            nc = (nc+abs(nc))/2;
            y = sign(u).*nc;
        end

        function resetImpl(obj)
            % Initialize / reset discrete-state properties
        end
    end
end

