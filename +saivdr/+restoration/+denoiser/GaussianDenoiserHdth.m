classdef GaussianDenoiserHdth < saivdr.restoration.denoiser.AbstGaussianDenoiseSystem % codegen
    % GAUSSIANDENOIZERHDTH Gaussian denoizer with hard thresholding
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2018, Shogo MURAMATSU
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

    properties (Access = private)
        threshold
    end
    
    properties(DiscreteState)

    end
    
    methods
        function obj = GaussianDenoiserHdth(varargin)
            obj = obj@saivdr.restoration.denoiser.AbstGaussianDenoiseSystem(varargin{:});
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
        end

        function y = stepImpl(obj,u)
            % Implement algorithm. Calculate y as a function of input u and
            % discrete states.
            obj.threshold = obj.Sigma.^2;
            y = u;
            y((abs(u)-obj.threshold)<=0) = 0;
        end

        function resetImpl(obj)
            % Initialize / reset discrete-state properties
        end
    end
end

