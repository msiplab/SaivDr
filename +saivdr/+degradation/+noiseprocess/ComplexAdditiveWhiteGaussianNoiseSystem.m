classdef ComplexAdditiveWhiteGaussianNoiseSystem < ...
        saivdr.degradation.noiseprocess.AbstNoiseSystem %#codegen
    %ADDITIVEWHITEGAUSSIANOISESYSTEM Additive white Gaussian noise system
    %   
    % SVN identifier:
    % $Id: AdditiveWhiteGaussianNoiseSystem.m 683 2015-05-29 08:22:13Z sho $
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
        Mean = 0
        Variance = 0.01
    end
    
    methods
        % Constractor
        function obj = ComplexAdditiveWhiteGaussianNoiseSystem(varargin)
            setProperties(obj,nargin,varargin{:});
        end
    end
    
    methods (Access = protected)
        function output = stepImpl(obj,input)
                output = input + obj.Mean + sqrt(obj.Variance/2)...
                    *(randn(size(input)) + 1i*randn(size(input)));
        end
    end
end
