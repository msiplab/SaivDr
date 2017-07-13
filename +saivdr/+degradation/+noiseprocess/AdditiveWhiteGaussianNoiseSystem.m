classdef AdditiveWhiteGaussianNoiseSystem < ...
        saivdr.degradation.noiseprocess.AbstNoiseSystem %#codegen
    %ADDITIVEWHITEGAUSSIANOISESYSTEM Additive white Gaussian noise system
    %   
    % SVN identifier:
    % $Id: AdditiveWhiteGaussianNoiseSystem.m 683 2015-05-29 08:22:13Z sho $
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
        Mean = 0
        Variance = 0.01
    end
    
    methods
        % Constractor
        function obj = AdditiveWhiteGaussianNoiseSystem(varargin)
            setProperties(obj,nargin,varargin{:});
        end
    end
    
    methods (Access = protected)
        function output = stepImpl(obj,input)
            output = imnoise(input,'gaussian',obj.Mean,obj.Variance);
        end
    end
end
