classdef PoissonNoiseSystem < ...
        saivdr.degradation.noiseprocess.AbstNoiseSystem %#codegen
    %POISSONNOISESYSTEM Poisson noise system
    %   
    % SVN identifier:
    % $Id: PoissonNoiseSystem.m 683 2015-05-29 08:22:13Z sho $
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
    methods
        % Constractor
        function obj = PoissonNoiseSystem(varargin)
            setProperties(obj,nargin,varargin{:});
        end
    end
    
    methods (Access = protected)
        function output = stepImpl(~,input)
            output = imnoise(input,'poisson');
        end
    end
end
