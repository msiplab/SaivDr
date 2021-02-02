classdef (Abstract) AbstGaussianDenoiseSystem < matlab.System %#codegen
    % ABSTGAUSSIANDENOIZESYSTEM Abstract class of Gaussian Denoiser
    %
    % Requirements: MATLAB R2018a
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
    
    properties
        Sigma
    end    
    
    properties (Nontunable, Logical)
        UseGpu = false
    end
    
    methods
        
        % Constractor
        function obj = AbstGaussianDenoiseSystem(varargin)
            setProperties(obj,nargin,varargin{:})
            %if ~license('checkout','distrib_computing_toolbox')
            %    obj.UseGpu = false;
            %elseif gpuDeviceCount() < 1
            %    obj.UseGpu = false;
            %end
        end
        
    end
    
    methods (Access = protected, Sealed = true)
        
        function N = getNumInputsImpl(~)
            % Specify number of System inputs
            N = 1; % Because stepImpl has one argument beyond obj
        end
        
        function N = getNumOutputsImpl(~)
            % Specify number of System outputs
            N = 1; % Because stepImpl has one output
        end
    end
    
end
