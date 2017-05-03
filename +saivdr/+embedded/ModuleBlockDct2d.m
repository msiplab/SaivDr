classdef ModuleBlockDct2d < matlab.System & ...
        matlab.system.mixin.CustomIcon %#codegen
    %MODULEBLOCKDCT2D Block 2-d DCT module
    %
    % SVN identifier:
    % $Id: ModuleBlockDct2d.m 683 2015-05-29 08:22:13Z sho $
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
    
    properties  (Nontunable, PositiveInteger)
        NumberOfSymmetricChannels = 2;
        NumberOfAntisymmetricChannels = 2;
    end
    
    properties (Access=private)
        E0J = [ 1  1  1  1;
            1 -1 -1  1;
            1 -1  1 -1;
            1  1 -1 -1 ]/2;
    end
    
    properties (Access=private,Constant=true)
        NUMBER_OF_VERTICAL_DECIMATION_FACTOR = 2;
        NUMBER_OF_HORIZONAL_DECIMATION_FACTOR = 2;
        HALF_NUMBER_OF_DECIMATION = 2;
        NUMBER_OF_DECIMATION = 4;
    end
    
    properties (DiscreteState)
    end
    
    methods
        function obj = ModuleBlockDct2d(varargin)
            setProperties(obj,nargin,varargin{:});
        end
    end
    
    methods (Access=protected)
        function setupImpl(~,~)
        end
        
        function resetImpl(~)
        end
        
        function [cs, ca] = stepImpl(obj, u)
            coefs = obj.E0J*u(:);
            cs = zeros(obj.NumberOfSymmetricChannels,1);
            cs(1:obj.HALF_NUMBER_OF_DECIMATION) = coefs(1:obj.HALF_NUMBER_OF_DECIMATION);
            ca = zeros(obj.NumberOfAntisymmetricChannels,1);
            ca(1:obj.HALF_NUMBER_OF_DECIMATION) = coefs(obj.HALF_NUMBER_OF_DECIMATION+1:obj.NUMBER_OF_DECIMATION);
        end
        
        function N = getNumInputsImpl(~)
            % Specify number of System inputs
            N = 1; 
        end
        
        function n1 = getInputNamesImpl(~)
            n1 = 'X';
        end
        
        function N = getNumOutputsImpl(~)
            % Specify number of System outputs
            N = 2; 
        end
        
        function [n1,n2] = getOutputNamesImpl(~)
            n1 = 'Cs';
            n2 = 'Ca';
        end
        
        
        function icon = getIconImpl(~)
            coder.extrinsic('sprintf')
            icon = sprintf('Block\n2-D DCT');
        end
    end
end
