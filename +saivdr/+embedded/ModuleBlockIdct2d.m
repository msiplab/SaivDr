classdef ModuleBlockIdct2d < matlab.System & ...
        matlab.system.mixin.CustomIcon & ...
        matlab.system.mixin.Propagates %#codegen
    %MODULEBLOCKIDCT Block 2-D IDCT module
    %
    % SVN identifier:
    % $Id: ModuleBlockIdct2d.m 683 2015-05-29 08:22:13Z sho $
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
    end

    properties (Access=private)
        JR0 = [  1  1  1  1 ;
            1 -1 -1  1 ;
            1 -1  1 -1 ;
            1  1 -1 -1 ]/2;
        NumberOfSymmetricChannels;
        NumberOfAntisymmetricChannels;
        dataType = 'double';
    end

    properties(Access=private,Constant=true)
        NUMBER_OF_VERTICAL_DECIMATION_FACTOR = 2;
        NUMBER_OF_HORIZONTAL_DECIMATION_FACTOR = 2;
        HALF_NUMBER_OF_DECIMATION = 2;
    end
    
    properties (DiscreteState)
    end
    
    methods
        function obj = ModuleBlockIdct2d(varargin)
            setProperties(obj,nargin,varargin{:});
        end
    end
    
    methods (Access=protected)
        function setupImpl(obj,cs,ca)
            obj.NumberOfSymmetricChannels = length(cs);
            obj.NumberOfAntisymmetricChannels = length(ca);
        end
        
        function resetImpl(~)
        end
        
        function y = stepImpl(obj, cs, ca)
            y = reshape(obj.JR0 * [ cs(1:obj.HALF_NUMBER_OF_DECIMATION) ; ca(1:obj.HALF_NUMBER_OF_DECIMATION) ],...
                obj.NUMBER_OF_VERTICAL_DECIMATION_FACTOR,...
                obj.NUMBER_OF_HORIZONTAL_DECIMATION_FACTOR);
        end
        
        function N = getNumInputsImpl(~)
            % Specify number of System inputs
            N = 2; 
        end
        
        function [n1,n2] = getInputNamesImpl(~)
            n1 = 'Cs';
            n2 = 'Ca';
        end
        
        function N = getNumOutputsImpl(~)
            % Specify number of System outputs
            N = 1; 
        end
        
        function n1 = getOutputNamesImpl(~)
            n1 = 'X';
        end        
        
        function icon = getIconImpl(~)
            coder.extrinsic('sprintf')
            icon = sprintf('Block\n2-D IDCT');
        end

        function fixedout = isOutputFixedSizeImpl(~)
            fixedout = true;
        end

        function sizeout = getOutputSizeImpl(obj)
            sizeout = [obj.NUMBER_OF_VERTICAL_DECIMATION_FACTOR obj.NUMBER_OF_HORIZONTAL_DECIMATION_FACTOR ];
        end
        
        function dataout = getOutputDataTypeImpl(~)
            dataout = 'double';
        end
        
        function cplxout = isOutputComplexImpl(~)
            cplxout = false;
        end

    end
end

