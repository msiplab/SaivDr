classdef ModuleButterfly < matlab.System & ...
        matlab.system.mixin.CustomIcon %#codegen
    %MODULEBUTTERFLY Butterfly module
    %
    % SVN identifier:
    % $Id: ModuleButterfly.m 683 2015-05-29 08:22:13Z sho $
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
        NumberOfSymmetricChannels = 2;
        NumberOfAntisymmetricChannels = 2;
        minChs;
    end
    
    properties (DiscreteState)
    end
    
    methods
        function obj = ModuleButterfly(varargin)
            % Support name-value pair arguments
            setProperties(obj,nargin,varargin{:});
        end
    end
    
    methods (Access=protected)
        function setupImpl(obj, cs, ca)
            obj.NumberOfSymmetricChannels = length(cs);
            obj.NumberOfAntisymmetricChannels = length(ca);
            obj.minChs = min(obj.NumberOfSymmetricChannels,...
                obj.NumberOfAntisymmetricChannels);
        end
        
        function resetImpl(~)
        end
        
        function [cs, ca] = stepImpl(obj, cs, ca)
            tmpCs = cs;
            tmpCa = ca;
            cs(1:obj.minChs) = ...
                (tmpCs(1:obj.minChs) + tmpCa(1:obj.minChs))/sqrt(2);
            ca(1:obj.minChs) = ...
                (tmpCs(1:obj.minChs) - tmpCa(1:obj.minChs))/sqrt(2);
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
            N = 2; 
        end
        
        function [n1,n2] = getOutputNamesImpl(~)
            n1 = 'Cs';
            n2 = 'Ca';
        end        
        
        function icon = getIconImpl(~)
            coder.extrinsic('sprintf')
            icon = sprintf('Butterfly');
        end
    end
end

