classdef ModulePartialLineBuffer < matlab.System  & ...
        matlab.system.mixin.Nondirect & ...
        matlab.system.mixin.CustomIcon %#codegen
    %MODULEPARTIALLINEBUFFER Partial line buffer module
    %
    % SVN identifier:
    % $Id: ModulePartialLineBuffer.m 683 2015-05-29 08:22:13Z sho $
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
    
    properties (Nontunable, PositiveInteger)
        LineLength = 1;
        StartIndexOfDelayChannel = 3;
        NumberOfDelayChannels = 2;
        NumberOfSymmetricChannels = 2;
        NumberOfAntisymmetricChannels = 2;
    end
    
    properties (Access=private)
        previousCd;
        currentIndex;
    end
    
    properties (DiscreteState)
    end
    
    methods
        function obj = ModulePartialLineBuffer(varargin)
            % Support name-value pair arguments
            setProperties(obj,nargin,varargin{:});
        end
    end
    
    methods (Access=protected)
        function setupImpl(obj, ~, ~)
            obj.previousCd = zeros(obj.NumberOfDelayChannels,obj.LineLength);
            obj.currentIndex = 1;
        end
        
        function resetImpl(obj)
            obj.currentIndex = 1;
        end

        function [cs, ca] = outputImpl(obj, cs, ca)
            coefs = [ cs; ca ];
            coefs(obj.StartIndexOfDelayChannel:obj.StartIndexOfDelayChannel+obj.NumberOfDelayChannels-1) = ...
                obj.previousCd(:,obj.currentIndex);
            cs = coefs(1:obj.NumberOfSymmetricChannels);
            ca = coefs(obj.NumberOfSymmetricChannels+1:obj.NumberOfSymmetricChannels+obj.NumberOfAntisymmetricChannels);
        end
        
        function updateImpl(obj, cs, ca)
            coefs = [ cs; ca ];
            obj.previousCd(:,obj.currentIndex) = ...
                coefs(obj.StartIndexOfDelayChannel:obj.StartIndexOfDelayChannel+obj.NumberOfDelayChannels-1);
            obj.currentIndex = mod(obj.currentIndex,obj.LineLength)+1;
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
            icon = sprintf('Partial\nLine Buf.');
        end
    end
end

