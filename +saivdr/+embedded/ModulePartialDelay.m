classdef ModulePartialDelay < matlab.System  & ...
        matlab.system.mixin.Nondirect & ...
        matlab.system.mixin.CustomIcon & ...
        matlab.system.mixin.Propagates %#codegen
    %MODULEPARTIALDELAY Partial delay moduel
    %
    % SVN identifier:
    % $Id: ModulePartialDelay.m 683 2015-05-29 08:22:13Z sho $
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
        StartIndexOfDelayChannel = 3;
        NumberOfDelayChannels = 2;
        NumberOfSymmetricChannels = 2;
        NumberOfAntisymmetricChannels = 2;
    end
    
    properties (Access=private)
        previousCd;
    end
    
    properties (DiscreteState)
    end
    
    methods
        function obj = ModulePartialDelay(varargin)
            % Support name-value pair arguments
            setProperties(obj,nargin,varargin{:});
        end
    end
    
    methods (Access=protected)
        function setupImpl(obj,~,~)
            obj.previousCd = zeros(obj.NumberOfDelayChannels,1);
        end
        
        function resetImpl(~)
        end

        function [cs, ca] = outputImpl(obj, cs, ca)
            coefs = [ cs; ca ];
            coefs(obj.StartIndexOfDelayChannel:obj.StartIndexOfDelayChannel+obj.NumberOfDelayChannels-1) = obj.previousCd;
            cs = coefs(1:obj.NumberOfSymmetricChannels);
            ca = coefs(obj.NumberOfSymmetricChannels+1:obj.NumberOfSymmetricChannels+obj.NumberOfAntisymmetricChannels);
        end
        
        function updateImpl(obj, cs, ca)
            coefs = [ cs; ca ];
            obj.previousCd = coefs(obj.StartIndexOfDelayChannel:obj.StartIndexOfDelayChannel+obj.NumberOfDelayChannels-1);
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
            icon = sprintf('Partial\nDelay');
        end
         
         function [ f1, f2 ] = isOutputFixedSizeImpl(~)
             f1 = true;
             f2 = true;
         end
 
         function [ s1, s2 ] = getOutputSizeImpl(obj)
             s1 = [ obj.NumberOfSymmetricChannels 1 ];
             s2 = [ obj.NumberOfAntisymmetricChannels 1 ];
         end
         
         function [ d1, d2 ] = getOutputDataTypeImpl(~)
             d1  = 'double';
             d2  = 'double';
         end
         
         function [ c1, c2 ] = isOutputComplexImpl(~)
             c1 = false;
             c2 = false;
         end
    end
end

