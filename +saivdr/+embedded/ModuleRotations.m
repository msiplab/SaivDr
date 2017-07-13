classdef ModuleRotations < matlab.System & ...
        matlab.system.mixin.CustomIcon %#codegen
    %MODULEROTATIONS Rotation module
    %
    % SVN identifier:
    % $Id: ModuleRotations.m 683 2015-05-29 08:22:13Z sho $
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
        MatrixW = 1;
        MatrixU = 1;
    end
    
    properties (DiscreteState)
    end
    
    methods
        function obj = ModuleRotations(varargin)
            % Support name-value pair arguments
            setProperties(obj,nargin,varargin{:});
        end
    end
    
    methods (Access=protected)
        function setupImpl(~,~,~,~)
        end
        
        function resetImpl(~)
        end
        
        function [cs, ca] = stepImpl(obj, cs, ca, isTermination)
            if ~isempty(obj.MatrixW) || obj.MatrixW ~= 1
                cs = obj.MatrixW*cs;
            end
            if isTermination
                ca = -ca;
            elseif ~isempty(obj.MatrixU) || obj.MatrixU ~= 1
                ca = obj.MatrixU*ca;
            end
        end
        
        function N = getNumInputsImpl(~)
            N = 3; 
        end
        
        function [n1,n2,n3] = getInputNamesImpl(~)
            n1 = 'Cs';
            n2 = 'Ca';
            n3 = 'isTerm.';
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
            icon = sprintf('Rotations');
        end
    end
end

