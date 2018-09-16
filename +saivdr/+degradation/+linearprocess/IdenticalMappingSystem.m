classdef IdenticalMappingSystem < ...
        saivdr.degradation.linearprocess.AbstLinearSystem %#codegen
    %BLURSYSTEM Bluring process
    %   
    % Requirements: MATLAB R2015b
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
    
    methods
        % Constractor
        function obj = IdenticalMappingSystem(varargin)
            obj = ...
                obj@saivdr.degradation.linearprocess.AbstLinearSystem(...
                varargin{:});
        end
        
    end
    
    methods (Access = protected)
        
        function flag = isInactiveSubPropertyImpl(~,~)
            flag = false;
        end        
        
        function output = normalStepImpl(~,input)
            output = input;
        end
        
        function output = adjointStepImpl(~,input)
            output = input;
        end

        function originalDim = getOriginalDimension(~,ovservedDim)
            originalDim = ovservedDim;
        end        
      
    end
    
end
