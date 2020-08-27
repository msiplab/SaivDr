classdef SurfaceExtractionSystem < ...
        saivdr.degradation.linearprocess.AbstLinearSystem %#codegen
    %SURFACEEXTRACTIONSYSTEM Surface extraction process
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
        function obj = SurfaceExtractionSystem(varargin)
            obj.LambdaMax = 1;            
        end
        
    end
    
    methods (Access = protected)
        
        function flag = isInactiveSubPropertyImpl(~,~)
            flag = false;
        end        
        
        function output = normalStepImpl(~,input)
            output = input(:,:,1);
        end
        
        function output = adjointStepImpl(~,input)
            output = cat(3,input,zeros(size(input),'like',input));
        end

        function originalDim = getOriginalDimension(~,ovservedDim)
            originalDim = [ovservedDim(1:2) 2];
        end        
      
    end
    
end
