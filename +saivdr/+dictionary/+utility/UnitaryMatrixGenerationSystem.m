classdef UnitaryMatrixGenerationSystem < matlab.System %#codegen
    %UNITARYMATRIXGENERATIONSYSTEM Unitary matrix generator
    %
    % Requirements: MATLAB R2020b
    %
    % Copyright (c) 2021, Shogo MURAMATSU
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
        NumberOfDimensions
    end
      
    methods
        function obj = UnitaryMatrixGenerationSystem(varargin)
            % Support name-value pair arguments
            setProperties(obj,nargin,varargin{:});
        end
    end
    
    methods (Access = protected)
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@matlab.System(obj);
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            loadObjectImpl@matlab.System(obj,s,wasLocked);
        end
        
        function setupImpl(obj,weights,~,~)
            if isempty(obj.NumberOfDimensions)
                obj.NumberOfDimensions = (1+sqrt(1+8*length(weights)))/2;
            end
        end
        
        function resetImpl(obj)        
        end
        
        function validateInputsImpl(~,~,mus,~)
            if ~isempty(mus) && any(abs(mus(:))~=1)
                error('All entries of mus must be 1 or -1.');
            end
        end
        
        function matrix = stepImpl(obj,weights,mus)
            
            if isempty(weights)
                matrix = diag(mus);
            else
                % Normal mode
                matrix = obj.stepNormal_(weights,mus);
            end
        end

        function N = getNumInputsImpl(~)
            N = 2;
        end

        function N = getNumOutputsImpl(~)
            N = 1;
        end
    end
    
    methods (Access = private)
        
        function matrix = stepNormal_(obj,weights,mus)
            nDim_ = obj.NumberOfDimensions;
            matrix = eye(nDim_);
            matrix = diag(mus)*matrix;
        end
    end
end
